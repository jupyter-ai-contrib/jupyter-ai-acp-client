"""Render the cells named by a notebook attachment as prompt text.

A ``NotebookAttachment`` carries ``cells``: the ids of the cells the user
dragged into the chat. This module resolves those ids to their content and
renders it as one text content block, so the agent receives the cells
themselves rather than a link to the whole notebook.

The ids are resolved from the live document first. ``jupyter_ydoc`` gives a
cell an id when it loads a notebook and drops it again on save when the
notebook is nbformat 4.4 or older, so for such notebooks the ids in an
attachment exist only in the running document. The file on disk is the
fallback, which is right for nbformat 4.5+ notebooks and for notebooks that
nobody has open.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from acp.schema import TextContentBlock
from jupyterlab_chat.models import NotebookAttachment

MAX_SOURCE_CHARS = 20_000
"""Source beyond this is cut, so one pathological cell cannot dominate the prompt."""

MAX_OUTPUT_CHARS = 4_000
"""Likewise for a cell's combined text outputs."""

MAX_TOTAL_CHARS = 60_000
"""The budget for one attachment. The frontend puts every cell selected in a
drag into one attachment; once its rendered cells reach this many
characters, the rest are listed by position and id only. Each attachment in
a prompt has its own budget."""

LIVE_LOOKUP_TIMEOUT = 10.0
"""Seconds allowed for the live-document lookup before the file is used."""

# Wording sent to the agent, as module-level constants so it can be adjusted
# in one place.
LIVE_UNAVAILABLE = (
    "the live document could not be consulted (jupyter_ai_tools is not available)"
)
LIVE_NO_RTC = "real-time collaboration is not enabled, so there is no live document"
LIVE_NOT_OPEN = "no open copy of the notebook was found on the server"
LIVE_FAILED = "the live document lookup failed"
NO_IDS_ON_DISK = "its file on disk stores no cell ids (nbformat 4.4 or older)"
UNREADABLE = "its file could not be read ({error})"

HEADER = "The user attached {count} from {name}, read from the {origin}:"
CELL_HEADER = "Cell {position} of {total} ({kind}{extra}, id={id}):"
NOT_FOUND = "(cell {id} was not found in the {origin})"
OVER_BUDGET = (
    "({count} not shown, to keep the prompt short: {cells}. Read them from the "
    "notebook.)"
)
HELD_OUTPUTS = "[{count} held by the server's outputs service, not shown]"
UNRESOLVED = (
    "The user attached {count} from {name}, but they could not be resolved: "
    "{why}. Read the notebook at {path} (with the notebook tools, e.g. "
    "read_notebook, if you have them) and ask which cell they mean if it is "
    "ambiguous."
)
FOOTER = (
    "(Full notebook: {path}. Read it, with the notebook tools if you have them, "
    "for the cells around these.)"
)

_ANSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


class NotebookUnreadable(Exception):
    """The notebook file could not be read, or is not a notebook."""


@dataclass
class _Cell:
    cell_type: str
    source: str
    outputs: list
    execution_count: Any


@dataclass
class _Resolved:
    origin: str
    """Where the cells came from, in words meant for the agent."""

    language: str
    """The notebook's kernel language, for the code fence."""

    total: int
    """How many cells the notebook has."""

    by_id: dict[str, tuple[int, _Cell]]
    """The wanted cells that were found, as ``id -> (index, cell)``."""


def _field(obj: Any, name: str) -> Any:
    """Read a field from a dataclass or a plain dict.

    The chat model's ``get_attachments()`` (``YChat`` and ``WsChatModel``
    alike) builds ``NotebookAttachment(**att_dict)`` without converting the
    nested cells, so at runtime ``cells`` holds dicts, while code
    constructing attachments directly passes ``NotebookAttachmentCell``
    instances. Both shapes are accepted; the dict case can go once
    jupyter-chat converts the nested cells itself.
    """
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj.get(name)
    return getattr(obj, name, None)


def _to_py(value: Any) -> Any:
    """A pycrdt container as plain Python; anything else unchanged."""
    if hasattr(value, "to_py"):
        return value.to_py()
    return value


def _as_text(value: Any) -> str:
    """Notebook text (a string, a list of strings or a pycrdt Text) as one string."""
    value = _to_py(value)
    if value is None:
        return ""
    if isinstance(value, list):
        return "".join(item for item in value if isinstance(item, str))
    return str(value)


def _language(metadata: Any, default: str = "python") -> str:
    """The notebook's kernel language, from its ``language_info``."""
    metadata = _to_py(metadata)
    if isinstance(metadata, dict):
        info = metadata.get("language_info")
        if isinstance(info, dict) and info.get("name"):
            return str(info["name"])
    return default


def _cell(raw: Any) -> _Cell:
    """A cell record from a notebook dict or a pycrdt ``Map``."""
    outputs = _to_py(raw.get("outputs"))
    return _Cell(
        cell_type=str(raw.get("cell_type") or ""),
        source=_as_text(raw.get("source")),
        outputs=outputs if isinstance(outputs, list) else [],
        execution_count=_to_py(raw.get("execution_count")),
    )


def _count(n: int, noun: str) -> str:
    return f"{n} {noun}" if n == 1 else f"{n} {noun}s"


async def cells_from_live_document(
    abs_path: str, wanted: set[str], log: logging.Logger
) -> tuple[_Resolved | None, str | None]:
    """Resolve ``wanted`` cell ids from the notebook's live document.

    Returns ``(resolved, None)`` when the notebook is open on the server and
    ``(None, reason)`` otherwise, ``reason`` being one of the ``LIVE_*``
    strings.

    The lookup goes through ``jupyter_ai_tools``, which is not a dependency
    of this package but is installed next to it by jupyter-ai. It is skipped
    when that package reports no real-time collaboration provider, since
    there is then no live document to find. Its ``get_jupyter_ydoc`` returns
    None unless the document's room already exists, so a notebook nobody has
    open is never loaded as a side effect. ``ycells`` is read directly
    because ``YNotebook.get()`` and ``get_cell()`` strip the very ids being
    matched.
    """
    try:
        from jupyter_ai_tools import utils
    except Exception:  # not installed, or its own imports failed
        log.debug(
            "jupyter_ai_tools is unavailable; not consulting the live document",
            exc_info=True,
        )
        return None, LIVE_UNAVAILABLE

    rtc_available = getattr(utils, "rtc_available", None)
    try:
        if callable(rtc_available) and not rtc_available():
            return None, LIVE_NO_RTC
    except Exception:
        log.debug("rtc_available() failed; trying the lookup anyway", exc_info=True)

    async def lookup() -> Any:
        file_id = await utils.get_file_id(abs_path)
        return await utils.get_jupyter_ydoc(file_id) if file_id else None

    try:
        ydoc = await asyncio.wait_for(lookup(), LIVE_LOOKUP_TIMEOUT)
    except Exception:
        log.warning(
            "Live-document lookup failed for %r; reading the file instead",
            abs_path,
            exc_info=True,
        )
        return None, LIVE_FAILED
    ycells = getattr(ydoc, "ycells", None)
    if ycells is None:
        return None, LIVE_NOT_OPEN

    by_id: dict[str, tuple[int, _Cell]] = {}
    for index in range(len(ycells)):
        ycell = ycells[index]
        cell_id = ycell.get("id")
        if cell_id in wanted:
            by_id[cell_id] = (index, _cell(ycell))

    # jupyter_ydoc keeps the notebook-level metadata in the private ``_ymeta``
    # map. A public name is tried first, and a miss only costs the fence
    # language.
    meta = _to_py(getattr(ydoc, "ymeta", None) or getattr(ydoc, "_ymeta", None))
    metadata = meta.get("metadata") if isinstance(meta, dict) else None
    return (
        _Resolved(
            origin="live document",
            language=_language(metadata),
            total=len(ycells),
            by_id=by_id,
        ),
        None,
    )


def cells_from_file(path: Path, wanted: set[str]) -> _Resolved | None:
    """Resolve ``wanted`` cell ids from the notebook file on disk.

    Right for nbformat 4.5+ notebooks, whose cells carry ids, and for
    notebooks that are not open. Returns None when no cell in the file has
    an id at all, so the caller can say why nothing matched instead of
    reporting every cell as missing. Raises ``NotebookUnreadable`` when the
    file cannot be read or is not a notebook.
    """
    try:
        notebook = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:  # ValueError: bad JSON or encoding
        raise NotebookUnreadable(str(error)) from error
    cells = notebook.get("cells") if isinstance(notebook, dict) else None
    if not isinstance(cells, list):
        raise NotebookUnreadable("it has no list of cells")

    by_id: dict[str, tuple[int, _Cell]] = {}
    has_ids = False
    for index, raw in enumerate(cells):
        if not isinstance(raw, dict) or not raw.get("id"):
            continue
        has_ids = True
        cell_id = str(raw["id"])
        if cell_id in wanted:
            by_id[cell_id] = (index, _cell(raw))
    if not has_ids:
        return None
    return _Resolved(
        origin="file on disk",
        language=_language(notebook.get("metadata")),
        total=len(cells),
        by_id=by_id,
    )


def _fence(text: str) -> str:
    """A code fence one backtick longer than any backtick run in ``text``.

    A markdown cell that itself contains a fence would otherwise close the
    block early. CommonMark closes a fence only with a run at least as long
    as the opening one.
    """
    longest = max((len(run) for run in re.findall(r"`+", text)), default=0)
    return "`" * max(3, longest + 1)


def _is_placeholder(output: dict) -> bool:
    """An output that only points at where its content is stored."""
    metadata = output.get("metadata")
    return (
        isinstance(metadata, dict)
        and bool(metadata.get("url"))
        and not output.get("data")
        and not _as_text(output.get("text"))
        and not output.get("traceback")
    )


def _render_outputs(outputs: list) -> list[str]:
    """A cell's text outputs, fenced, plus a note for every output that had
    other representations.

    Only ``text/plain`` is inlined (with stream text and error tracebacks).
    Every other representation, images and HTML included, is listed so the
    agent knows what it is not seeing. So are placeholders left in the live
    document by jupyter_server_documents' outputs service, which carry only
    a ``metadata.url`` where the output can be fetched.
    """
    texts: list[str] = []
    omitted_outputs = 0
    omitted_mimes: set[str] = set()
    held_outputs = 0
    for output in outputs:
        if not isinstance(output, dict):
            continue
        kind = output.get("output_type")
        if _is_placeholder(output):
            held_outputs += 1
        elif kind == "stream":
            texts.append(_as_text(output.get("text")))
        elif kind == "error":
            traceback = output.get("traceback")
            if isinstance(traceback, list):
                texts.append("\n".join(str(line) for line in traceback))
            else:
                texts.append(_as_text(traceback) or _as_text(output.get("evalue")))
        elif kind in ("execute_result", "display_data"):
            data = output.get("data")
            if not isinstance(data, dict):
                continue
            if "text/plain" in data:
                texts.append(_as_text(data["text/plain"]))
            others = [mime for mime in data if mime != "text/plain"]
            if others:
                omitted_outputs += 1
                omitted_mimes.update(others)

    lines: list[str] = []
    text = _ANSI.sub("", "\n".join(t.rstrip("\n") for t in texts if t.strip()))
    if text:
        if len(text) > MAX_OUTPUT_CHARS:
            text = text[:MAX_OUTPUT_CHARS] + "\n... [output truncated]"
        fence = _fence(text)
        lines.append(f"Output:\n{fence}\n{text}\n{fence}")
    if omitted_outputs:
        lines.append(
            f"[{_count(omitted_outputs, 'output')} with non-text representations "
            f"omitted: {', '.join(sorted(omitted_mimes))}]"
        )
    if held_outputs:
        lines.append(HELD_OUTPUTS.format(count=_count(held_outputs, "output")))
    return lines


def _render_cell(
    cell_id: str, position: int, total: int, cell: _Cell, spec: Any, language: str
) -> list[str]:
    """One attached cell: a header line, the fenced source, then its outputs."""
    kind = cell.cell_type or str(_field(spec, "input_type") or "code")
    selected = _field(_field(spec, "selection"), "content")
    if selected:
        source, extra = str(selected), ", selection only"
    else:
        source, extra = cell.source, ""
    if kind == "code":
        count = cell.execution_count
        if isinstance(count, float) and count.is_integer():
            count = int(count)  # pycrdt hands JSON numbers back as floats
        extra += f", last run [{count}]" if count else ", never run"
    if len(source) > MAX_SOURCE_CHARS:
        source = source[:MAX_SOURCE_CHARS] + "\n... [source truncated]"
    fence = _fence(source)
    fence_language = {"code": language, "markdown": "markdown"}.get(kind, "")
    lines = [
        CELL_HEADER.format(
            position=position, total=total, kind=kind, extra=extra, id=cell_id
        ),
        f"{fence}{fence_language}\n{source.rstrip()}\n{fence}",
    ]
    lines.extend(_render_outputs(cell.outputs))
    return lines


async def render_cell_block(
    attachment: NotebookAttachment, abs_path: Path, log: logging.Logger
) -> TextContentBlock | None:
    """Render the cells named by ``attachment`` as one text content block.

    ``abs_path`` is the notebook's absolute path, already checked to lie
    inside the server root; it is given to the agent because
    ``attachment.value`` is relative to the server root, which need not be
    the agent's working directory. Returns None when the attachment names no
    cell with an id, so the caller can fall back to a resource link. When
    the ids cannot be resolved from either source the block says why, rather
    than reporting a cell that exists as missing.
    """
    specs: dict[str, Any] = {}
    for spec in attachment.cells or []:
        cell_id = _field(spec, "id")
        if cell_id and str(cell_id) not in specs:
            specs[str(cell_id)] = spec
    if not specs:
        return None

    name = attachment.value or abs_path.name
    count = _count(len(specs), "cell")
    resolved, live_reason = await cells_from_live_document(
        str(abs_path), set(specs), log
    )
    if resolved is None:
        try:
            resolved = await asyncio.to_thread(cells_from_file, abs_path, set(specs))
        except NotebookUnreadable as error:
            file_reason = UNREADABLE.format(error=error)
        else:
            file_reason = NO_IDS_ON_DISK
        if resolved is None:
            text = UNRESOLVED.format(
                count=count,
                name=name,
                why=f"{live_reason}, and {file_reason}",
                path=abs_path,
            )
            return TextContentBlock(type="text", text=text)

    lines = [HEADER.format(count=count, name=name, origin=resolved.origin), ""]
    found = sorted(
        ((index, cell_id, cell) for cell_id, (index, cell) in resolved.by_id.items()),
        key=lambda item: item[0],
    )
    used = 0
    over_budget: list[str] = []
    for index, cell_id, cell in found:
        if over_budget:
            over_budget.append(f"cell {index + 1} (id={cell_id})")
            continue
        rendered = _render_cell(
            cell_id, index + 1, resolved.total, cell, specs[cell_id], resolved.language
        )
        size = sum(len(line) + 1 for line in rendered)
        if used and used + size > MAX_TOTAL_CHARS:
            over_budget.append(f"cell {index + 1} (id={cell_id})")
            continue
        used += size
        lines.extend(rendered)
        lines.append("")
    if over_budget:
        cells = _count(len(over_budget), "attached cell")
        lines.append(OVER_BUDGET.format(count=cells, cells=", ".join(over_budget)))
        lines.append("")
    missing = [cell_id for cell_id in specs if cell_id not in resolved.by_id]
    if missing:
        lines.extend(NOT_FOUND.format(id=cell_id, origin=resolved.origin) for cell_id in missing)
        lines.append("")
    lines.append(FOOTER.format(path=abs_path))
    return TextContentBlock(type="text", text="\n".join(lines))
