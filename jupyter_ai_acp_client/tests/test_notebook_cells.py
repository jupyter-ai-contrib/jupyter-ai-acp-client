"""Tests for rendering the cells named by a notebook attachment as text."""

import asyncio
import json
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from acp.schema import TextContentBlock
from jupyterlab_chat.models import (
    AttachmentSelection,
    NotebookAttachment,
    NotebookAttachmentCell,
)

from jupyter_ai_acp_client import notebook_cells
from jupyter_ai_acp_client.notebook_cells import render_cell_block


NOTEBOOK_MIME = "application/x-ipynb+json"
"""What the chat frontend sets on every notebook-cell attachment."""


def _notebook(cells: list[dict], minor: int = 5, language: str = "python") -> dict:
    """A minimal nbformat 4.x notebook dict. Minor 5 cells carry ids."""
    return {
        "nbformat": 4,
        "nbformat_minor": minor,
        "metadata": {"language_info": {"name": language}},
        "cells": cells,
    }


def _code(source, cell_id=None, outputs=None, execution_count=None) -> dict:
    cell = {
        "cell_type": "code",
        "source": source,
        "metadata": {},
        "outputs": outputs or [],
        "execution_count": execution_count,
    }
    if cell_id:
        cell["id"] = cell_id
    return cell


def _markdown(source, cell_id=None) -> dict:
    cell = {"cell_type": "markdown", "source": source, "metadata": {}}
    if cell_id:
        cell["id"] = cell_id
    return cell


def _write(tmp_path: Path, name: str, notebook: dict) -> Path:
    path = tmp_path / name
    path.write_text(json.dumps(notebook), encoding="utf-8")
    return path


def _cells(*ids: str, input_type: str = "code") -> list[NotebookAttachmentCell]:
    return [NotebookAttachmentCell(id=cell_id, input_type=input_type) for cell_id in ids]


def _dragged(value: str, *ids: str) -> NotebookAttachment:
    """A cell attachment as the chat frontend builds it: the notebook's
    path, the notebook media type and the dragged cells' ids."""
    return NotebookAttachment(value=value, mimetype=NOTEBOOK_MIME, cells=_cells(*ids))


async def _render(attachment, root_dir, log=None) -> str:
    """Render one attachment the way prompt_and_reply() calls the renderer,
    with the notebook's resolved absolute path, and return the block's text."""
    abs_path = (Path(root_dir) / attachment.value).resolve()
    block = await render_cell_block(attachment, abs_path, log or MagicMock())
    assert isinstance(block, TextContentBlock)
    return block.text


def _install_tools_stub(monkeypatch, *, file_id, ydoc, get_file_id=None, rtc=True) -> list:
    """Put a stand-in ``jupyter_ai_tools.utils`` in ``sys.modules`` and return
    the list that records every path ``get_file_id`` is asked about."""
    calls: list[str] = []

    async def _get_file_id(path):
        calls.append(path)
        return file_id

    async def _get_jupyter_ydoc(fid):
        assert fid == file_id
        return ydoc

    utils = types.ModuleType("jupyter_ai_tools.utils")
    utils.rtc_available = lambda: rtc
    utils.get_file_id = get_file_id or _get_file_id
    utils.get_jupyter_ydoc = _get_jupyter_ydoc
    package = types.ModuleType("jupyter_ai_tools")
    package.utils = utils
    monkeypatch.setitem(sys.modules, "jupyter_ai_tools", package)
    monkeypatch.setitem(sys.modules, "jupyter_ai_tools.utils", utils)
    return calls


@pytest.fixture(autouse=True)
def _without_jupyter_ai_tools(monkeypatch):
    """Make the live-document lookup unavailable unless a test installs a
    stand-in, so the file on disk is the source regardless of what the
    environment has installed."""
    monkeypatch.setitem(sys.modules, "jupyter_ai_tools", None)


class TestRenderFromFile:
    """Cells resolved from the notebook file on disk."""

    async def test_named_cell_is_rendered(self, tmp_path):
        """The named cell's source, outputs, execution count and position are
        rendered with the notebook's absolute path; the other cell is not."""
        stdout = {"output_type": "stream", "name": "stdout", "text": "1\n"}
        path = _write(
            tmp_path,
            "analysis.ipynb",
            _notebook(
                [
                    _code("secret = 1\n", "c1"),
                    _code("print(secret)\n", "c2", outputs=[stdout], execution_count=3),
                ]
            ),
        )

        text = await _render(_dragged("analysis.ipynb", "c2"), tmp_path)

        assert "print(secret)" in text
        assert "Output:" in text and "\n1\n" in text
        assert "Cell 2 of 2 (code, last run [3], id=c2):" in text
        assert (
            "The user attached 1 cell from analysis.ipynb, read from the file on disk:"
            in text
        )
        assert str(path.resolve()) in text
        assert "secret = 1" not in text

    async def test_dict_specs_from_chat_model(self, tmp_path):
        """The chat model's get_attachments() builds NotebookAttachment(**dict)
        without converting the nested cells, so at runtime they are dicts.
        The dict is the one the chat frontend sends."""
        _write(tmp_path, "nb.ipynb", _notebook([_code("x = 1\ny = 2\n", "c1")]))
        raw = {
            "value": "nb.ipynb",
            "type": "notebook",
            "mimetype": NOTEBOOK_MIME,
            "cells": [
                {
                    "id": "c1",
                    "input_type": "code",
                    "selection": {"start": [0, 0], "end": [0, 5], "content": "x = 1"},
                }
            ],
        }

        text = await _render(NotebookAttachment(**raw), tmp_path)

        assert "x = 1" in text
        assert "selection only" in text
        assert "y = 2" not in text

    async def test_selection_only(self, tmp_path):
        """A selection inside the cell replaces the source and is labelled."""
        _write(tmp_path, "nb.ipynb", _notebook([_code("x = 1\ny = 2\n", "c1")]))
        cell = NotebookAttachmentCell(
            id="c1",
            input_type="code",
            selection=AttachmentSelection(start=(1, 0), end=(1, 5), content="y = 2"),
        )
        attachment = NotebookAttachment(value="nb.ipynb", mimetype=NOTEBOOK_MIME, cells=[cell])

        text = await _render(attachment, tmp_path)

        assert "(code, selection only, never run, id=c1)" in text
        assert "y = 2" in text
        assert "x = 1" not in text

    async def test_multiple_in_document_order(self, tmp_path):
        """Several cells are rendered in notebook order, whatever the
        attachment's order, announced with their count."""
        cells = [
            _markdown("# First", "c1"),
            _code("middle = True\n", "c2"),
            _code("last = True\n", "c3"),
        ]
        _write(tmp_path, "nb.ipynb", _notebook(cells))

        text = await _render(_dragged("nb.ipynb", "c3", "c1"), tmp_path)

        assert "The user attached 2 cells from nb.ipynb" in text
        first = text.index("Cell 1 of 3 (markdown, id=c1):")
        assert first < text.index("Cell 3 of 3 (code, never run, id=c3):")
        assert "```markdown\n# First\n```" in text
        assert "middle = True" not in text

    async def test_cells_beyond_total_budget_are_listed(self, tmp_path):
        """A large multi-cell selection is rendered only up to the
        attachment's budget; the remaining cells are listed by position and
        id."""
        ids = [f"c{i}" for i in range(10)]
        cells = [_code(f"# {cell_id}\n" + "x" * 15_000 + "\n", cell_id) for cell_id in ids]
        _write(tmp_path, "big.ipynb", _notebook(cells))

        text = await _render(_dragged("big.ipynb", *ids), tmp_path)

        assert len(text) < 70_000
        assert "The user attached 10 cells from big.ipynb" in text
        assert "# c0\n" in text and "# c2\n" in text
        assert "# c3\n" not in text and "# c9\n" not in text
        assert "(7 attached cells not shown, to keep the prompt short: " in text
        assert "cell 4 (id=c3)" in text and "cell 10 (id=c9)" in text

    async def test_specs_without_id_and_duplicates_ignored(self, tmp_path):
        """A spec without an id is skipped and a repeated id is rendered once."""
        _write(tmp_path, "nb.ipynb", _notebook([_code("x = 1\n", "c1")]))
        raw = {
            "value": "nb.ipynb",
            "type": "notebook",
            "mimetype": NOTEBOOK_MIME,
            "cells": [
                {"id": "c1", "input_type": "code"},
                {"id": "c1", "input_type": "code"},
                {"input_type": "code"},
            ],
        }

        text = await _render(NotebookAttachment(**raw), tmp_path)

        assert text.count("Cell 1 of 1") == 1
        assert "The user attached 1 cell from" in text
        assert "None" not in text

    async def test_missing_id_reported(self, tmp_path):
        """An id absent from a notebook that does carry ids is reported by
        name, next to the cell that was found."""
        _write(tmp_path, "nb.ipynb", _notebook([_code("x = 1\n", "c1")]))

        text = await _render(_dragged("nb.ipynb", "c1", "gone"), tmp_path)

        assert "x = 1" in text
        assert "(cell gone was not found in the file on disk)" in text

    async def test_unreadable_file_reported(self, tmp_path):
        """A notebook that cannot be read is reported as such, not as one that
        stores no ids."""
        (tmp_path / "broken.ipynb").write_text("{not json", encoding="utf-8")

        text = await _render(_dragged("broken.ipynb", "c1"), tmp_path)

        assert "could not be resolved" in text
        assert "its file could not be read (" in text
        assert "stores no cell ids" not in text

    async def test_tolerates_malformed_cells(self, tmp_path):
        """Odd shapes in a notebook file (a None in a source list, string
        output data, a string traceback, a non-dict output) render what they
        can instead of failing."""
        outputs = [
            {"output_type": "execute_result", "data": "text/plain"},
            {"output_type": "error", "traceback": "Boom"},
            "junk",
        ]
        cell = _code(["x = 1\n", None], "c1", outputs=outputs)
        _write(tmp_path, "nb.ipynb", _notebook([cell]))

        text = await _render(_dragged("nb.ipynb", "c1"), tmp_path)

        assert "x = 1" in text
        assert "Boom" in text


class TestRenderFormatting:
    """Fences, caps and outputs."""

    async def test_fence_grows_past_backticks(self, tmp_path):
        """A markdown cell containing a fence is wrapped in a longer one."""
        source = "Run:\n\n```python\nprint(1)\n```\n"
        _write(tmp_path, "nb.ipynb", _notebook([_markdown(source, "c1")]))

        text = await _render(_dragged("nb.ipynb", "c1"), tmp_path)

        assert "````markdown\nRun:\n\n```python\nprint(1)\n```\n````" in text

    async def test_output_truncated(self, tmp_path):
        """A long output is cut at the cap and marked."""
        stdout = {"output_type": "stream", "name": "stdout", "text": "x" * 5000}
        cell = _code("print('x' * 5000)\n", "c1", outputs=[stdout])
        _write(tmp_path, "nb.ipynb", _notebook([cell]))

        text = await _render(_dragged("nb.ipynb", "c1"), tmp_path)

        assert "x" * 4000 in text
        assert "x" * 4001 not in text
        assert "[output truncated]" in text

    async def test_error_and_non_text_outputs(self, tmp_path):
        """Tracebacks (without ANSI colour codes) and text/plain results are
        inlined; every output with other representations, text/html-only
        ones included, is listed as omitted, once per output."""
        png = "iVBORw0KGgoAAAANSUhEUg=="
        outputs = [
            {
                "output_type": "error",
                "ename": "ValueError",
                "evalue": "bad",
                "traceback": ["\x1b[0;31mValueError\x1b[0m: bad"],
            },
            {
                "output_type": "execute_result",
                "execution_count": 2,
                "data": {"text/plain": "<Figure>", "image/png": png, "image/svg+xml": "<svg/>"},
            },
            {"output_type": "display_data", "data": {"text/html": "<b>hi</b>"}},
        ]
        _write(tmp_path, "nb.ipynb", _notebook([_code("plot()\n", "c1", outputs=outputs)]))

        text = await _render(_dragged("nb.ipynb", "c1"), tmp_path)

        assert "ValueError: bad" in text
        assert "\x1b" not in text
        assert "<Figure>" in text
        assert (
            "[2 outputs with non-text representations omitted: "
            "image/png, image/svg+xml, text/html]"
        ) in text
        assert png not in text
        assert "<b>hi</b>" not in text


class TestRenderUnresolved:
    """When neither source can resolve the ids, the block says why."""

    async def test_without_tools_says_so(self, tmp_path):
        """An nbformat 4.4 file stores no cell ids; with no live lookup possible
        the block explains both, and names the path, rather than reporting a
        missing cell."""
        path = _write(tmp_path, "old.ipynb", _notebook([_code("x = 1\n")], minor=4))

        text = await _render(_dragged("old.ipynb", "minted-id"), tmp_path)

        assert "could not be resolved" in text
        assert "jupyter_ai_tools is not available" in text
        assert "stores no cell ids (nbformat 4.4 or older)" in text
        assert str(path.resolve()) in text
        assert "was not found" not in text
        assert "x = 1" not in text

    async def test_when_closed_says_so(self, tmp_path, monkeypatch):
        """With the live lookup available but no open copy of the notebook,
        the explanation names that instead."""
        _write(tmp_path, "old.ipynb", _notebook([_code("x = 1\n")], minor=4))
        _install_tools_stub(monkeypatch, file_id=None, ydoc=None)

        text = await _render(_dragged("old.ipynb", "minted-id"), tmp_path)

        assert (
            "no open copy of the notebook was found on the server, and its file on "
            "disk stores no cell ids"
        ) in text
        assert "jupyter_ai_tools" not in text

    async def test_without_rtc_skips_live_lookup(self, tmp_path, monkeypatch):
        """Without a real-time collaboration provider there is no live document:
        the lookup is not attempted, nothing is logged as a failure, and the
        explanation says so."""
        _write(tmp_path, "old.ipynb", _notebook([_code("x = 1\n")], minor=4))
        calls = _install_tools_stub(monkeypatch, file_id="file-1", ydoc=None, rtc=False)
        log = MagicMock()

        text = await _render(_dragged("old.ipynb", "minted-id"), tmp_path, log)

        assert calls == []
        assert "real-time collaboration is not enabled, so there is no live document" in text
        assert "lookup failed" not in text
        assert not log.warning.called


class TestRenderFromLiveDocument:
    """Cells resolved from the notebook's live document."""

    async def test_live_document_preferred(self, tmp_path, monkeypatch):
        """The live document is the source when the notebook is open: its ids
        are the ones in the attachment (an nbformat 4.4 file has none), its
        content wins over a stale file, and pycrdt's float execution count
        renders as an integer."""
        YNotebook = pytest.importorskip("jupyter_ydoc").YNotebook
        path = _write(tmp_path, "live.ipynb", _notebook([_code("x = 1\n")], minor=4))
        ydoc = YNotebook()
        edited = _code("x = 2  # edited, unsaved\n", execution_count=7)
        ydoc.set(_notebook([edited], minor=4, language="R"))
        cell_id = ydoc.ycells[0]["id"]
        assert cell_id  # minted on load ...
        assert "id" not in ydoc.get()["cells"][0]  # ... and stripped again on the way out
        calls = _install_tools_stub(monkeypatch, file_id="file-1", ydoc=ydoc)

        text = await _render(_dragged("live.ipynb", cell_id), tmp_path)

        assert calls == [str(path.resolve())]
        assert "read from the live document:" in text
        assert f"Cell 1 of 1 (code, last run [7], id={cell_id}):" in text
        assert "```R\nx = 2  # edited, unsaved\n```" in text
        assert "x = 1" not in text

    async def test_outputs_service_placeholders_counted(self, tmp_path, monkeypatch):
        """With jupyter_server_documents, the live document holds URL
        placeholders in place of a code cell's outputs. They are counted, so
        a cell that did produce output does not look like one that printed
        nothing; their content is not fetched."""
        YNotebook = pytest.importorskip("jupyter_ydoc").YNotebook
        _write(tmp_path, "nb.ipynb", _notebook([_code("run()\n", "c1")]))
        url = {"url": "/api/outputs/f/c1/0.output"}
        placeholders = [  # the shapes of the outputs service's placeholders
            {"output_type": "stream", "text": "", "metadata": url},
            {"output_type": "display_data", "metadata": url},
            {"output_type": "execute_result", "metadata": url},
            {"output_type": "error", "metadata": url},
        ]
        stdout = {"output_type": "stream", "name": "stdout", "text": "kept\n"}
        ydoc = YNotebook()
        ydoc.set(_notebook([_code("run()\n", "c1", outputs=[*placeholders, stdout])]))
        _install_tools_stub(monkeypatch, file_id="file-1", ydoc=ydoc)

        text = await _render(_dragged("nb.ipynb", "c1"), tmp_path)

        assert "read from the live document:" in text
        assert "[4 outputs held by the server's outputs service, not shown]" in text
        assert "kept" in text

    async def test_live_lookup_failure_falls_back_to_file(self, tmp_path, monkeypatch):
        """A failing live lookup is logged and the file is used."""
        _write(tmp_path, "nb.ipynb", _notebook([_code("x = 1\n", "c1")]))

        async def broken(path):
            raise RuntimeError("no file id manager")

        _install_tools_stub(monkeypatch, file_id=None, ydoc=None, get_file_id=broken)
        log = MagicMock()

        text = await _render(_dragged("nb.ipynb", "c1"), tmp_path, log)

        assert "read from the file on disk:" in text
        assert "x = 1" in text
        assert "Live-document lookup failed" in log.warning.call_args.args[0]

    async def test_live_lookup_timeout_falls_back_to_file(self, tmp_path, monkeypatch):
        """A live lookup that hangs is abandoned after LIVE_LOOKUP_TIMEOUT,
        logged, and the file is used."""
        _write(tmp_path, "nb.ipynb", _notebook([_code("x = 1\n", "c1")]))

        async def hangs(path):
            await asyncio.sleep(60)

        _install_tools_stub(monkeypatch, file_id=None, ydoc=None, get_file_id=hangs)
        monkeypatch.setattr(notebook_cells, "LIVE_LOOKUP_TIMEOUT", 0.05)
        log = MagicMock()

        text = await asyncio.wait_for(_render(_dragged("nb.ipynb", "c1"), tmp_path, log), 5)

        assert "read from the file on disk:" in text
        assert "x = 1" in text
        assert "Live-document lookup failed" in log.warning.call_args.args[0]
