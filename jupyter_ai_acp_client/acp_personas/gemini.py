import os
import re
import shutil
import subprocess

from jupyter_ai_persona_manager import PersonaDefaults, PersonaRequirementsUnmet
from ..base_acp_persona import BaseAcpPersona

# Raise `PersonaRequirementsUnmet` if `gemini` not installed
if shutil.which("gemini") is None:
    raise PersonaRequirementsUnmet(
        "This persona requires `gemini` CLI to be installed."
        " See https://geminicli.com/ for installation instructions."
    )

# Raise `PersonaRequirementsUnmet` if `gemini` version check fails
try:
    result = subprocess.run(
        ["gemini", "--version"],
        capture_output=True,
        text=True,
        timeout=5
    )

    # Check for non-zero exit code
    if result.returncode != 0:
        stderr = result.stderr.strip()
        error_msg = (
            f"gemini --version returned non-zero exit code {result.returncode}."
            " Please ensure gemini CLI is properly installed."
        )
        if stderr:
            error_msg += f"\nStderr output: {stderr}"

        raise PersonaRequirementsUnmet(error_msg)

    # Extract semver from stdout using regex
    version_match = re.search(r'(\d+\.\d+\.\d+)', result.stdout)
    if not version_match:
        raise PersonaRequirementsUnmet(
            "Could not extract version number from gemini --version output."
            f" Got: {result.stdout.strip()}"
        )

    version_str = version_match.group(1)
    version_parts = [int(x) for x in version_str.split('.')]

    # Check if version >= 0.1.0
    required_version = (0, 1, 0)
    current_version = tuple(version_parts)

    if current_version < required_version:
        raise PersonaRequirementsUnmet(
            f"gemini CLI version {version_str} is installed, but version >=0.1.0 is required."
            " Please upgrade gemini CLI. See https://ai.google.dev for instructions."
        )

except subprocess.TimeoutExpired:
    raise PersonaRequirementsUnmet(
        "gemini --version command timed out."
        " Please ensure gemini CLI is properly installed."
    )
except FileNotFoundError:
    # This shouldn't happen since we checked with shutil.which, but handle it anyway
    raise PersonaRequirementsUnmet(
        "gemini command not found."
        " Please ensure gemini CLI is properly installed."
    )

class GeminiAcpPersona(BaseAcpPersona):
    def __init__(self, *args, **kwargs):
        executable = ["gemini", "--experimental-acp"] # For a specific model, use additional entries "-m", "<model_id>"
        super().__init__(*args, executable=executable, **kwargs)

    @property
    def defaults(self) -> PersonaDefaults:
        avatar_path = str(os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "static", "gemini.svg")
        ))

        return PersonaDefaults(
            name="Gemini",
            description="Gemini in Jupyter AI!",
            avatar_path=avatar_path,
            system_prompt="unused"
        )
