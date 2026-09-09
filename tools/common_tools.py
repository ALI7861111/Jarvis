# Plain functions with type hints + docstrings. AutoGen auto-wraps these
# into tool schemas when passed to AssistantAgent(tools=[...]).
# Add your own here and import them into the agent that needs them.
from pathlib import Path
import subprocess

def read_file(path: str) -> str:
    """Read and return the contents of a text file."""
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"File not found: {path}")
    return file_path.read_text(encoding="utf-8")


def write_file(path: str, content: str) -> str:
    """Write content to a file, returning a status message."""
    try:
        file_path = Path(path)
        file_path.write_text(content, encoding="utf-8")

        return f"Successfully wrote file: {file_path}"

    except OSError as e:
        return f"Failed to write file: {e}"


def run_shell(command: str) -> str:
    """Run a shell command and return its output."""
    result = subprocess.run(
        command,
        shell=True,
        capture_output=True,
        text=True
    )
    if result.returncode != 0:
        return f"Command failed (exit code {result.returncode}):\n{result.stderr.strip()}"
    return result.stdout.strip()

