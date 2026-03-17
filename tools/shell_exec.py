"""
Shell execution tool — run terminal commands with an allowlist.

Only commands on the allowlist can be executed. This prevents the model
from running destructive operations while still being useful for system queries.
"""

import subprocess
import shlex

_TIMEOUT_SECS = 15
_MAX_OUTPUT_CHARS = 4000

# Commands the model is allowed to run. Extend as needed.
ALLOWED_COMMANDS = {
    # System info
    "date", "uptime", "whoami", "hostname", "uname",
    "df", "du", "free", "top",
    # File inspection (read-only)
    "ls", "cat", "head", "tail", "wc", "file", "find", "which",
    # Network (read-only)
    "ping", "curl", "dig", "nslookup", "ifconfig",
    # Dev tools (read-only)
    "git", "python3", "pip", "node", "npm",
    # Process info
    "ps", "lsof",
    # Text processing
    "grep", "sort", "uniq", "cut", "tr", "sed", "awk", "jq",
    # macOS specific
    "sw_vers", "system_profiler", "diskutil", "pmset",
    "pbcopy", "pbpaste", "open", "say",
}


def run_shell(command: str) -> dict:
    """
    Execute a shell command if it's on the allowlist.

    Returns:
        {"command": str, "output": str, "exit_code": int, "error": str | None}
    """
    try:
        parts = shlex.split(command)
    except ValueError as e:
        return {"command": command, "output": "", "exit_code": -1,
                "error": f"Invalid command: {e}"}

    if not parts:
        return {"command": command, "output": "", "exit_code": -1,
                "error": "Empty command"}

    base_cmd = parts[0].split("/")[-1]  # Handle full paths like /usr/bin/git
    if base_cmd not in ALLOWED_COMMANDS:
        return {"command": command, "output": "", "exit_code": -1,
                "error": f"Command '{base_cmd}' is not on the allowlist. "
                         f"Allowed: {', '.join(sorted(ALLOWED_COMMANDS))}"}

    try:
        result = subprocess.run(
            parts,
            capture_output=True,
            text=True,
            timeout=_TIMEOUT_SECS,
            cwd=None,
        )
        output = result.stdout
        if result.stderr:
            output += ("\n--- stderr ---\n" + result.stderr) if output else result.stderr

    except subprocess.TimeoutExpired:
        return {"command": command, "output": "", "exit_code": -1,
                "error": f"Command timed out after {_TIMEOUT_SECS}s"}
    except FileNotFoundError:
        return {"command": command, "output": "", "exit_code": -1,
                "error": f"Command '{base_cmd}' not found on this system"}
    except Exception as e:
        return {"command": command, "output": "", "exit_code": -1,
                "error": f"Execution failed: {e}"}

    truncated = len(output) > _MAX_OUTPUT_CHARS
    if truncated:
        output = output[:_MAX_OUTPUT_CHARS] + "\n[Output truncated]"

    return {
        "command": command,
        "output": output.strip(),
        "exit_code": result.returncode,
        "error": None,
    }
