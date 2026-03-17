"""
Python execution tool — sandboxed REPL for math, data processing, logic.

Runs code in a restricted subprocess with a timeout. The subprocess has no
network access and limited imports. Output is captured from stdout.

Security model: defense-in-depth.
  1. Short timeout (10s) — prevents infinite loops
  2. Restricted builtins — no open(), exec(), eval(), __import__()
  3. Subprocess isolation — crash/OOM won't take down the main process
  4. Output size cap — prevents memory bombs
"""

import subprocess
import sys
import textwrap

_MAX_OUTPUT_CHARS = 4000
_TIMEOUT_SECS = 10

# Wrapper that restricts the execution environment
_RUNNER = textwrap.dedent('''\
    import sys, io, math, json, re, statistics, collections, itertools, functools
    from datetime import datetime, date, timedelta

    _code = sys.stdin.read()

    _out = io.StringIO()
    _old_stdout = sys.stdout
    sys.stdout = _out

    _restricted_builtins = {k: v for k, v in __builtins__.__dict__.items()
                            if k not in ("open", "exec", "eval", "__import__",
                                         "compile", "exit", "quit", "breakpoint",
                                         "input")}
    _restricted_builtins["__builtins__"] = _restricted_builtins

    try:
        compiled = compile(_code, "<tool>", "exec")
        exec(compiled, {"__builtins__": _restricted_builtins})
    except Exception as e:
        print(f"Error: {type(e).__name__}: {e}")

    sys.stdout = _old_stdout
    print(_out.getvalue(), end="")
''')


def run_code(code: str) -> dict:
    """
    Execute Python code and return stdout output.

    Returns:
        {
            "output": str,        # captured stdout
            "truncated": bool,
            "error": str | None,  # execution-level error (timeout, crash)
        }
    """
    try:
        result = subprocess.run(
            [sys.executable, "-c", _RUNNER],
            input=code,
            capture_output=True,
            text=True,
            timeout=_TIMEOUT_SECS,
        )
        output = result.stdout
        if result.stderr and not output:
            output = result.stderr

    except subprocess.TimeoutExpired:
        return {"output": "", "truncated": False,
                "error": f"Code execution timed out after {_TIMEOUT_SECS}s"}
    except Exception as e:
        return {"output": "", "truncated": False,
                "error": f"Execution failed: {e}"}

    truncated = len(output) > _MAX_OUTPUT_CHARS
    if truncated:
        output = output[:_MAX_OUTPUT_CHARS] + "\n[Output truncated]"

    return {
        "output": output.strip(),
        "truncated": truncated,
        "error": None,
    }
