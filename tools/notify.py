"""
Notification tool — macOS desktop notifications via osascript.
"""

import subprocess
import sys
import platform


def send_notification(title: str, message: str) -> dict:
    """
    Send a macOS desktop notification.

    Returns:
        {"title": str, "message": str, "sent": bool, "error": str | None}
    """
    if platform.system() != "Darwin":
        return {"title": title, "message": message, "sent": False,
                "error": "Notifications only supported on macOS"}

    script = (
        f'display notification "{_escape(message)}" '
        f'with title "{_escape(title)}"'
    )

    try:
        subprocess.run(
            ["osascript", "-e", script],
            capture_output=True,
            timeout=5,
        )
        return {"title": title, "message": message, "sent": True, "error": None}
    except Exception as e:
        return {"title": title, "message": message, "sent": False,
                "error": f"Failed: {e}"}


def _escape(text: str) -> str:
    """Escape special characters for AppleScript strings."""
    return text.replace("\\", "\\\\").replace('"', '\\"')
