"""
HTTP request tool — generic REST API caller.

Supports GET and POST with JSON bodies. Useful for interacting with
local services, APIs, webhooks, etc.
"""

import json
import requests

_TIMEOUT = 10
_MAX_RESPONSE_CHARS = 4000


def http_get(url: str, headers: dict = None) -> dict:
    """
    Make an HTTP GET request.

    Returns:
        {"url": str, "status": int, "body": str, "truncated": bool, "error": str | None}
    """
    return _request("GET", url, headers=headers)


def http_post(url: str, body: dict = None, headers: dict = None) -> dict:
    """
    Make an HTTP POST request with a JSON body.

    Returns:
        {"url": str, "status": int, "body": str, "truncated": bool, "error": str | None}
    """
    return _request("POST", url, body=body, headers=headers)


def _request(method: str, url: str, body: dict = None, headers: dict = None) -> dict:
    try:
        resp = requests.request(
            method,
            url,
            json=body if method == "POST" and body else None,
            headers=headers or {},
            timeout=_TIMEOUT,
        )
    except Exception as e:
        return {"url": url, "status": 0, "body": "", "truncated": False,
                "error": f"Request failed: {e}"}

    text = resp.text
    truncated = len(text) > _MAX_RESPONSE_CHARS
    if truncated:
        text = text[:_MAX_RESPONSE_CHARS] + "\n[Response truncated]"

    return {
        "url": url,
        "status": resp.status_code,
        "body": text,
        "truncated": truncated,
        "error": None,
    }
