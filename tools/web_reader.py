"""
Web Reader — fetches a URL and extracts clean readable text.

Uses trafilatura, which strips boilerplate, navigation, ads, and other
page furniture to return just the main content. This keeps extracted text
short enough to fit inside the model's limited context window.

Constraints:
  - 10-second timeout (configurable)
  - 4000-char content limit (truncated)
  - No JavaScript rendering — static HTML only
  - Domain blocklist for known spam/malicious sites
"""

import config

# Domains we refuse to fetch from. Extend as needed.
_BLOCKLIST = frozenset({
    "malware.com", "phishing.org",
    # Add spam/SEO-farm domains here as encountered
})


def read_page(url: str, max_chars: int = None) -> dict:
    """
    Fetch a URL and extract its main text content.

    Returns:
        {
            "url": str,
            "title": str | None,
            "content": str | None,
            "char_count": int,
            "error": str | None,
        }

    On any failure (timeout, blocked, no content), content is None and
    error describes what went wrong. Callers should check error != None.
    """
    max_chars = max_chars or config.WEB_READER_MAX_CHARS

    # Check blocklist
    try:
        from urllib.parse import urlparse
        domain = urlparse(url).netloc.lower().removeprefix("www.")
        if domain in _BLOCKLIST:
            return _err(url, f"Domain '{domain}' is blocked")
    except Exception:
        pass

    try:
        import trafilatura
    except ImportError:
        raise RuntimeError(
            "trafilatura not installed.\n"
            "Run: pip install trafilatura"
        )

    try:
        downloaded = trafilatura.fetch_url(
            url,
            config=trafilatura.settings.use_config(),
        )
    except Exception as e:
        return _err(url, f"Fetch failed: {e}")

    if not downloaded:
        return _err(url, "No content returned (page may require JavaScript or be blocked)")

    try:
        text = trafilatura.extract(
            downloaded,
            include_links=False,
            include_images=False,
            include_tables=True,
            output_format="txt",
        )
    except Exception as e:
        return _err(url, f"Extraction failed: {e}")

    if not text or not text.strip():
        return _err(url, "No extractable text content (dynamic page or paywall?)")

    # Extract title if available
    title = None
    try:
        meta = trafilatura.extract_metadata(downloaded)
        if meta:
            title = meta.title
    except Exception:
        pass

    text = text.strip()
    truncated = len(text) > max_chars
    if truncated:
        text = text[:max_chars] + "\n\n[Content truncated — fetch full URL to read more]"

    return {
        "url": url,
        "title": title,
        "content": text,
        "char_count": len(text),
        "truncated": truncated,
        "error": None,
    }


def _err(url: str, message: str) -> dict:
    return {"url": url, "title": None, "content": None, "char_count": 0,
            "truncated": False, "error": message}
