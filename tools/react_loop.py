"""
ReAct Loop — intercepts tool calls in LLM responses and executes them.

The flow for each user message:
  1. Send messages to LLM
  2. If response contains <tool_call>...</tool_call>, parse and execute it
  3. Append model's response + tool result to message history
  4. Repeat until no tool calls remain OR max_iterations hit
  5. Return (final_response, search_log)

The search_log captures everything that was searched and read this turn.
Engine uses it to decide whether to extract and store web knowledge.

Rate limiting is session-scoped: ReactLoop is created once per engine
and reset at each new_session(). This tracks cumulative searches across
turns within a session.

Why class-based rather than a function?
Session state (search count, last search time) needs to persist across
multiple respond() calls within a single session.
"""

import json
import re
import time
from typing import Optional

import config

# Matches <tool_call>...</tool_call> with any whitespace/newlines inside
_TOOL_CALL_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)

# Matches URLs in text for citation verification
_URL_RE = re.compile(r"https?://[^\s\)\"'>]+")


class ReactLoop:
    """
    Runs the Reason-Act cycle for a single engine instance.

    Created once in ConversationEngine.__init__, reset at each new session.
    """

    def __init__(
        self,
        max_iterations: int = 3,
        max_calls_per_iteration: int = 2,
    ):
        self.max_iterations = max_iterations
        self.max_calls_per_iteration = max_calls_per_iteration

        # Session-scoped rate limiting
        self._session_search_count = 0
        self._last_search_time = 0.0

    def reset_session(self):
        """Call at the start of each new conversation session."""
        self._session_search_count = 0
        self._last_search_time = 0.0

    def run(self, backend, messages: list[dict]) -> tuple[str, list[dict]]:
        """
        Execute the ReAct loop.

        Args:
            backend: LLMBackend or MLXBackend — anything with a .chat() method
            messages: conversation history including system prompt

        Returns:
            (final_response_text, search_log)
            search_log is a list of dicts describing each tool call made.
        """
        search_log: list[dict] = []
        current_messages = list(messages)  # Don't mutate caller's list

        for iteration in range(self.max_iterations):
            response = backend.chat(current_messages)

            tool_call_jsons = _TOOL_CALL_RE.findall(response)

            if tool_call_jsons:
                print(f"  [ReAct] Iteration {iteration + 1}: {len(tool_call_jsons)} tool call(s) found")

            if not tool_call_jsons:
                # No tool calls — model has produced its final answer
                return _strip_tool_tags(response), search_log

            # Execute tool calls (up to per-iteration cap)
            result_parts: list[str] = []
            for call_json in tool_call_jsons[: self.max_calls_per_iteration]:
                result = self._execute(call_json.strip(), search_log)
                result_parts.append(result)

            # Append the model's tool-using turn + results to history
            current_messages.append({"role": "assistant", "content": response})
            current_messages.append({"role": "user", "content": "\n\n".join(result_parts)})

        # Max iterations reached — ask for a final synthesis without more tools
        current_messages.append({
            "role": "user",
            "content": "Please now provide your final answer based on the search results above.",
        })
        final = backend.chat(current_messages)
        return _strip_tool_tags(final), search_log

    # ── Execution ─────────────────────────────────────────────

    def _execute(self, call_json: str, search_log: list[dict]) -> str:
        """Parse a tool call JSON string and dispatch to the right tool."""
        try:
            call = json.loads(call_json)
        except json.JSONDecodeError as e:
            return _tool_result(f"Could not parse tool call JSON: {e}")

        tool = call.get("tool", "")

        if tool == "web_search":
            query = call.get("query", "").strip()
            if not query:
                return _tool_result("web_search requires a 'query' field")
            return self._do_search(query, search_log)

        elif tool == "read_page":
            url = call.get("url", "").strip()
            if not url:
                return _tool_result("read_page requires a 'url' field")
            return self._do_read(url, search_log)

        elif tool == "datetime":
            return self._do_datetime()

        elif tool == "python":
            code = call.get("code", "").strip()
            if not code:
                return _tool_result("python requires a 'code' field")
            return self._do_python(code)

        elif tool == "read_file":
            path = call.get("path", "").strip()
            if not path:
                return _tool_result("read_file requires a 'path' field")
            return self._do_read_file(path)

        elif tool == "write_file":
            path = call.get("path", "").strip()
            content = call.get("content", "")
            if not path:
                return _tool_result("write_file requires a 'path' field")
            return self._do_write_file(path, content)

        elif tool == "list_files":
            path = call.get("path", ".")
            return self._do_list_files(path)

        elif tool == "shell":
            command = call.get("command", "").strip()
            if not command:
                return _tool_result("shell requires a 'command' field")
            return self._do_shell(command)

        elif tool == "weather":
            location = call.get("location", "").strip()
            if not location:
                return _tool_result("weather requires a 'location' field")
            return self._do_weather(location)

        elif tool == "wikipedia":
            query = call.get("query", "").strip()
            if not query:
                return _tool_result("wikipedia requires a 'query' field")
            return self._do_wikipedia(query)

        elif tool == "notify":
            title = call.get("title", "Living LLM")
            message = call.get("message", "").strip()
            if not message:
                return _tool_result("notify requires a 'message' field")
            return self._do_notify(title, message)

        elif tool == "http_get":
            url = call.get("url", "").strip()
            if not url:
                return _tool_result("http_get requires a 'url' field")
            return self._do_http("GET", url, call.get("headers"))

        elif tool == "http_post":
            url = call.get("url", "").strip()
            if not url:
                return _tool_result("http_post requires a 'url' field")
            return self._do_http("POST", url, call.get("headers"), call.get("body"))

        else:
            available = (
                "web_search, read_page, datetime, python, read_file, write_file, "
                "list_files, shell, weather, wikipedia, notify, http_get, http_post"
            )
            return _tool_result(f"Unknown tool: {tool!r}. Available: {available}")

    def _do_search(self, query: str, search_log: list[dict]) -> str:
        # Session cap
        if self._session_search_count >= config.SEARCH_MAX_PER_SESSION:
            return _tool_result(
                f"Search limit reached ({config.SEARCH_MAX_PER_SESSION} searches this session)."
            )

        # Cooldown
        elapsed = time.time() - self._last_search_time
        if elapsed < config.SEARCH_COOLDOWN_SECS:
            time.sleep(config.SEARCH_COOLDOWN_SECS - elapsed)

        self._last_search_time = time.time()
        self._session_search_count += 1

        try:
            from tools.web_search import search, format_results_for_prompt
            results = search(query)
        except Exception as e:
            search_log.append({"type": "search", "query": query, "results": [], "error": str(e)})
            return _tool_result(f"Search failed: {e}. Try again or answer from existing knowledge.")

        search_log.append({"type": "search", "query": query, "results": results})

        if not results:
            return _tool_result(
                f"No results found for: {query}\n"
                "The query may be too specific, or the network may be unavailable."
            )

        return _tool_result(format_results_for_prompt(results, query))

    def _do_read(self, url: str, search_log: list[dict]) -> str:
        try:
            from tools.web_reader import read_page
            result = read_page(url)
        except Exception as e:
            search_log.append({"type": "read", "url": url, "success": False, "error": str(e)})
            return _tool_result(f"Failed to read page: {e}")

        search_log.append({
            "type": "read",
            "url": url,
            "success": result["error"] is None,
            "char_count": result["char_count"],
        })

        if result["error"]:
            return _tool_result(f"Could not read {url}: {result['error']}")

        title = f"\nTitle: {result['title']}\n" if result["title"] else ""
        return _tool_result(f"Content from {url}:{title}\n\n{result['content']}")

    # ── New tools ──────────────────────────────────────────────

    def _do_datetime(self) -> str:
        from tools.datetime_tool import now
        result = now()
        return _tool_result(
            f"Current time: {result['local_time']} ({result['timezone']})\n"
            f"Day: {result['day_of_week']}\n"
            f"UTC: {result['utc_time']}"
        )

    def _do_python(self, code: str) -> str:
        from tools.python_exec import run_code
        print(f"  [Tool] python: {code[:80]}...")
        result = run_code(code)
        if result["error"]:
            return _tool_result(f"Python error: {result['error']}")
        output = result["output"] or "(no output)"
        return _tool_result(f"Python output:\n{output}")

    def _do_read_file(self, path: str) -> str:
        from tools.file_tools import read_file
        result = read_file(path)
        if result["error"]:
            return _tool_result(f"File error: {result['error']}")
        return _tool_result(f"File: {result['path']}\n\n{result['content']}")

    def _do_write_file(self, path: str, content: str) -> str:
        from tools.file_tools import write_file
        result = write_file(path, content)
        if result["error"]:
            return _tool_result(f"Write error: {result['error']}")
        return _tool_result(f"Written {result['bytes_written']} bytes to {result['path']}")

    def _do_list_files(self, path: str) -> str:
        from tools.file_tools import list_files
        result = list_files(path)
        if result["error"]:
            return _tool_result(f"List error: {result['error']}")
        entries = "\n".join(result["files"]) if result["files"] else "(empty)"
        return _tool_result(f"Directory: {result['path']}\n{entries}")

    def _do_shell(self, command: str) -> str:
        from tools.shell_exec import run_shell
        print(f"  [Tool] shell: {command}")
        result = run_shell(command)
        if result["error"]:
            return _tool_result(f"Shell error: {result['error']}")
        output = result["output"] or "(no output)"
        return _tool_result(f"$ {command}\n{output}\n(exit code: {result['exit_code']})")

    def _do_weather(self, location: str) -> str:
        from tools.weather import get_weather
        result = get_weather(location)
        if result["error"]:
            return _tool_result(f"Weather error: {result['error']}")
        return _tool_result(
            f"Weather for {result['location']}:\n"
            f"  Temperature: {result['temperature_c']}°C / {result['temperature_f']}°F\n"
            f"  Conditions: {result['conditions']}\n"
            f"  Humidity: {result['humidity']}%\n"
            f"  Wind: {result['wind_speed_kmh']} km/h"
        )

    def _do_wikipedia(self, query: str) -> str:
        from tools.wikipedia import search_wikipedia
        result = search_wikipedia(query)
        if result["error"]:
            return _tool_result(f"Wikipedia error: {result['error']}")
        if not result["results"]:
            return _tool_result(f"No Wikipedia articles found for: {query}")
        parts = []
        for r in result["results"]:
            parts.append(f"**{r['title']}** ({r['url']})\n{r['summary']}")
        return _tool_result("\n\n".join(parts))

    def _do_notify(self, title: str, message: str) -> str:
        from tools.notify import send_notification
        result = send_notification(title, message)
        if result["error"]:
            return _tool_result(f"Notification error: {result['error']}")
        return _tool_result(f"Notification sent: {title} — {message}")

    def _do_http(self, method: str, url: str, headers: dict = None, body: dict = None) -> str:
        from tools.http_request import http_get, http_post
        if method == "POST":
            result = http_post(url, body=body, headers=headers)
        else:
            result = http_get(url, headers=headers)
        if result["error"]:
            return _tool_result(f"HTTP error: {result['error']}")
        return _tool_result(
            f"HTTP {method} {url} → {result['status']}\n{result['body']}"
        )

    # ── Citation verification ─────────────────────────────────

    @staticmethod
    def verify_citations(response: str, search_log: list[dict]) -> list[str]:
        """
        Return a list of URLs mentioned in the response that were NOT in
        any search result. These are potential hallucinated citations.
        """
        # Collect all URLs that actually came from searches
        known_urls: set[str] = set()
        for entry in search_log:
            if entry["type"] == "search":
                for r in entry.get("results", []):
                    known_urls.add(r.get("url", ""))
            elif entry["type"] == "read":
                known_urls.add(entry.get("url", ""))

        # Find URLs mentioned in the response
        mentioned = set(_URL_RE.findall(response))

        return [u for u in mentioned if u not in known_urls]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _tool_result(content: str) -> str:
    return f"<tool_result>\n{content}\n</tool_result>"


def _strip_tool_tags(text: str) -> str:
    """Remove any leftover <tool_call> or <tool_result> blocks from the final response."""
    text = re.sub(r"<tool_call>.*?</tool_call>", "", text, flags=re.DOTALL)
    text = re.sub(r"<tool_result>.*?</tool_result>", "", text, flags=re.DOTALL)
    return text.strip()
