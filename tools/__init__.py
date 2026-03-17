"""
Tool-use subsystem for Living LLM.

Components:
  web_search   — DuckDuckGo / SearXNG search backends
  web_reader   — Clean text extraction from URLs (trafilatura)
  react_loop   — ReAct reasoning loop: intercepts tool calls, executes them,
                 feeds results back until the model produces a final answer
"""

from tools.web_search import search
from tools.web_reader import read_page
from tools.react_loop import ReactLoop

__all__ = ["search", "read_page", "ReactLoop"]
