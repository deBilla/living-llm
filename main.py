"""
Living LLM — Main entry point.

Terminal chat interface with optional Gradio web UI.
Supports special commands for memory inspection.

Usage:
    python main.py           # Terminal mode
    python main.py --ui      # Gradio web UI
"""

import sys
import argparse

from engine import ConversationEngine


# ── Display helpers ───────────────────────────────────────────

def _print_adapter_status(status: dict, console, use_rich: bool):
    if use_rich:
        from rich.panel import Panel
        lines = []
        lines.append(f"[bold]LoRA enabled:[/bold] {status.get('lora_enabled', False)}")
        lines.append(f"[bold]MLX available:[/bold] {status.get('mlx_available', False)}")
        lines.append(f"[bold]Adapter active:[/bold] {status.get('adapter_active', False)}")
        lines.append(f"[bold]Training running:[/bold] {status.get('training_running', False)}")
        lines.append(f"[bold]Adapters stored:[/bold] {status.get('num_adapters', 0)}")
        if status.get("adapter_name"):
            lines.append(f"[bold]Current adapter:[/bold] {status['adapter_name']}")
            lines.append(f"[bold]Trained at:[/bold] {status.get('trained_at', '?')}")
            lines.append(f"[bold]Trained on:[/bold] {status.get('trained_on_conversations', '?')} conversation(s)")
        if status.get("total_training_runs"):
            lines.append(f"[bold]Training runs:[/bold] {status['total_training_runs']}")
        if status.get("avg_response_similarity") is not None:
            lines.append(f"[bold]Avg response similarity:[/bold] {status['avg_response_similarity']}")
        console.print(Panel("\n".join(lines), title="adapter status", border_style="cyan"))
    else:
        print("\n--- Adapter Status ---")
        for k, v in status.items():
            print(f"  {k}: {v}")
        print()


def _print_web_knowledge(memories: list, console, use_rich: bool):
    import time
    if not memories:
        msg = "No web knowledge stored. Ask something that requires a web search."
        if use_rich:
            from rich.panel import Panel
            console.print(Panel(msg, title="web knowledge", border_style="cyan"))
        else:
            print(f"\n{msg}\n")
        return

    if use_rich:
        from rich.panel import Panel
        lines = [f"[bold]{len(memories)} entries stored[/bold]\n"]
        for m in memories:
            conf = m.metadata.get("confidence", 0.7)
            urls = m.metadata.get("source_urls", [])
            src = urls[0] if urls else "web"
            age_days = (time.time() - m.metadata.get("retrieved_at", m.created_at)) / 86400
            is_news = m.metadata.get("is_news", False)
            lines.append(
                f"  [{conf:.0%}] {m.content}\n"
                f"        Source: {src}  |  Age: {age_days:.1f}d  |  {'news' if is_news else 'general'}"
            )
        console.print(Panel("\n".join(lines), title="web knowledge", border_style="cyan"))
    else:
        print(f"\n--- Web Knowledge ({len(memories)} entries) ---")
        for m in memories:
            conf = m.metadata.get("confidence", 0.7)
            src = (m.metadata.get("source_urls") or ["web"])[0]
            print(f"  [{conf:.0%}] {m.content}")
            print(f"        {src}")
        print()


def _print_comparison(result: dict, console, use_rich: bool):
    if use_rich:
        from rich.panel import Panel
        from rich.columns import Columns
        console.print(Panel(result["base"], title="base model", border_style="yellow"))
        console.print(Panel(result["adapted"], title="adapted model", border_style="green"))
    else:
        print(f"\n[BASE MODEL]\n{result['base']}\n")
        print(f"[ADAPTED MODEL]\n{result['adapted']}\n")


# ── Terminal Chat ─────────────────────────────────────────────

def run_terminal():
    """Rich terminal chat interface."""
    try:
        from rich.console import Console
        from rich.panel import Panel
        from rich.markdown import Markdown
        console = Console()
        use_rich = True
    except ImportError:
        console = None
        use_rich = False

    def print_header():
        if use_rich:
            console.print(Panel(
                "[bold]Living LLM[/bold] — A continuously learning language model\n"
                "  [bold]/memory[/bold]           inspect memory state\n"
                "  [bold]/search <query>[/bold]   force a web search\n"
                "  [bold]/knowledge[/bold]        show stored web knowledge\n"
                "  [bold]/knowledge clear[/bold]  clear all web knowledge\n"
                "  [bold]/knowledge decay[/bold]  run confidence decay\n"
                "  [bold]/train[/bold]            train LoRA adapter\n"
                "  [bold]/adapter[/bold]          show adapter status\n"
                "  [bold]/adapter compare[/bold]  compare base vs adapted\n"
                "  [bold]/adapter off|on[/bold]   toggle adapter\n"
                "  [bold]/new[/bold]              start a new session\n"
                "  [bold]/quit[/bold]             end session",
                title="living-llm",
                border_style="blue",
            ))
        else:
            print("\n" + "=" * 50)
            print("  Living LLM — A continuously learning language model")
            print("  /memory              — inspect memory state")
            print("  /search <query>      — force web search")
            print("  /knowledge           — show web knowledge")
            print("  /knowledge clear     — clear web knowledge")
            print("  /knowledge decay     — run confidence decay")
            print("  /train               — train LoRA adapter")
            print("  /adapter             — adapter status")
            print("  /adapter compare     — compare base vs adapted")
            print("  /adapter off|on      — toggle adapter")
            print("  /quit                — end session")
            print("=" * 50)

    def print_response(text):
        if use_rich:
            console.print(Panel(Markdown(text), border_style="green", title="assistant"))
        else:
            print(f"\nAssistant: {text}\n")

    def print_memory(debug_info):
        if use_rich:
            stats = debug_info["stats"]
            console.print(Panel(
                f"[bold]Session:[/bold] {debug_info['session_id']}  |  "
                f"[bold]Turns:[/bold] {debug_info['turn_count']}\n\n"
                f"[bold blue]Short-term:[/bold blue] {stats['short']} memories\n"
                f"[bold yellow]Mid-term:[/bold yellow] {stats['mid']} gists\n"
                f"[bold green]Long-term:[/bold green] {stats['long']} facts\n"
                f"[bold cyan]Web knowledge:[/bold cyan] {stats.get('web', 0)} entries\n"
                f"[bold]Conversations stored:[/bold] {stats['conversations']}",
                title="memory state",
                border_style="yellow",
            ))
            if debug_info["long_term"]:
                console.print("\n[bold green]Long-term knowledge:[/bold green]")
                for fact in debug_info["long_term"]:
                    console.print(f"  {fact}")
            if debug_info["mid_term"]:
                console.print("\n[bold yellow]Mid-term gists:[/bold yellow]")
                for gist in debug_info["mid_term"]:
                    console.print(f"  {gist}")
            console.print()
        else:
            stats = debug_info["stats"]
            print(f"\n--- Memory State ---")
            print(f"Session: {debug_info['session_id']}  |  Turns: {debug_info['turn_count']}")
            print(f"Short-term: {stats['short']} | Mid-term: {stats['mid']} | Long-term: {stats['long']}")
            if debug_info["long_term"]:
                print("\nLong-term knowledge:")
                for fact in debug_info["long_term"]:
                    print(f"  {fact}")
            if debug_info["mid_term"]:
                print("\nMid-term gists:")
                for gist in debug_info["mid_term"]:
                    print(f"  {gist}")
            print()

    # Initialize
    print("\nInitializing...")
    engine = ConversationEngine()
    engine.start_session()
    print_header()

    # Chat loop
    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (KeyboardInterrupt, EOFError):
            user_input = "/quit"

        if not user_input:
            continue

        # Handle commands
        if user_input.lower() in ("/quit", "/exit", "/q"):
            print("\nEnding session...")
            results = engine.end_session()
            print(f"\nSession complete. {results.get('turns', 0)} turns.")
            break

        if user_input.lower() in ("/memory", "/mem", "/m"):
            print_memory(engine.get_memory_debug())
            continue

        if user_input.lower() == "/new":
            print("\nEnding current session and starting new one...")
            engine.end_session()
            engine.start_session()
            print_header()
            continue

        if user_input.lower().startswith("/search "):
            query = user_input[8:].strip()
            if not query:
                print("  Usage: /search <query>")
            else:
                print(f"\n  Searching: {query!r}...")
                if use_rich:
                    with console.status("[bold cyan]Searching...", spinner="dots"):
                        response = engine.forced_search(query)
                else:
                    response = engine.forced_search(query)
                print_response(response)
            continue

        if user_input.lower() == "/knowledge":
            _print_web_knowledge(engine.get_web_knowledge(), console, use_rich)
            continue

        if user_input.lower() == "/knowledge clear":
            engine.clear_web_knowledge()
            continue

        if user_input.lower() == "/knowledge decay":
            engine.decay_web_knowledge()
            continue

        if user_input.lower() == "/train":
            print("\nTriggering LoRA training...")
            engine.train_now()
            continue

        if user_input.lower() == "/adapter":
            _print_adapter_status(engine.get_adapter_status(), console, use_rich)
            continue

        if user_input.lower() == "/adapter off":
            engine.adapter_off()
            continue

        if user_input.lower() == "/adapter on":
            engine.adapter_on()
            continue

        if user_input.lower().startswith("/adapter compare"):
            # Use the last user message as the comparison prompt, or ask for one
            parts = user_input.split(maxsplit=2)
            prompt = parts[2] if len(parts) > 2 else None
            if not prompt:
                # Use the last user turn from this session if available
                user_turns = [m["content"] for m in engine.messages if m["role"] == "user"]
                prompt = user_turns[-1] if user_turns else "Tell me about yourself."
            print(f"\nComparing responses for: {prompt!r}\n")
            result = engine.compare_responses(prompt)
            if result:
                _print_comparison(result, console, use_rich)
            continue

        # Get response
        if use_rich:
            with console.status("[bold blue]Thinking...", spinner="dots"):
                response = engine.respond(user_input)
        else:
            print("  Thinking...")
            response = engine.respond(user_input)

        print_response(response)


# ── Gradio Web UI ─────────────────────────────────────────────

def run_gradio():
    """Launch Gradio web interface."""
    import gradio as gr
    import time

    engine = ConversationEngine()
    engine.start_session()

    CSS = """
    .chat-col { display: flex; flex-direction: column; height: calc(100vh - 120px); }
    .memory-col { display: flex; flex-direction: column; height: calc(100vh - 120px); }
    .mem-tab-content { overflow-y: auto; max-height: 340px; padding: 4px 0; }
    .input-row { flex-shrink: 0; margin-top: 8px; }
    #chatbot { flex: 1; min-height: 0; }
    #send-btn { min-width: 80px; }
    """

    # ── Data helpers ──────────────────────────────────────────

    def _overview_md():
        debug = engine.get_memory_debug()
        stats = debug["stats"]
        status = engine.get_adapter_status()
        adapter_line = (
            f"✅ **Adapter active** (`{status.get('adapter_name', '?')}`)"
            if status.get("adapter_active")
            else ("⏳ **Training running**" if status.get("training_running") else "⚪ No adapter")
        )
        lines = [
            f"**Session:** `{debug['session_id']}`  |  **Turns:** {debug['turn_count']}",
            "",
            f"| Tier | Count |",
            f"|------|-------|",
            f"| Short-term | {stats['short']} |",
            f"| Mid-term | {stats['mid']} |",
            f"| Long-term | {stats['long']} |",
            f"| Web knowledge | {stats.get('web', 0)} |",
            f"| Conversations | {stats['conversations']} |",
            "",
            f"**LoRA:** {adapter_line}",
        ]
        return "\n".join(lines)

    def _longterm_md():
        debug = engine.get_memory_debug()
        facts = debug["long_term"]
        if not facts:
            return "_No long-term facts yet. Have a few conversations and end them with /quit._"
        return "\n".join(f"- {f}" for f in facts)

    def _midterm_md():
        debug = engine.get_memory_debug()
        gists = debug["mid_term"]
        if not gists:
            return "_No mid-term gists yet._"
        return "\n".join(f"- {g}" for g in gists)

    def _web_md():
        import time as _time
        memories = engine.get_web_knowledge()
        if not memories:
            return "_No web knowledge stored. Use the search bar or ask questions that need current info._"
        lines = [f"**{len(memories)} entries**\n"]
        for m in memories:
            conf = m.metadata.get("confidence", 0.7)
            urls = m.metadata.get("source_urls", [])
            src = urls[0] if urls else "web"
            age_days = (_time.time() - m.metadata.get("retrieved_at", m.created_at)) / 86400
            is_news = m.metadata.get("is_news", False)
            tag = "📰" if is_news else "🌐"
            lines.append(
                f"{tag} **[{conf:.0%}]** {m.content}  \n"
                f"  _{src}_  ·  {age_days:.1f}d ago"
            )
        return "\n\n".join(lines)

    def _all_memory():
        return _overview_md(), _longterm_md(), _midterm_md(), _web_md()

    # ── Event handlers ────────────────────────────────────────

    def chat(message, history):
        if not message.strip():
            return history, "", *_all_memory()
        response = engine.respond(message)
        history = (history or []) + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": response},
        ]
        return history, "", *_all_memory()

    def do_search(query):
        if not query.strip():
            return gr.update(), "", *_all_memory()
        response = engine.forced_search(query)
        return response, "", *_all_memory()

    def new_session():
        engine.end_session()
        engine.start_session()
        return [], *_all_memory()

    def clear_web():
        engine.clear_web_knowledge()
        return _web_md()

    def decay_web():
        engine.decay_web_knowledge()
        return _web_md()

    # ── Layout ────────────────────────────────────────────────

    with gr.Blocks(title="Living LLM") as demo:
        gr.Markdown("## 🧠 Living LLM — continuously learning language model")

        with gr.Row(equal_height=True):
            # ── Left: chat panel ──────────────────────────────
            with gr.Column(scale=3, elem_classes="chat-col"):
                chatbot = gr.Chatbot(
                    height=480,
                    elem_id="chatbot",
                    label="Conversation",
                )
                with gr.Row(elem_classes="input-row"):
                    msg_box = gr.Textbox(
                        placeholder="Type a message… (Enter to send)",
                        show_label=False,
                        lines=1,
                        scale=8,
                        container=False,
                    )
                    send_btn = gr.Button("Send", variant="primary", scale=1, elem_id="send-btn")

                with gr.Accordion("🔍 Web search", open=False):
                    with gr.Row():
                        search_box = gr.Textbox(
                            placeholder="Search query…",
                            show_label=False,
                            scale=7,
                            container=False,
                        )
                        search_btn = gr.Button("Search", scale=1)
                    search_result = gr.Markdown(label="Search result")

            # ── Right: memory panel ───────────────────────────
            with gr.Column(scale=2, elem_classes="memory-col"):
                with gr.Tabs():
                    with gr.Tab("Overview"):
                        overview_md = gr.Markdown(
                            value=_overview_md,
                            elem_classes="mem-tab-content",
                        )
                    with gr.Tab("Long-term"):
                        longterm_md = gr.Markdown(
                            value=_longterm_md,
                            elem_classes="mem-tab-content",
                        )
                    with gr.Tab("Mid-term"):
                        midterm_md = gr.Markdown(
                            value=_midterm_md,
                            elem_classes="mem-tab-content",
                        )
                    with gr.Tab("Web knowledge"):
                        web_md = gr.Markdown(
                            value=_web_md,
                            elem_classes="mem-tab-content",
                        )
                        with gr.Row():
                            decay_btn = gr.Button("Decay", size="sm")
                            clear_web_btn = gr.Button("Clear all", size="sm", variant="stop")

                with gr.Row():
                    refresh_btn = gr.Button("↺ Refresh memory", size="sm")
                    new_session_btn = gr.Button("New session", size="sm", variant="secondary")

        # ── Wire events ───────────────────────────────────────

        all_mem_outputs = [overview_md, longterm_md, midterm_md, web_md]

        # Send message (button or Enter)
        send_btn.click(
            fn=chat,
            inputs=[msg_box, chatbot],
            outputs=[chatbot, msg_box, *all_mem_outputs],
        )
        msg_box.submit(
            fn=chat,
            inputs=[msg_box, chatbot],
            outputs=[chatbot, msg_box, *all_mem_outputs],
        )

        # Web search
        search_btn.click(
            fn=do_search,
            inputs=[search_box],
            outputs=[search_result, search_box, *all_mem_outputs],
        )
        search_box.submit(
            fn=do_search,
            inputs=[search_box],
            outputs=[search_result, search_box, *all_mem_outputs],
        )

        # Memory controls
        refresh_btn.click(fn=_all_memory, outputs=all_mem_outputs)
        new_session_btn.click(fn=new_session, outputs=[chatbot, *all_mem_outputs])
        decay_btn.click(fn=decay_web, outputs=[web_md])
        clear_web_btn.click(fn=clear_web, outputs=[web_md])

    demo.launch(theme=gr.themes.Soft(), css=CSS)


# ── Entry Point ───────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Living LLM")
    parser.add_argument("--ui", action="store_true", help="Launch Gradio web UI")
    args = parser.parse_args()

    if args.ui:
        run_gradio()
    else:
        run_terminal()
