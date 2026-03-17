"""
Living LLM — Main entry point.

Terminal chat interface with optional Gradio web UI.
Powered by limbiq for neurotransmitter-inspired adaptive learning.

Usage:
    python main.py           # Terminal mode
    python main.py --ui      # Gradio web UI
"""

import sys
import json
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


def _print_comparison(result: dict, console, use_rich: bool):
    if use_rich:
        from rich.panel import Panel
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
                "[bold]Living LLM[/bold] — powered by [cyan]limbiq[/cyan]\n"
                "  [bold]/memory[/bold]           limbiq memory state\n"
                "  [bold]/signals[/bold]          recent signal history\n"
                "  [bold]/priority[/bold]         dopamine-tagged memories\n"
                "  [bold]/suppress[/bold]         GABA-suppressed memories\n"
                "  [bold]/dopamine <fact>[/bold]  tag a fact as high-priority\n"
                "  [bold]/correct <info>[/bold]   correct a wrong memory\n"
                "  [bold]/good[/bold]             mark last response as positive\n"
                "  [bold]/bad[/bold]              mark last response as negative\n"
                "  [bold]/restore <id>[/bold]     restore a suppressed memory\n"
                "  [bold]/search <query>[/bold]   force a web search\n"
                "  [bold]/export[/bold]           export limbiq state as JSON\n"
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
            print("\n" + "=" * 55)
            print("  Living LLM — powered by limbiq")
            print("  /memory              — limbiq memory state")
            print("  /signals             — recent signal history")
            print("  /priority            — dopamine-tagged memories")
            print("  /suppress            — GABA-suppressed memories")
            print("  /dopamine <fact>     — tag fact as priority")
            print("  /correct <info>      — correct a wrong memory")
            print("  /good                — positive feedback")
            print("  /bad                 — negative feedback")
            print("  /restore <id>        — restore suppressed memory")
            print("  /search <query>      — force web search")
            print("  /export              — export limbiq state")
            print("  /train               — train LoRA adapter")
            print("  /adapter             — adapter status")
            print("  /adapter compare     — compare base vs adapted")
            print("  /adapter off|on      — toggle adapter")
            print("  /new                 — new session")
            print("  /quit                — end session")
            print("=" * 55)

    def print_response(text):
        if use_rich:
            console.print(Panel(Markdown(text), border_style="green", title="assistant"))
        else:
            print(f"\nAssistant: {text}\n")

    def print_memory(debug_info):
        stats = debug_info["stats"]
        if use_rich:
            lines = [
                f"[bold]Turns:[/bold] {debug_info['turn_count']}\n",
            ]
            for key, val in stats.items():
                lines.append(f"[bold]{key}:[/bold] {val}")

            if debug_info["priority"]:
                lines.append("\n[bold green]Priority memories (Dopamine-tagged):[/bold green]")
                for fact in debug_info["priority"]:
                    lines.append(f"  {fact}")

            lines.append(f"\n[bold red]Suppressed:[/bold red] {debug_info['suppressed_count']} memories")

            if debug_info["recent_signals"]:
                lines.append("\n[bold cyan]Recent signals:[/bold cyan]")
                for sig in debug_info["recent_signals"]:
                    lines.append(f"  [{sig['type']}] {sig['trigger']}")

            console.print(Panel("\n".join(lines), title="limbiq memory", border_style="yellow"))
        else:
            print(f"\n--- Limbiq Memory ---")
            print(f"Turns: {debug_info['turn_count']}")
            for key, val in stats.items():
                print(f"  {key}: {val}")
            if debug_info["priority"]:
                print("\nPriority memories:")
                for fact in debug_info["priority"]:
                    print(f"  {fact}")
            print(f"\nSuppressed: {debug_info['suppressed_count']} memories")
            if debug_info["recent_signals"]:
                print("\nRecent signals:")
                for sig in debug_info["recent_signals"]:
                    print(f"  [{sig['type']}] {sig['trigger']}")
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
            print(f"\nSession complete.")
            break

        if user_input.lower() in ("/memory", "/mem", "/m"):
            print_memory(engine.get_memory_debug())
            continue

        if user_input.lower() in ("/signals", "/sig"):
            signals = engine.lq.get_signal_log(limit=20)
            if not signals:
                print("  No signals fired yet.")
            else:
                for s in signals:
                    stype = s.signal_type if isinstance(s.signal_type, str) else s.signal_type.value
                    print(f"  [{stype}] {s.trigger} — {s.details}")
            continue

        if user_input.lower() in ("/priority", "/pri"):
            memories = engine.lq.get_priority_memories()
            if memories:
                for m in memories:
                    print(f"  [{m.id[:8]}] {m.content}")
            else:
                print("  No priority memories yet.")
            continue

        if user_input.lower() == "/suppress":
            memories = engine.lq.get_suppressed()
            if memories:
                for m in memories:
                    print(f"  [{m.id[:8]}] {m.content} (reason: {m.suppression_reason})")
            else:
                print("  No suppressed memories.")
            continue

        if user_input.lower().startswith("/dopamine "):
            fact = user_input[len("/dopamine "):].strip()
            if fact:
                engine.lq.dopamine(fact)
                print(f"  Dopamine tagged: {fact}")
            continue

        if user_input.lower().startswith("/gaba "):
            memory_id = user_input[len("/gaba "):].strip()
            if memory_id:
                engine.lq.gaba(memory_id)
                print(f"  Memory {memory_id} suppressed.")
            continue

        if user_input.lower().startswith("/correct "):
            correction = user_input[len("/correct "):].strip()
            if correction:
                engine.lq.correct(correction)
                print(f"  Correction applied: {correction}")
            continue

        if user_input.lower() == "/good":
            engine.handle_feedback("positive")
            print("  Positive feedback recorded.")
            continue

        if user_input.lower() == "/bad":
            engine.handle_feedback("negative")
            print("  Negative feedback recorded.")
            continue

        if user_input.lower().startswith("/restore "):
            memory_id = user_input[len("/restore "):].strip()
            if memory_id:
                engine.lq.restore_memory(memory_id)
                print(f"  Memory {memory_id} restored.")
            continue

        if user_input.lower() == "/export":
            state = engine.lq.export_state()
            export_path = "data/limbiq_export.json"
            with open(export_path, "w") as f:
                json.dump(state, f, indent=2, default=str)
            print(f"  Exported to {export_path}")
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
            parts = user_input.split(maxsplit=2)
            prompt = parts[2] if len(parts) > 2 else None
            if not prompt:
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

    engine = ConversationEngine()
    engine.start_session()

    CSS = """
    body, .gradio-container { height: 100vh; overflow: hidden; }
    .main-row { height: calc(100vh - 80px) !important; align-items: stretch; }
    .chat-col { display: flex; flex-direction: column; height: 100%; min-height: 0; }
    .memory-col { display: flex; flex-direction: column; height: 100%; min-height: 0; }
    #chatbot { flex: 1 1 auto; min-height: 0; }
    #chatbot > div { height: 100% !important; }
    .mem-tab-content { overflow-y: auto; flex: 1 1 auto; min-height: 0; max-height: calc(100vh - 260px); padding: 4px 0; }
    .input-row { flex-shrink: 0; margin-top: 6px; }
    #send-btn { min-width: 80px; }
    """

    # ── Data helpers ──────────────────────────────────────────

    def _memory_md():
        debug = engine.get_memory_debug()
        stats = debug["stats"]
        status = engine.get_adapter_status()
        adapter_line = (
            f"**Adapter active** (`{status.get('adapter_name', '?')}`)"
            if status.get("adapter_active")
            else ("**Training running**" if status.get("training_running") else "No adapter")
        )
        lines = [
            f"**Turns:** {debug['turn_count']}",
            "",
            "| Metric | Count |",
            "|--------|-------|",
        ]
        for key, val in stats.items():
            lines.append(f"| {key} | {val} |")
        lines.append("")
        lines.append(f"**LoRA:** {adapter_line}")
        return "\n".join(lines)

    def _priority_md():
        debug = engine.get_memory_debug()
        facts = debug["priority"]
        if not facts:
            return "_No priority memories yet. Share some personal info or use the Dopamine action._"
        return "\n".join(f"- {f}" for f in facts)

    def _signals_md():
        signals = engine.lq.get_signal_log(limit=30)
        if not signals:
            return "_No signals fired yet._"
        lines = []
        for s in signals:
            stype = s.signal_type if isinstance(s.signal_type, str) else s.signal_type.value
            lines.append(f"**[{stype}]** {s.trigger}")
        return "\n".join(lines)

    def _suppressed_md():
        memories = engine.lq.get_suppressed()
        if not memories:
            return "_No suppressed memories._"
        lines = []
        for m in memories:
            lines.append(f"- `{m.id[:8]}` {m.content} _(reason: {m.suppression_reason})_")
        return "\n".join(lines)

    def _all_panels():
        return _memory_md(), _priority_md(), _signals_md(), _suppressed_md()

    # ── Event handlers ────────────────────────────────────────

    def chat(message, history):
        if not message.strip():
            return history, "", *_all_panels()
        response = engine.respond(message)
        history = (history or []) + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": response},
        ]
        return history, "", *_all_panels()

    def do_search(query):
        if not query.strip():
            return gr.update(), "", *_all_panels()
        response = engine.forced_search(query)
        return response, "", *_all_panels()

    def new_session():
        engine.end_session()
        engine.start_session()
        return [], *_all_panels()

    def apply_dopamine(fact):
        if fact.strip():
            engine.lq.dopamine(fact.strip())
            return f"Tagged as priority: {fact}", *_all_panels()
        return "Enter a fact to tag.", *_all_panels()

    def apply_correction(correction):
        if correction.strip():
            engine.lq.correct(correction.strip())
            return f"Correction applied: {correction}", *_all_panels()
        return "Enter a correction.", *_all_panels()

    def mark_good():
        engine.handle_feedback("positive")
        return _all_panels()

    # ── Layout ────────────────────────────────────────────────

    with gr.Blocks(title="Living LLM + Limbiq") as demo:
        gr.Markdown("## Living LLM — powered by **limbiq** (neurotransmitter-inspired adaptive learning)")

        with gr.Row(equal_height=True, elem_classes="main-row"):
            # ── Left: chat panel ──────────────────────────────
            with gr.Column(scale=3, elem_classes="chat-col"):
                chatbot = gr.Chatbot(
                    height=700,
                    elem_id="chatbot",
                    label="Conversation",
                )
                with gr.Row(elem_classes="input-row"):
                    msg_box = gr.Textbox(
                        placeholder="Type a message... (Enter to send)",
                        show_label=False,
                        lines=1,
                        scale=8,
                        container=False,
                    )
                    send_btn = gr.Button("Send", variant="primary", scale=1, elem_id="send-btn")

                with gr.Accordion("Web search", open=False):
                    with gr.Row():
                        search_box = gr.Textbox(
                            placeholder="Search query...",
                            show_label=False,
                            scale=7,
                            container=False,
                        )
                        search_btn = gr.Button("Search", scale=1)
                    search_result = gr.Markdown(label="Search result")

            # ── Right: limbiq panel ──────────────────────────
            with gr.Column(scale=2, elem_classes="memory-col"):
                with gr.Tabs():
                    with gr.Tab("Memory"):
                        memory_display = gr.Markdown(
                            value=_memory_md,
                            elem_classes="mem-tab-content",
                        )
                    with gr.Tab("Priority"):
                        priority_display = gr.Markdown(
                            value=_priority_md,
                            elem_classes="mem-tab-content",
                        )
                    with gr.Tab("Signals"):
                        signal_display = gr.Markdown(
                            value=_signals_md,
                            elem_classes="mem-tab-content",
                        )
                    with gr.Tab("Suppressed"):
                        suppressed_display = gr.Markdown(
                            value=_suppressed_md,
                            elem_classes="mem-tab-content",
                        )
                    with gr.Tab("Actions"):
                        gr.Markdown("**Tag priority memory (Dopamine)**")
                        dopamine_input = gr.Textbox(placeholder="Enter a fact...")
                        dopamine_btn = gr.Button("Tag as priority")
                        dopamine_status = gr.Markdown()

                        gr.Markdown("**Apply correction**")
                        correct_input = gr.Textbox(placeholder="Enter correction...")
                        correct_btn = gr.Button("Correct")
                        correct_status = gr.Markdown()

                        gr.Markdown("---")
                        good_btn = gr.Button("Last response was good")

                with gr.Row():
                    refresh_btn = gr.Button("Refresh", size="sm")
                    new_session_btn = gr.Button("New session", size="sm", variant="secondary")

        # ── Wire events ───────────────────────────────────────

        all_panel_outputs = [memory_display, priority_display, signal_display, suppressed_display]

        # Send message (button or Enter)
        send_btn.click(
            fn=chat,
            inputs=[msg_box, chatbot],
            outputs=[chatbot, msg_box, *all_panel_outputs],
        )
        msg_box.submit(
            fn=chat,
            inputs=[msg_box, chatbot],
            outputs=[chatbot, msg_box, *all_panel_outputs],
        )

        # Web search
        search_btn.click(
            fn=do_search,
            inputs=[search_box],
            outputs=[search_result, search_box, *all_panel_outputs],
        )
        search_box.submit(
            fn=do_search,
            inputs=[search_box],
            outputs=[search_result, search_box, *all_panel_outputs],
        )

        # Actions
        dopamine_btn.click(
            fn=apply_dopamine,
            inputs=dopamine_input,
            outputs=[dopamine_status, *all_panel_outputs],
        )
        correct_btn.click(
            fn=apply_correction,
            inputs=correct_input,
            outputs=[correct_status, *all_panel_outputs],
        )
        good_btn.click(fn=mark_good, outputs=all_panel_outputs)

        # Controls
        refresh_btn.click(fn=_all_panels, outputs=all_panel_outputs)
        new_session_btn.click(fn=new_session, outputs=[chatbot, *all_panel_outputs])

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
