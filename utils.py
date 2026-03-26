#!/usr/bin/env python3
"""
Utility functions for Career Copilot CLI and backend setup.
"""

from __future__ import annotations

import uuid
from typing import Any

from deepagents.backends import CompositeBackend, StateBackend, StoreBackend
from langgraph.types import Command


def make_backend(runtime: Any) -> CompositeBackend:
    """
    Composite backend:
      default paths → StateBackend  (ephemeral, thread-scoped)
      /memories/    → StoreBackend  (persistent across threads)
    """
    return CompositeBackend(
        default=StateBackend(runtime),
        routes={"/memories/": StoreBackend(runtime)},
    )


def _handle_interrupt(interrupt_value: dict) -> list[dict]:
    """
    Present pending tool-call interrupts to the user and collect y/edit/n
    decisions. Returns a list of decision dicts for Command(resume={"decisions": [...]}).
    """
    action_requests: list[dict] = interrupt_value.get("action_requests", [])
    review_configs: list[dict] = interrupt_value.get("review_configs", [])

    config_map: dict[str, list[str]] = {}
    for cfg in review_configs:
        name = cfg.get("action_name", "")
        config_map[name] = cfg.get("allowed_decisions", ["approve", "edit", "reject"])

    decisions: list[dict] = []

    print("\n" + "=" * 60)
    print("  ⚠  HUMAN REVIEW REQUIRED")
    print("=" * 60)

    for i, action in enumerate(action_requests, start=1):
        tool_name = action.get("name", "unknown")
        tool_args = action.get("args", {})
        allowed = config_map.get(tool_name, ["approve", "reject"])
        
        # Map short inputs to full decision types
        short_to_full = {
            "y": "approve",
            "n": "reject",
            "edit": "edit"
        }
        allowed_short = [k for k, v in short_to_full.items() if v in allowed]

        print(f"\n[{i}/{len(action_requests)}] Tool: {tool_name}")
        print("  Arguments:")
        for k, v in tool_args.items():
            val_str = str(v)
            if len(val_str) > 200:
                val_str = val_str[:200] + "…"
            print(f"    {k}: {val_str}")
        print(f"  Options: {'/'.join(allowed_short)}")

        while True:
            choice = input(f"\n  Decision [{'/'.join(allowed_short)}]: ").strip().lower()
            if choice in allowed_short:
                break
            print(f"  Please enter one of: {', '.join(allowed_short)}")

        # Map short input to full decision type
        full_choice = short_to_full[choice]

        if full_choice == "approve":
            decisions.append({"type": "approve"})
        elif full_choice == "reject":
            decisions.append({"type": "reject"})
        elif full_choice == "edit":
            print("  Enter edited arguments as key=value pairs (one per line).")
            print("  Press Enter on an empty line when done.")
            edited_args = dict(tool_args)
            while True:
                line = input("  > ").strip()
                if not line:
                    break
                if "=" in line:
                    k, _, v = line.partition("=")
                    edited_args[k.strip()] = v.strip()
            decisions.append({
                "type": "edit",
                "edited_action": {"name": tool_name, "args": edited_args},
            })
        else:
            decisions.append({"type": "reject"})

    print("=" * 60 + "\n")
    return decisions


def run_interactive(agent: Any, store: Any, _get_workspace_path: callable) -> None:
    """Run Career Copilot in an interactive CLI session with full interrupt support."""
    session_id = str(uuid.uuid4())[:8]
    thread_id = f"career-copilot-{session_id}"

    config = {
        "configurable": {"thread_id": thread_id},
        "context": {
            "user_id": "local-user",
            "session_id": session_id,
            "active_job_url": "",
            "target_role": "",
            "target_company": "",
            "active_resume_path": _get_workspace_path("resume", "master.md"),
            "active_jd_path": _get_workspace_path("job", "jd_normalized.json"),
            "current_mode": "mixed",
        },
    }

    print()
    print("═" * 65)
    print("  Career Copilot — Resume Tailoring & Interview Prep Agent")
    print("═" * 65)
    print(f"  Session: {session_id}  |  Thread: {thread_id}")
    print("─" * 65)
    print("  Get started:")
    print("  1. Provide a job posting URL (resume will be read from CV.pdf).")
    print("  2. Ask to tailor your resume or generate interview questions.")
    print("  3. Ask 'What do people say about interviewing at <company>?'")
    print("     to search PTT, Dcard, Glassdoor, and the web for interview experiences.")
    print("  Type 'quit' or 'exit' to end the session.")
    print("─" * 65)
    print()

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() in {"quit", "exit"}:
            print("Goodbye!")
            break

        try:
            result = agent.invoke(
                {"messages": [{"role": "user", "content": user_input}]},
                config=config,
                version="v2",
            )

            # Handle any human-in-the-loop interrupts
            while getattr(result, "interrupts", None):
                interrupt_value = result.interrupts[0].value
                decisions = _handle_interrupt(interrupt_value)
                result = agent.invoke(
                    Command(resume={"decisions": decisions}),
                    config=config,
                    version="v2",
                )

            # Extract and display the final assistant message
            messages = result.value.get("messages", []) if hasattr(result, "value") else result.get("messages", [])
            if messages:
                last = messages[-1]
                if hasattr(last, "content"):
                    content = last.content
                    if isinstance(content, list):
                        text_parts = [
                            b.get("text", "")
                            for b in content
                            if isinstance(b, dict) and b.get("type") == "text"
                        ]
                        reply = "\n".join(text_parts)
                    else:
                        reply = str(content)
                    print(f"\nCopilot: {reply}\n")

        except Exception as exc:
            print(f"\nERROR: {exc}\n")
            import traceback
            traceback.print_exc()
