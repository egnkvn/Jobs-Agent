#!/usr/bin/env python3
"""
Career Copilot — Resume Tailoring & Interview Preparation Agent

Architecture: DeepAgent coordinator-worker pattern
  - career-copilot  : coordinator (owns state, routes tasks, applies edits)
  - job-researcher  : fetches and normalizes job postings
  - resume-editor   : proposes structured patch operations against the resume
  - interview-designer: generates grounded behavioral/technical/whiteboard questions

Persistent storage layout (relative to current working directory):
  ./resume/       — canonical resume artifacts (master.md, parsed.json, changelog.md)
  ./job/          — job context (jd_normalized.json, gap_analysis.json)
  ./interview/    — question banks per mode
  ./session/      — session brief, active objectives, pending patch
  /memories/      — cross-session user preferences (StoreBackend)
"""

from __future__ import annotations

from pathlib import Path

from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.memory import InMemoryStore

from deepagents import create_deep_agent

from dotenv import load_dotenv
load_dotenv()

# Import utilities
from utils import make_backend, _handle_interrupt, run_interactive

# Import all tools from tools.py
from tools import (
    # Workspace helpers
    _get_workspace_path,
    get_workspace_path,
    # File I/O
    write_file,
    read_file,
    edit_file,
    # Job research
    fetch_url_content,
    extract_jd_keywords,
    # Company research
    web_search,
    fetch_ptt_posts,
    fetch_dcard_posts,
    # Resume processing
    read_pdf_resume,
    ingest_resume_text,
    render_resume_to_markdown,
    # Resume editing
    validate_resume_patch,
    score_bullet_against_jd,
    compute_gap_analysis,
    apply_resume_patch,
    # Interview preparation
    validate_question_bank,
    # Session utilities
    generate_resume_changelog_entry,
    generate_session_brief,
)

# Import all prompts from prompts.py
from prompts import (
    COORDINATOR_SYSTEM_PROMPT,
    JOB_RESEARCHER_DESCRIPTION,
    JOB_RESEARCHER_SYSTEM_PROMPT,
    RESUME_EDITOR_DESCRIPTION,
    RESUME_EDITOR_SYSTEM_PROMPT,
    INTERVIEW_DESIGNER_DESCRIPTION,
    INTERVIEW_DESIGNER_SYSTEM_PROMPT,
)


# ─────────────────────────────────────────────────────────────────────────────
# Backend and persistence setup
# ─────────────────────────────────────────────────────────────────────────────

checkpointer = MemorySaver()
store = InMemoryStore()


# ─────────────────────────────────────────────────────────────────────────────
# Subagent definitions
# ─────────────────────────────────────────────────────────────────────────────

job_researcher = {
    "name": "job-researcher",
    "description": JOB_RESEARCHER_DESCRIPTION,
    "system_prompt": JOB_RESEARCHER_SYSTEM_PROMPT,
    "tools": [fetch_url_content, extract_jd_keywords, web_search, fetch_ptt_posts, fetch_dcard_posts, get_workspace_path, write_file, read_file],
}

resume_editor = {
    "name": "resume-editor",
    "description": RESUME_EDITOR_DESCRIPTION,
    "system_prompt": RESUME_EDITOR_SYSTEM_PROMPT,
    "tools": [score_bullet_against_jd, validate_resume_patch, get_workspace_path, write_file, read_file],
}

interview_designer = {
    "name": "interview-designer",
    "description": INTERVIEW_DESIGNER_DESCRIPTION,
    "system_prompt": INTERVIEW_DESIGNER_SYSTEM_PROMPT,
    "tools": [validate_question_bank, web_search, fetch_ptt_posts, fetch_dcard_posts, fetch_url_content, get_workspace_path, write_file, read_file],
}

# ─────────────────────────────────────────────────────────────────────────────
# Coordinator agent
# ─────────────────────────────────────────────────────────────────────────────

agent = create_deep_agent(
    model="openai:gpt-5.4-nano",
    name="career-copilot",
    system_prompt=COORDINATOR_SYSTEM_PROMPT,
    tools=[
        # Workspace path management
        get_workspace_path,
        # File I/O operations
        write_file,
        read_file,
        edit_file,
        # Job ingestion (coordinator should delegate to job-researcher, not use directly)
        fetch_url_content,
        extract_jd_keywords,
        # Company & interview experience research
        web_search,
        fetch_ptt_posts,
        fetch_dcard_posts,
        # Resume processing
        read_pdf_resume,
        ingest_resume_text,
        render_resume_to_markdown,
        compute_gap_analysis,
        # Patch lifecycle
        validate_resume_patch,
        apply_resume_patch,
        generate_resume_changelog_entry,
        # Interview tools
        validate_question_bank,
        # Session utilities
        generate_session_brief,
    ],
    backend=make_backend,
    store=store,
    checkpointer=checkpointer,
    subagents=[job_researcher, resume_editor, interview_designer],
    context_schema={
        "user_id": str,
        "session_id": str,
        "active_job_url": str,
        "target_role": str,
        "target_company": str,
        "active_resume_path": str,
        "active_jd_path": str,
        "current_mode": str,  # resume_edit | interview | whiteboard | mixed
    },
    interrupt_on={
        # High risk — full control
        "write_file": {"allowed_decisions": ["approve", "edit", "reject"]},
        "edit_file": {"allowed_decisions": ["approve", "edit", "reject"]},
        # Medium risk — apply or reject
        "apply_resume_patch": {"allowed_decisions": ["approve", "reject"]},
    },
)


if __name__ == "__main__":
    run_interactive(agent, store, _get_workspace_path)
