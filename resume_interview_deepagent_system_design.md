# Resume Tailoring + Interview Preparation Agent
## Formal System Design Document (DeepAgent-based)

**Version:** 1.0  
**Date:** 2026-03-26  
**Authoring context:** Prepared for a product that supports resume tailoring, iterative resume editing, interview question generation, whiteboard practice, and mock-interview style interactions in a single conversation.

---

## 1. Executive Summary

This document proposes a production-oriented system design for a conversational career assistant built with **LangChain Deep Agents**. The system supports a user journey in which the user uploads a resume, provides a job posting URL, and then continues working in the **same thread** to:

- tailor the resume to the target role,
- add, remove, or rewrite specific resume content,
- generate behavioral, technical, and whiteboard interview questions,
- practice likely follow-up questions,
- and iteratively refine outputs over time.

The recommended architecture is:

- **one coordinator agent** that owns the user conversation and system state,
- **two required subagents**:
  - `resume-editor`
  - `interview-designer`
- **one optional but strongly recommended subagent**:
  - `job-researcher`
- **one optional later-stage subagent**:
  - `whiteboard-coach` or `mock-interviewer`
- a **filesystem-backed artifact layer** for canonical working files,
- a **persistent memory layer** for long-lived user preferences,
- and a **human-in-the-loop review path** for sensitive write operations.

The central architectural decision is that the product should **not** be implemented as two peer agents that independently manage the user relationship. Instead, it should use a **single coordinator** that maintains a canonical resume, canonical job context, and canonical session state, while delegating specialized tasks to subagents for context isolation and task specialization.

---

## 2. Problem Statement

Users preparing for a job application often need several tightly coupled workflows:

1. ingest a resume,
2. ingest a job description,
3. compare the resume with the role,
4. tailor the resume,
5. continue making local edits over multiple turns,
6. generate likely interview questions from the same role and resume,
7. practice answers or coding questions,
8. preserve user preferences across sessions.

A naïve single-agent design tends to accumulate too much context and becomes brittle as the conversation grows. A naïve two-agent design fragments ownership of state and introduces synchronization problems, especially when resume edits and interview preparation both depend on the same target job and user profile.

The design goal is therefore to create a system that:

- preserves **one coherent conversation**,
- keeps **one canonical set of application artifacts**,
- and still benefits from multi-agent specialization.

---

## 3. Goals and Non-Goals

### 3.1 Goals

The system must:

1. support a **single-thread, multi-stage** user workflow,
2. maintain a **canonical resume artifact** that can be incrementally updated,
3. maintain a **canonical job artifact** derived from the job posting,
4. allow targeted modifications such as “add this project” or “remove this bullet,”
5. generate multiple categories of interview questions from the same context,
6. isolate large intermediate work products so the main conversation remains clean,
7. persist relevant user preferences across conversations,
8. support review/approval for sensitive write actions,
9. allow structured outputs suitable for validation and downstream rendering,
10. remain extensible to mock interviews, whiteboard coaching, and multi-job management.

### 3.2 Non-Goals (for MVP)

The first version does **not** need to fully solve:

- autonomous job application submission,
- recruiter outreach automation,
- calendar scheduling,
- company-wide intelligence collection across many external sources,
- full collaborative editing with multi-user concurrency,
- perfect resume parsing for every arbitrary PDF layout,
- advanced scoring calibrated against real hiring outcomes.

These may be added later, but they should not shape the initial architecture in a way that increases complexity prematurely.

---

## 4. Why Deep Agents Is a Good Fit

Deep Agents is a strong fit for this product because it is designed for complex, multi-step tasks, includes built-in task planning, exposes filesystem tools for context management, supports subagents for context isolation, and supports persistent memory via backend routing. It also runs on LangGraph, which enables durable execution, streaming, and human-in-the-loop patterns.

For this use case, the most relevant Deep Agents capabilities are:

- **task decomposition** for multi-step workflows,
- **filesystem tools** for canonical artifacts,
- **pluggable backends** for transient vs persistent storage,
- **subagents** for isolating resume editing and interview generation work,
- **skills** for reusable domain rules,
- **interrupt-based review** for sensitive edits,
- **frontend streaming** for coordinator + subagent visibility.

This product should use Deep Agents as an **agent harness**, not merely as a prompt wrapper.

---

## 5. Architectural Principles

The system is guided by the following principles.

### 5.1 Single Ownership of Conversation

Only one top-level coordinator should directly own the user interaction loop. This avoids fragmented state, conflicting interpretations, and duplicated artifact management.

### 5.2 Canonical Artifacts over Implicit Memory

All important working state should be materialized into files rather than relying on chat history alone. The system should treat the filesystem as the source of truth for the current application workflow.

### 5.3 Specialized Workers, Not Independent Personas

Subagents are specialized workers used for context isolation and task specialization. They do not become independent, long-term conversational identities.

### 5.4 Incremental Edits over Full Regeneration

The system should favor **patch-based updates** to canonical files rather than repeatedly rewriting the entire resume or question bank.

### 5.5 Deterministic Ingestion + Agentic Refinement

Parsing and normalization tasks should be as deterministic as possible. LLM reasoning should be used where judgment is needed, such as tailoring language, finding relevance gaps, or generating interview questions.

### 5.6 Reviewable Writes

Sensitive write operations should be reviewable. The system should make edits legible and reversible.

---

## 6. High-Level Architecture

### 6.1 Component Diagram

```text
User
  ↓
Frontend (chat + diff viewer + question bank + subagent progress)
  ↓
Career Copilot Coordinator (DeepAgent main agent)
  ├─ Resume Editor subagent
  ├─ Interview Designer subagent
  ├─ Job Researcher subagent (optional, recommended)
  └─ Whiteboard Coach / Mock Interviewer (optional, later)
  ↓
Tools + FileSystem + Memory + Validation + Export pipeline
```

### 6.2 Coordinator-Worker Pattern

The main agent acts as coordinator. It interprets the user request, determines whether the request is a resume edit, question generation task, job research task, or mixed request, and delegates to subagents where appropriate. Subagents operate in isolation and return concise outputs to the coordinator. The coordinator then updates canonical artifacts and responds to the user.

This matches the intended coordinator-worker design of Deep Agents, where the main agent delegates specialized work to isolated subagents and the frontend can visualize both coordinator messages and subagent progress.

---

## 7. Agent Topology and Responsibilities

## 7.1 Coordinator Agent: `career-copilot`

### Primary Responsibility

Maintain session coherence, route tasks, manage canonical artifacts, and synthesize final responses.

### Responsibilities

- maintain the overall conversation state,
- decide when to delegate,
- own the canonical resume path,
- own the canonical job context,
- own the session brief and active objectives,
- apply validated patches,
- explain changes to the user,
- preserve preferences to memory when appropriate.

### The Coordinator Should Not

- perform every large rewrite itself,
- hold every intermediate result in prompt context,
- allow multiple agents to mutate canonical state independently.

---

## 7.2 Subagent: `resume-editor`

### Purpose

Tailor and revise the resume against a specific job context.

### Responsibilities

- rewrite summary / headline,
- rewrite bullets with stronger action-impact framing,
- align wording to target job requirements,
- strengthen metrics, ownership, and outcomes,
- perform targeted modifications to specific sections,
- propose edits as structured patch operations,
- provide compact rationale for each change.

### Preferred Output Mode

Structured output, not raw prose only.

#### Example Patch Contract

```json
{
  "operations": [
    {
      "op": "replace_section",
      "target": "experience.company_x.bullets[1]",
      "before": "Built internal tools...",
      "after": "Built internal tooling that reduced labeling time by 35% and improved annotation throughput for the production ML pipeline."
    }
  ],
  "rationale": [
    "Adds measurable business impact",
    "Aligns wording with target JD emphasis on production ML systems"
  ]
}
```

### Why Patch-Based Output Is Preferred

- safer than unconstrained full rewrites,
- easier to validate,
- easier to show in a diff viewer,
- easier to accept or reject with human review,
- easier to compose over many turns.

---

## 7.3 Subagent: `interview-designer`

### Purpose

Generate interview materials grounded in the target role and the current resume.

### Responsibilities

- generate behavioral questions,
- generate technical deep-dive questions,
- generate project-specific follow-up questions,
- generate whiteboard/coding prompts,
- annotate each question with reasoning and signals,
- optionally generate ideal answer points or rubrics.

### Preferred Output Mode

Structured question sets.

#### Example Question Contract

```json
{
  "mode": "behavioral",
  "questions": [
    {
      "question": "Tell me about a time you improved a weak baseline into a production-ready system.",
      "why_this_might_be_asked": "The role emphasizes ownership, iteration, and production impact.",
      "signals": ["ownership", "experimentation", "production impact"],
      "follow_ups": [
        "How did you choose your evaluation metric?",
        "What trade-offs did you make to move faster?"
      ]
    }
  ]
}
```

### Extension Path

The same subagent can initially cover behavioral, technical, and whiteboard question generation. In later phases, high-interaction mock interview or code review behavior can be moved into a dedicated subagent.

---

## 7.4 Subagent: `job-researcher` (Optional but Recommended)

### Purpose

Normalize and enrich job context so other subagents do not repeatedly parse raw job text.

### Responsibilities

- fetch job posting content,
- extract responsibilities and qualifications,
- normalize the posting into machine-usable fields,
- identify explicit vs implicit requirements,
- optionally summarize company or product context,
- optionally infer likely interview focus areas.

### Why This Subagent Is Recommended

Both resume tailoring and interview generation depend on the same target job. Centralizing this analysis reduces redundant work, reduces prompt size, and improves consistency.

---

## 7.5 Subagent: `whiteboard-coach` / `mock-interviewer` (Later Phase)

### Purpose

Support interactive practice beyond static question generation.

### Responsibilities

- present coding or system-design problems,
- evaluate user answers step by step,
- probe edge cases and complexity,
- provide structured feedback,
- score performance against a rubric.

### Recommended Timing

Do not implement this in the initial MVP unless interactive interview practice is the primary product differentiator.

---

## 8. Storage Model

The product should separate **ephemeral application state** from **cross-session memory**.

### 8.1 Workspace Filesystem (Ephemeral / Thread-Scoped)

The workspace stores the current application’s working files.

```text
/workspace/
  resume/
    original.pdf
    parsed.json
    master.md
    latest.docx
    latest.pdf
    changelog.md

  job/
    source_url.txt
    jd_raw.md
    jd_normalized.json
    company_notes.md
    gap_analysis.json

  interview/
    behavior_questions.json
    technical_questions.json
    whiteboard_questions.json
    mock_feedback.md

  session/
    brief.md
    active_objectives.json
    pending_patch.json
```

### 8.2 Persistent Memory (Cross-Thread)

Persistent user facts should be routed to `/memories/`.

```text
/memories/
  user_profile.md
  writing_preferences.md
  recurring_strengths.md
  recurring_weaknesses.md
  past_target_roles.md
```

### 8.3 Backend Strategy

A `CompositeBackend` should be used so that:

- `/workspace/` remains thread-scoped and ephemeral,
- `/memories/` persists across conversations,
- the agent can treat both through the same filesystem abstraction.

This is the correct mental model for a product where “this application” and “this user” are related but not identical scopes.

---

## 9. Runtime State Model

The coordinator should also maintain lightweight runtime context.

```python
context_schema = {
    "user_id": str,
    "session_id": str,
    "active_job_url": str,
    "target_role": str,
    "target_company": str,
    "active_resume_path": str,
    "active_jd_path": str,
    "current_mode": str,
}
```

### 9.1 Meaning of `current_mode`

Possible values:

- `resume_edit`
- `interview`
- `whiteboard`
- `mixed`

### 9.2 Context Propagation

The coordinator should pass runtime context such as `user_id`, `session_id`, and active artifact paths. Deep Agents propagates parent runtime context to subagents, which allows tools inside subagents to access the same shared context.

### 9.3 Agent-Specific Namespaced Context

If a specific subagent needs local configuration, use namespaced keys such as:

- `resume-editor:max_rewrite_depth`
- `interview-designer:difficulty`
- `job-researcher:max_pages`

This keeps shared and agent-specific context cleanly separated.

---

## 10. Canonical Data Contracts

To keep the system maintainable, core artifacts should be represented by stable schemas.

## 10.1 Resume Contract

```json
{
  "basics": {
    "name": "",
    "email": "",
    "phone": "",
    "location": "",
    "links": []
  },
  "summary": "",
  "skills": [],
  "experience": [
    {
      "company": "",
      "title": "",
      "start_date": "",
      "end_date": "",
      "bullets": []
    }
  ],
  "projects": [],
  "education": [],
  "awards": []
}
```

## 10.2 Job Description Contract

```json
{
  "job_title": "",
  "company": "",
  "location": "",
  "employment_type": "",
  "seniority": "",
  "responsibilities": [],
  "required_qualifications": [],
  "preferred_qualifications": [],
  "keywords": [],
  "inferred_focus_areas": [],
  "source_url": ""
}
```

## 10.3 Gap Analysis Contract

```json
{
  "matched_strengths": [],
  "coverage_gaps": [],
  "missing_keywords": [],
  "resume_sections_to_strengthen": [],
  "interview_risk_areas": []
}
```

## 10.4 Patch Operation Contract

```json
{
  "operations": [
    {
      "op": "replace_section | insert_after | delete_section | append_bullet",
      "target": "section path",
      "before": "optional previous content",
      "after": "optional new content"
    }
  ],
  "rationale": [],
  "confidence": 0.0
}
```

## 10.5 Question Bank Contract

```json
{
  "mode": "behavioral | technical | whiteboard | mixed",
  "difficulty": "easy | medium | hard",
  "questions": [
    {
      "question": "",
      "why_this_might_be_asked": "",
      "signals": [],
      "follow_ups": [],
      "ideal_answer_points": []
    }
  ]
}
```

---

## 11. Core Workflows

## 11.1 Workflow A: Initial Ingestion

### Trigger

User uploads a resume and provides a job posting URL.

### Steps

1. ingest resume file,
2. extract raw text or structured content,
3. convert into normalized resume JSON and markdown,
4. fetch and parse job posting,
5. normalize the job description,
6. compute initial gap analysis,
7. generate a short session brief,
8. store canonical artifacts in `/workspace/`.

### Outputs

- `/workspace/resume/parsed.json`
- `/workspace/resume/master.md`
- `/workspace/job/jd_normalized.json`
- `/workspace/job/gap_analysis.json`
- `/workspace/session/brief.md`

### Design Note

This workflow should rely more heavily on deterministic parsing and validation than on open-ended generation.

---

## 11.2 Workflow B: Resume Tailoring

### Trigger

User asks the system to tailor or improve the resume.

### Steps

1. coordinator reads canonical resume and normalized JD,
2. coordinator decides whether to delegate,
3. `resume-editor` generates structured patch proposals,
4. validation layer checks patch integrity,
5. human review may be requested,
6. patch is applied to canonical resume,
7. changelog is updated,
8. rendered export is refreshed if needed.

### Outputs

- updated `/workspace/resume/master.md`
- updated `/workspace/resume/changelog.md`
- optional refreshed `/workspace/resume/latest.docx`
- optional refreshed `/workspace/resume/latest.pdf`

---

## 11.3 Workflow C: Targeted Resume Edit

### Trigger

User says things like:

- “Add my LLM evaluation project.”
- “Remove the internship bullet about data cleaning.”
- “Rewrite this summary to sound more senior.”

### Steps

1. coordinator extracts the intent and scope,
2. unresolved edit instructions are written to `pending_patch.json`,
3. only the relevant resume sections are loaded,
4. `resume-editor` produces a targeted patch,
5. validation runs,
6. patch is applied,
7. the user sees the diff and rationale.

### Rationale

Targeted editing is much safer and cheaper than regenerating the whole resume.

---

## 11.4 Workflow D: Behavioral Question Generation

### Trigger

User asks for likely behavioral questions.

### Steps

1. coordinator loads resume + JD + gap analysis + preferences,
2. coordinator invokes `interview-designer`,
3. subagent returns structured question set,
4. system stores result in `behavior_questions.json`,
5. coordinator formats a user-friendly response.

### Optional Enhancements

- tag questions by competency,
- mark risk areas,
- attach sample answer skeletons.

---

## 11.5 Workflow E: Whiteboard Question Generation

### Trigger

User asks to practice whiteboard or coding questions.

### Steps

1. coordinator identifies desired format and difficulty,
2. invokes `interview-designer` or `whiteboard-coach`,
3. stores result in `whiteboard_questions.json`,
4. if the user begins solving, transition to interactive evaluation mode.

---

## 11.6 Workflow F: Preference Learning

### Trigger

User reveals stable preferences, such as:

- “Always output English resume bullets.”
- “Do not mention GPA.”
- “Prefer concise bullets with metrics.”

### Steps

1. coordinator identifies whether the preference is stable and reusable,
2. write/update the appropriate memory file,
3. incorporate the preference in subsequent edits and question generation.

---

## 12. Tools and Interfaces

## 12.1 Coordinator Tool Set

Recommended tool surface:

- `ingest_resume(file)`
- `fetch_job_posting(url)`
- `parse_job_description(raw_html_or_text)`
- `compute_gap_analysis(resume_json, jd_json)`
- `apply_resume_patch(patch)`
- `render_resume(format)`
- `save_user_preference(key, value)`
- `load_question_bank(mode)`

The coordinator tool set should remain small and orchestration-oriented.

## 12.2 Resume Editor Tool Set

- `read_resume_section(section_id)`
- `read_jd_requirements()`
- `suggest_resume_patch()`
- `score_bullet_against_jd()`

## 12.3 Interview Designer Tool Set

- `read_resume_summary()`
- `read_gap_analysis()`
- `generate_questions(mode, count, difficulty)`
- `save_question_bank(mode, data)`

## 12.4 Job Researcher Tool Set

- `fetch_url()`
- `extract_main_content()`
- `normalize_jd()`
- `summarize_company_context()`

---

## 13. Skills Strategy

This system should use both **subagents** and **skills**.

### 13.1 Why Skills Matter Here

Skills are reusable capability bundles that can hold workflow instructions, templates, rubrics, and supporting assets. Deep Agents loads them progressively, which means the agent can defer loading detailed skill content until it is actually relevant.

This makes skills ideal for:

- ATS rewrite heuristics,
- bullet-writing rules,
- behavioral interview rubrics,
- question design guides,
- formatting and tone preferences.

### 13.2 Proposed Skills Layout

```text
/skills/
  main/
    routing.md
    session-management.md

  resume/
    ats-alignment.md
    bullet-rewrite.md
    resume-style-guide.md
    section-tailoring.md

  interview/
    behavioral-star.md
    technical-question-design.md
    whiteboard-question-design.md
    mock-evaluation-rubric.md
```

### 13.3 Skills Inheritance Rule

The main agent and general-purpose subagent can inherit main skills, but **custom subagents do not inherit skills by default**. Therefore, custom subagents should be explicitly configured with their own `skills` lists.

### 13.4 Recommended Usage

- main agent: routing, session, memory hygiene skills,
- `resume-editor`: resume-specific skills only,
- `interview-designer`: interview-specific skills only,
- `job-researcher`: research/JD parsing skills only.

This avoids accidental context leakage and keeps each worker specialized.

---

## 14. Human-in-the-Loop Design

Because this product edits user documents, human review should be treated as a first-class feature rather than a later add-on.

### 14.1 Operations That Should Be Reviewable

At minimum:

- `write_file`
- `edit_file`
- `apply_resume_patch`
- `render_resume` when it overwrites an export

Potential future tools that should definitely require approval:

- `send_email`
- `submit_application`
- `delete_file`

### 14.2 Deep Agents Configuration Pattern

Use `interrupt_on` to configure review behavior. Human-in-the-loop requires a checkpointer.

### 14.3 Risk-Tiered Review Policy

Suggested policy:

- **High risk**: approve / edit / reject
  - destructive edits,
  - file deletion,
  - external sends
- **Medium risk**: approve / reject
  - writing resume exports,
  - patch application
- **Low risk**: no interrupt
  - reads,
  - scoring,
  - temporary computation

### 14.4 UX Requirement

Every reviewable edit should display:

- the tool action,
- the target file/section,
- the proposed arguments,
- the before/after diff,
- the rationale,
- the available human decisions.

---

## 15. Frontend and User Experience

A three-pane layout is recommended.

### 15.1 Left Pane: Conversation

Displays the user/coordinator chat thread.

### 15.2 Center Pane: Artifact Viewer

Displays the canonical resume, diffs, and exported versions.

### 15.3 Right Pane: Agent Activity

Displays:

- subagent progress,
- task list / todos,
- generated question banks,
- interrupt/review requests.

### 15.4 Streaming

Deep Agents frontend patterns support showing coordinator and subagent streams in real time. `useStream` can expose subagent-specific state and todo state, which is useful for making the system feel inspectable rather than opaque.

---

## 16. Prompting and Routing Strategy

## 16.1 Coordinator Prompt Responsibilities

The coordinator system prompt should explicitly instruct the agent to:

- maintain a single canonical resume,
- maintain a single active target job,
- prefer patch-based editing,
- delegate specialized work to subagents,
- preserve stable user preferences,
- ask for approval before sensitive writes,
- summarize changes in user-friendly terms.

## 16.2 Subagent Prompt Responsibilities

Each subagent prompt should be narrow and action-oriented.

### `resume-editor`

Should emphasize:

- precise, targeted edits,
- measurable impact,
- alignment with the current JD,
- concise rationale,
- structured patch output.

### `interview-designer`

Should emphasize:

- role-grounded question generation,
- competency coverage,
- follow-up depth,
- structured question output,
- avoiding generic questions when resume evidence suggests deeper probes.

### `job-researcher`

Should emphasize:

- clean normalization of the job posting,
- separation of explicit vs inferred requirements,
- concise summaries,
- avoiding speculative claims unless labeled as inference.

---

## 17. Validation Layer

The system should include deterministic validation between subagent output and canonical file mutation.

### 17.1 Why Validation Is Necessary

Subagents may return malformed or overly broad edits. Validation prevents accidental corruption of artifacts.

### 17.2 Patch Validation Rules

Before applying a patch:

- ensure target section exists,
- ensure operation type is allowed,
- ensure `before` matches or approximately matches current content when required,
- reject patches that remove entire sections unless explicitly requested,
- reject patches that exceed size limits for “targeted edit” mode,
- log patch provenance.

### 17.3 Question Set Validation Rules

Before saving a question bank:

- ensure required fields exist,
- ensure the requested mode matches the result,
- ensure duplicate questions are deduplicated,
- ensure output count matches or slightly exceeds request,
- ensure questions do not contain disallowed personal data leakage.

---

## 18. Export and Rendering

The canonical resume should be stored in a text-friendly format such as markdown, then rendered into user-facing formats.

### 18.1 Recommended Source of Truth

`/workspace/resume/master.md`

### 18.2 Render Targets

- DOCX
- PDF
- possibly plain text or email-safe versions later

### 18.3 Rendering Pipeline

1. load canonical markdown,
2. apply formatting template,
3. generate DOCX,
4. optionally generate PDF,
5. store under `/workspace/resume/`.

### 18.4 Recommendation

Do not let subagents write directly to final DOCX/PDF. They should only propose structured edits against canonical content.

---

## 19. Observability and Tracing

The implementation should include tracing and evaluation hooks from day one.

### 19.1 Recommended Observability

Use LangSmith tracing to inspect:

- delegation decisions,
- tool call sequences,
- patch proposals,
- memory writes,
- human review interruptions,
- export failures.

### 19.2 Metrics to Track

- number of coordinator-only turns vs delegated turns,
- patch acceptance rate,
- average number of edits per session,
- repeated user corrections,
- memory write acceptance rate,
- question-bank reuse rate,
- export success rate,
- end-to-end latency by workflow.

---

## 20. Evaluation Framework

The system should be evaluated along both product and agent dimensions.

### 20.1 Resume Quality Evaluation

Metrics:

- relevance to JD,
- specificity of impact statements,
- keyword coverage,
- readability,
- factual preservation,
- overclaim rate.

### 20.2 Interview Question Evaluation

Metrics:

- relevance to target role,
- grounding in the user’s resume,
- diversity across competencies,
- depth of follow-up questions,
- absence of generic filler.

### 20.3 Interaction Quality Evaluation

Metrics:

- correctness of routing,
- edit reversibility,
- user-perceived coherence across turns,
- correct use of preferences,
- human review usability.

---

## 21. Security, Safety, and Privacy

### 21.1 User Data Sensitivity

Resumes often contain personal data including names, phone numbers, emails, employers, education history, and potentially protected characteristics.

### 21.2 Controls

- minimize retention outside explicit memory scope,
- avoid writing personal data into long-term memory unless needed,
- isolate external fetch behavior to job URLs and approved sources,
- log sensitive write operations,
- allow the user to inspect canonical files,
- support deletion of workspace artifacts.

### 21.3 Safety Note on Edits

The system should avoid inventing achievements or metrics. Suggested additions should be clearly marked as drafts when the source resume does not substantiate them.

---

## 22. Risks and Mitigations

## 22.1 Risk: Resume Corruption Through Overwrite

**Mitigation:** patch-based editing, validation, changelog, review step.

## 22.2 Risk: Context Bloat

**Mitigation:** offload canonical files to filesystem, use specialized subagents, keep subagent return values concise.

## 22.3 Risk: Inconsistent Job Understanding Across Tasks

**Mitigation:** normalize the JD once, store it canonically, reuse it for both resume and interview tasks.

## 22.4 Risk: Preference Drift

**Mitigation:** store stable preferences in `/memories/`, keep session-level overrides separate.

## 22.5 Risk: Generic Interview Questions

**Mitigation:** require the question generator to use resume evidence, gap analysis, and role focus areas.

## 22.6 Risk: Excessive System Complexity Too Early

**Mitigation:** launch with one coordinator and two subagents; keep job ingestion deterministic first.

---

## 23. MVP Scope Recommendation

### 23.1 MVP Components

Implement:

- `career-copilot` coordinator,
- `resume-editor` subagent,
- `interview-designer` subagent,
- deterministic resume/JD ingestion,
- canonical workspace filesystem,
- long-term preference memory,
- patch review UX,
- resume export pipeline.

### 23.2 MVP User Flows

Support:

1. upload resume,
2. paste JD URL,
3. tailor resume,
4. add/remove/rewrite selected content,
5. generate behavioral questions,
6. generate technical questions,
7. generate whiteboard prompts.

### 23.3 What to Delay

Delay until later:

- live mock interviewer,
- broad company research,
- auto-application,
- multi-job portfolio dashboards,
- advanced scoring models.

---

## 24. Roadmap

## Phase 1: Core Application Assistant

- ingestion,
- canonical artifacts,
- resume tailoring,
- question generation,
- export,
- memory of preferences.

## Phase 2: Better Research and Practice

- `job-researcher`,
- better competency tagging,
- answer rubrics,
- simple mock interview flows.

## Phase 3: Advanced Coaching

- dedicated `whiteboard-coach`,
- interactive answer grading,
- role-specific practice plans,
- multi-job application tracking.

---

## 25. Recommended Implementation Shape

A practical implementation shape is:

- use Deep Agents for the conversational harness,
- use deterministic helper tools for parsing and rendering,
- use subagents for specialized cognitive work,
- use backend routing for memory scope,
- optionally wrap the overall product with a custom LangGraph workflow later if stricter deterministic sequencing becomes necessary.

### When to Stay with Pure Deep Agents

Stay with a direct Deep Agents implementation when:

- the workflow is still heavily conversational,
- routing flexibility matters,
- the main pain point is context management,
- and the product is still evolving.

### When to Wrap with LangGraph Workflow

Introduce a higher-level custom workflow when:

- ingestion must always happen before any other step,
- validation and export become more complex,
- multiple deterministic gates are required,
- auditability requirements become stricter.

---

## 26. Example Pseudo-Code Skeleton

```python
from deepagents import create_deep_agent
from deepagents.backends import CompositeBackend, StateBackend, StoreBackend
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.memory import InMemoryStore

checkpointer = MemorySaver()
store = InMemoryStore()


def make_backend(runtime):
    return CompositeBackend(
        default=StateBackend(runtime),
        routes={
            "/memories/": StoreBackend(runtime)
        }
    )


resume_editor = {
    "name": "resume-editor",
    "description": "Tailor and revise the canonical resume for the active target job.",
    "system_prompt": (
        "Return targeted, structured patch operations. Prefer precise edits over full rewrites. "
        "Explain why each change improves alignment with the job description."
    ),
    "tools": [
        read_resume_section,
        read_jd_requirements,
        suggest_resume_patch,
        score_bullet_against_jd,
    ],
    "skills": ["/skills/resume/"],
}


interview_designer = {
    "name": "interview-designer",
    "description": "Generate behavioral, technical, and whiteboard interview materials from the active resume and job.",
    "system_prompt": (
        "Return structured question sets with rationale, signals, and follow-up prompts. "
        "Ground questions in the target job and the user's actual resume content."
    ),
    "tools": [
        read_resume_summary,
        read_gap_analysis,
        generate_questions,
        save_question_bank,
    ],
    "skills": ["/skills/interview/"],
}


agent = create_deep_agent(
    model="openai:gpt-5",
    system_prompt=(
        "You are a career copilot. Maintain one canonical resume and one active target job context. "
        "Delegate specialized work to subagents. Prefer patch-based edits. Save stable user preferences "
        "to /memories/. Use human review for sensitive file modifications."
    ),
    backend=make_backend,
    store=store,
    checkpointer=checkpointer,
    subagents=[resume_editor, interview_designer],
    skills=["/skills/main/"],
    context_schema={
        "user_id": str,
        "session_id": str,
        "active_job_url": str,
        "target_role": str,
        "target_company": str,
        "active_resume_path": str,
        "active_jd_path": str,
        "current_mode": str,
    },
    interrupt_on={
        "write_file": {"allowed_decisions": ["approve", "reject"]},
        "edit_file": {"allowed_decisions": ["approve", "edit", "reject"]},
        "apply_resume_patch": {"allowed_decisions": ["approve", "edit", "reject"]},
        "render_resume": {"allowed_decisions": ["approve", "reject"]},
    },
)
```

---

## 27. Final Recommendation

The best architecture for this product is:

- **one main coordinator agent** that owns the user experience,
- **two specialized subagents** as the initial minimum viable multi-agent design,
- **filesystem-backed canonical artifacts** for the active application,
- **persistent `/memories/`** for reusable preferences,
- **skills** to hold stable rubrics and domain workflows,
- **patch-based resume editing** rather than full rewrites,
- **structured question banks** rather than ad hoc prose-only generation,
- and **human-in-the-loop review** for sensitive state mutations.

The most important design decision can be summarized in one sentence:

> The system should maintain one canonical resume and one canonical job context under a single coordinator, while using subagents as specialized workers rather than independent user-facing personas.

---

## 28. References

Validated against the official LangChain documentation on 2026-03-26.

- Deep Agents overview: https://docs.langchain.com/oss/python/deepagents/overview
- Deep Agents subagents: https://docs.langchain.com/oss/python/deepagents/subagents
- Deep Agents skills: https://docs.langchain.com/oss/python/deepagents/skills
- Deep Agents long-term memory: https://docs.langchain.com/oss/python/deepagents/long-term-memory
- Deep Agents human-in-the-loop: https://docs.langchain.com/oss/python/deepagents/human-in-the-loop
- Deep Agents frontend overview: https://docs.langchain.com/oss/python/deepagents/frontend/overview
- LangChain multi-agent patterns: https://docs.langchain.com/oss/python/langchain/multi-agent
