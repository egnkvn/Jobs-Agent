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

import json
import os
import re
import uuid
from datetime import datetime
from typing import Literal
from pathlib import Path

import requests
from bs4 import BeautifulSoup
from langchain.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.memory import InMemoryStore
from langgraph.types import Command
from pydantic import BaseModel, Field
from pypdf import PdfReader

from deepagents import create_deep_agent
from deepagents.backends import CompositeBackend, StateBackend, StoreBackend

from dotenv import load_dotenv
load_dotenv()

# ─────────────────────────────────────────────────────────────────────────────
# Workspace path helpers
# ─────────────────────────────────────────────────────────────────────────────

WORKSPACE_ROOT = Path.cwd()

def ensure_workspace_dir(subdir: str) -> Path:
    """
    Ensure a workspace subdirectory exists and return its Path.
    Creates the directory if it doesn't exist.
    
    Args:
        subdir: Subdirectory name (e.g., 'resume', 'job', 'interview', 'session')
    """
    path = WORKSPACE_ROOT / subdir
    path.mkdir(parents=True, exist_ok=True)
    return path

def _get_workspace_path(subdir: str, filename: str) -> str:
    """
    Internal function to get the full path to a workspace file, ensuring the directory exists.
    
    Args:
        subdir: Subdirectory name (e.g., 'resume', 'job', 'interview', 'session')
        filename: File name (e.g., 'master.md', 'jd_normalized.json')
    
    Returns:
        Full path string to the file (e.g., './resume/master.md')
    """
    dir_path = ensure_workspace_dir(subdir)
    return str(dir_path / filename)

@tool
def get_workspace_path(subdir: str, filename: str) -> str:
    """
    Get the full path to a workspace file, ensuring the directory exists.
    This tool automatically creates subdirectories if they don't exist.
    
    Args:
        subdir: Subdirectory name (e.g., 'resume', 'job', 'interview', 'session')
        filename: File name (e.g., 'master.md', 'jd_normalized.json')
    
    Returns:
        Full path string to the file (e.g., './resume/master.md')
    """
    return _get_workspace_path(subdir, filename)

@tool
def write_file(file_path: str, content: str) -> str:
    """
    Write content to a file in the workspace. Creates parent directories if needed.
    Use this to save all workspace artifacts (job descriptions, resumes, patches, etc.).
    
    Args:
        file_path: Full path to the file (can be relative or absolute).
                   For workspace files, use get_workspace_path() first to get the correct path.
        content: String content to write to the file.
    
    Returns:
        Success message with the file path, or an error message.
    """
    try:
        # Handle relative paths from current working directory
        if not os.path.isabs(file_path):
            file_path = str(WORKSPACE_ROOT / file_path)
        
        # Ensure parent directory exists
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Write the file
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        return f"SUCCESS: Wrote {len(content)} characters to {file_path}"
    except Exception as exc:
        return f"ERROR writing to {file_path}: {exc}"

@tool
def read_file(file_path: str) -> str:
    """
    Read content from a file in the workspace.
    
    Args:
        file_path: Full path to the file (can be relative or absolute).
                   For workspace files, use get_workspace_path() first to get the correct path.
    
    Returns:
        File content as a string, or an error message if the file doesn't exist.
    """
    try:
        # Handle relative paths from current working directory
        if not os.path.isabs(file_path):
            file_path = str(WORKSPACE_ROOT / file_path)
        
        if not os.path.exists(file_path):
            return f"ERROR: File not found at {file_path}"
        
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        return content
    except Exception as exc:
        return f"ERROR reading {file_path}: {exc}"

@tool
def edit_file(file_path: str, old_content: str, new_content: str) -> str:
    """
    Edit a file by replacing old_content with new_content.
    This is safer than overwriting the entire file when making targeted changes.
    
    Args:
        file_path: Full path to the file (can be relative or absolute).
        old_content: The exact text to find and replace (must exist in the file).
        new_content: The new text to replace it with.
    
    Returns:
        Success message, or an error if old_content wasn't found or file doesn't exist.
    """
    try:
        # Handle relative paths from current working directory
        if not os.path.isabs(file_path):
            file_path = str(WORKSPACE_ROOT / file_path)
        
        if not os.path.exists(file_path):
            return f"ERROR: File not found at {file_path}"
        
        # Read current content
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check if old_content exists
        if old_content not in content:
            return f"ERROR: old_content not found in {file_path}. File may have changed."
        
        # Replace and write back
        updated_content = content.replace(old_content, new_content, 1)
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(updated_content)
        
        return f"SUCCESS: Edited {file_path} (replaced {len(old_content)} chars with {len(new_content)} chars)"
    except Exception as exc:
        return f"ERROR editing {file_path}: {exc}"

# ─────────────────────────────────────────────────────────────────────────────
# Canonical data schemas
# ─────────────────────────────────────────────────────────────────────────────


class ResumeBasics(BaseModel):
    name: str = ""
    email: str = ""
    phone: str = ""
    location: str = ""
    links: list[str] = []


class ExperienceEntry(BaseModel):
    company: str = ""
    title: str = ""
    start_date: str = ""
    end_date: str = ""
    bullets: list[str] = []


class ResumeSchema(BaseModel):
    basics: ResumeBasics = Field(default_factory=ResumeBasics)
    summary: str = ""
    skills: list[str] = []
    experience: list[ExperienceEntry] = []
    projects: list[dict] = []
    education: list[dict] = []
    awards: list[dict] = []


class JobDescriptionSchema(BaseModel):
    job_title: str = ""
    company: str = ""
    location: str = ""
    employment_type: str = ""
    seniority: str = ""
    responsibilities: list[str] = []
    required_qualifications: list[str] = []
    preferred_qualifications: list[str] = []
    keywords: list[str] = []
    inferred_focus_areas: list[str] = []
    source_url: str = ""


class GapAnalysisSchema(BaseModel):
    matched_strengths: list[str] = []
    coverage_gaps: list[str] = []
    missing_keywords: list[str] = []
    resume_sections_to_strengthen: list[str] = []
    interview_risk_areas: list[str] = []


class InterviewExperienceEntry(BaseModel):
    source: str = ""        # ptt, dcard, glassdoor, blind, reddit, etc.
    url: str = ""
    title: str = ""
    date: str = ""
    content_summary: str = ""
    questions_mentioned: list[str] = []
    difficulty: str = ""    # easy / medium / hard / unknown
    outcome: str = ""       # offer / rejected / unknown


class CompanyResearchSchema(BaseModel):
    company: str = ""
    industry: str = ""
    company_size: str = ""
    tech_stack: list[str] = []
    culture_notes: list[str] = []
    interview_process_steps: list[str] = []
    common_interview_questions: list[str] = []
    interview_experiences: list[InterviewExperienceEntry] = []
    sources: list[str] = []
    last_updated: str = ""


class PatchOperation(BaseModel):
    op: Literal["replace_section", "insert_after", "delete_section", "append_bullet"]
    target: str
    before: str = ""
    after: str = ""


class ResumePatch(BaseModel):
    operations: list[PatchOperation] = []
    rationale: list[str] = []
    confidence: float = 0.0


class InterviewQuestion(BaseModel):
    question: str = ""
    why_this_might_be_asked: str = ""
    signals: list[str] = []
    follow_ups: list[str] = []
    ideal_answer_points: list[str] = []


class QuestionBank(BaseModel):
    mode: Literal["behavioral", "technical", "whiteboard", "mixed"] = "behavioral"
    difficulty: Literal["easy", "medium", "hard"] = "medium"
    questions: list[InterviewQuestion] = []


# ─────────────────────────────────────────────────────────────────────────────
# Internal validation helpers  (not exposed as tools — called from tools)
# ─────────────────────────────────────────────────────────────────────────────

_ACTION_VERB_RE = re.compile(
    r"^(built|designed|led|drove|improved|reduced|increased|created|developed|"
    r"implemented|deployed|optimized|mentored|launched|scaled|managed|architected|"
    r"engineered|delivered|established|automated|migrated|refactored|owned|"
    r"coordinated|facilitated|produced|published|shipped|trained|analyzed|"
    r"researched|evaluated|streamlined)",
    re.IGNORECASE,
)

_COMMON_TECH_TERMS = [
    "python", "java", "scala", "spark", "kubernetes", "docker", "ml", "llm",
    "deep learning", "pytorch", "tensorflow", "sql", "nosql", "aws", "gcp",
    "azure", "react", "typescript", "go", "rust", "c++", "system design",
    "distributed systems", "microservices", "ci/cd", "agile", "leadership",
    "machine learning", "data pipeline", "api", "rest", "graphql", "kafka",
    "redis", "postgres", "mongodb", "elasticsearch", "airflow", "dbt",
]


def _validate_patch(patch_json: str, resume_md: str) -> tuple[bool, str]:
    """Validate a patch JSON against the current resume. Returns (ok, message)."""
    try:
        patch = ResumePatch.model_validate_json(patch_json)
    except Exception as exc:
        return False, f"Patch JSON parse error: {exc}"

    allowed_ops = {"replace_section", "insert_after", "delete_section", "append_bullet"}
    for op in patch.operations:
        if op.op not in allowed_ops:
            return False, f"Unknown operation type: '{op.op}'"
        if op.op == "delete_section" and len(op.target.split(".")) < 2:
            return False, (
                f"delete_section requires a specific sub-path target, not top-level: '{op.target}'"
            )
        if op.op in {"replace_section", "insert_after", "append_bullet"} and not op.after:
            return False, f"Operation '{op.op}' requires a non-empty 'after' field."
        if op.before and resume_md and op.before not in resume_md:
            return False, (
                f"Patch 'before' text not found in resume for target '{op.target}'. "
                "The resume may have changed — regenerate the patch."
            )

    return True, "Patch is valid."


def _apply_patch_to_md(patch_json: str, resume_md: str) -> str:
    """Apply validated patch operations to a resume markdown string. Returns updated markdown."""
    patch = ResumePatch.model_validate_json(patch_json)
    updated = resume_md
    for op in patch.operations:
        if op.op == "replace_section" and op.before:
            updated = updated.replace(op.before, op.after, 1)
        elif op.op == "append_bullet":
            updated = updated.rstrip() + f"\n- {op.after}\n"
        elif op.op == "insert_after" and op.before:
            idx = updated.find(op.before)
            if idx != -1:
                pos = idx + len(op.before)
                updated = updated[:pos] + f"\n{op.after}" + updated[pos:]
        elif op.op == "delete_section" and op.before:
            updated = updated.replace(op.before, "", 1)
    return updated


# ─────────────────────────────────────────────────────────────────────────────
# Job researcher tools
# ─────────────────────────────────────────────────────────────────────────────


@tool
def fetch_url_content(url: str, max_chars: int = 12000) -> str:
    """
    Fetch and return the main text content from a URL, stripping navigation
    and boilerplate. Returns up to max_chars characters.

    Args:
        url: The URL to fetch (e.g. a job posting page).
        max_chars: Maximum characters to return (default 12000).
    """
    headers = {"User-Agent": "Mozilla/5.0 (compatible; CareerCopilot/1.0)"}
    try:
        resp = requests.get(url, headers=headers, timeout=20)
        resp.raise_for_status()
    except Exception as exc:
        return f"ERROR fetching {url}: {exc}"

    soup = BeautifulSoup(resp.text, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header", "aside", "form"]):
        tag.decompose()
    text = soup.get_text(separator="\n", strip=True)
    return text[:max_chars]


@tool
def extract_jd_keywords(raw_jd_text: str, source_url: str = "") -> str:
    """
    Perform an initial keyword and structure extraction pass on raw job description
    text. Returns a JobDescriptionSchema JSON with detected keywords and source URL.
    The calling agent should enrich the output with role, company, and requirements.

    Args:
        raw_jd_text: Raw text of the job description page.
        source_url: Original URL of the posting (for provenance tracking).
    """
    text_lower = raw_jd_text.lower()
    keywords = sorted({term for term in _COMMON_TECH_TERMS if term in text_lower})

    jd = JobDescriptionSchema(source_url=source_url, keywords=keywords)
    return jd.model_dump_json(indent=2)


# ─────────────────────────────────────────────────────────────────────────────
# Company research & interview experience tools
# ─────────────────────────────────────────────────────────────────────────────


@tool
def web_search(query: str, max_results: int = 5) -> str:
    """
    Search the web using DuckDuckGo and return the top results with titles, URLs,
    and snippets. Use this to find company information, interview experiences,
    engineering blogs, Glassdoor/Blind reviews, and news articles.

    Args:
        query: Search query, e.g. 'Google L5 SWE interview experience 2024'
               or 'TSMC 面試心得 工程師'.
        max_results: Number of results to return (default 5, max 10).
    """
    url = "https://html.duckduckgo.com/html/"
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept-Language": "en-US,en;q=0.9,zh-TW;q=0.8,zh;q=0.7",
    }
    params = {"q": query, "kl": "wt-wt"}
    try:
        resp = requests.get(url, params=params, headers=headers, timeout=15)
        resp.raise_for_status()
    except Exception as exc:
        return f"ERROR: DuckDuckGo search failed: {exc}"

    soup = BeautifulSoup(resp.text, "html.parser")
    limit = max(1, min(max_results, 10))
    results = []
    for div in soup.select(".result")[:limit]:
        title_el = div.select_one(".result__title a")
        url_el = div.select_one(".result__url")
        snippet_el = div.select_one(".result__snippet")
        title = title_el.get_text(strip=True) if title_el else ""
        link = url_el.get_text(strip=True) if url_el else ""
        snippet = snippet_el.get_text(strip=True) if snippet_el else ""
        if title:
            results.append({"title": title, "url": link, "snippet": snippet})

    if not results:
        return f"No web search results found for: {query}"

    lines = [f"Web search results for '{query}':"]
    for i, r in enumerate(results, 1):
        lines.append(f"\n[{i}] {r['title']}")
        if r["url"]:
            lines.append(f"    URL: https://{r['url']}")
        if r["snippet"]:
            lines.append(f"    {r['snippet']}")
    lines.append("\nTip: Use fetch_url_content(url) to read the full content of any result above.")
    return "\n".join(lines)


@tool
def fetch_ptt_posts(query: str, board: str = "Salary", max_posts: int = 5) -> str:
    """
    Search PTT (ptt.cc) for posts matching a query on the given board.
    Common boards for job/interview research:
      - Salary   : salary and interview experience posts (面試心得)
      - Soft_Job : software industry job discussions
      - Tech_Job : tech company job discussions
      - job      : general job board

    Args:
        query: Search keyword, e.g. company name or '面試心得' (interview experience).
        board: PTT board name (default: Salary).
        max_posts: Number of post listings to retrieve (default 5).
    """
    session = requests.Session()
    session.cookies.set("over18", "1", domain="www.ptt.cc")
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; CareerCopilot/1.0)",
        "Accept": "text/html,application/xhtml+xml",
    }
    encoded_q = requests.utils.quote(query)
    search_url = f"https://www.ptt.cc/bbs/{board}/search?q={encoded_q}"
    try:
        resp = session.get(search_url, headers=headers, timeout=15)
        resp.raise_for_status()
    except Exception as exc:
        return f"ERROR fetching PTT/{board}: {exc}"

    soup = BeautifulSoup(resp.text, "html.parser")
    posts = []
    for div in soup.select("div.r-ent"):
        title_el = div.select_one("div.title a")
        date_el = div.select_one("div.date")
        author_el = div.select_one("div.author")
        nrec_el = div.select_one("div.nrec span")
        if title_el:
            title_text = title_el.get_text(strip=True)
            if title_text.startswith("(本文已被刪除)"):
                continue
            posts.append({
                "title": title_text,
                "url": "https://www.ptt.cc" + title_el.get("href", ""),
                "date": date_el.get_text(strip=True) if date_el else "",
                "author": author_el.get_text(strip=True) if author_el else "",
                "score": nrec_el.get_text(strip=True) if nrec_el else "0",
            })
        if len(posts) >= max_posts:
            break

    if not posts:
        return (
            f"No posts found on PTT/{board} for '{query}'. "
            "Try a different board (Soft_Job, Tech_Job, job) or broaden your query."
        )

    lines = [f"PTT/{board} search results for '{query}':"]
    for i, p in enumerate(posts, 1):
        lines.append(f"\n[{i}] {p['title']}")
        lines.append(f"    Date: {p['date']} | Author: {p['author']} | Score: {p['score']}")
        lines.append(f"    URL: {p['url']}")
    lines.append("\nTip: Use fetch_url_content(url) to read the full content of any post above.")
    return "\n".join(lines)


@tool
def fetch_dcard_posts(query: str, forum: str = "job", max_posts: int = 5) -> str:
    """
    Search Dcard (dcard.tw) for posts about interview experiences or company topics.
    Common forums:
      - job      : 求職/工作 (job seeking & workplace)
      - tech     : 科技 (technology industry)
      - career   : 職場 (career development)

    Args:
        query: Search keyword, e.g. company name or '面試' (interview).
        forum: Dcard forum slug (default: job).
        max_posts: Number of posts to retrieve (default 5).
    """
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept": "application/json",
        "Referer": "https://www.dcard.tw/",
    }
    api_url = "https://www.dcard.tw/service/api/v2/search/posts"
    params: dict = {"query": query, "limit": max_posts, "offset": 0}
    if forum:
        params["forum"] = forum

    try:
        resp = requests.get(api_url, params=params, headers=headers, timeout=15)
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        return f"ERROR fetching Dcard/{forum}: {exc}"

    if not data:
        return f"No posts found on Dcard/{forum} for query: '{query}'."

    lines = [f"Dcard/{forum} search results for '{query}':"]
    for i, post in enumerate(data[:max_posts], 1):
        title = post.get("title", "(no title)")
        excerpt = post.get("excerpt", "")
        post_id = post.get("id", "")
        like_count = post.get("likeCount", 0)
        comment_count = post.get("commentCount", 0)
        created_at = (post.get("createdAt", "") or "")[:10]
        post_url = f"https://www.dcard.tw/f/{forum}/p/{post_id}" if post_id else ""
        lines.append(f"\n[{i}] {title}")
        if excerpt:
            preview = excerpt[:220] + ("…" if len(excerpt) > 220 else "")
            lines.append(f"    {preview}")
        lines.append(
            f"    Date: {created_at} | Likes: {like_count} | Comments: {comment_count}"
        )
        if post_url:
            lines.append(f"    URL: {post_url}")
    lines.append("\nTip: Use fetch_url_content(url) to read the full content of any post above.")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Resume editor tools
# ─────────────────────────────────────────────────────────────────────────────


@tool
def validate_resume_patch(patch_json: str, resume_md: str) -> str:
    """
    Validate a resume patch JSON against the current resume markdown before applying it.
    Returns a VALID/INVALID status string with details.

    Args:
        patch_json: JSON string conforming to ResumePatch schema
                    (operations, rationale, confidence).
        resume_md: Current content of ./resume/master.md.
    """
    ok, message = _validate_patch(patch_json, resume_md)
    status = "VALID" if ok else "INVALID"
    return f"{status}: {message}"


@tool
def score_bullet_against_jd(bullet: str, jd_keywords_json: str) -> str:
    """
    Score a single resume bullet against JD keywords. Returns a JSON object with
    score (0.0–1.0), matched keywords, metric/verb presence, and a suggestion.

    Args:
        bullet: The resume bullet text to evaluate.
        jd_keywords_json: JSON array of keyword strings (e.g. ["python", "ml", "aws"]).
    """
    try:
        keywords: list[str] = json.loads(jd_keywords_json)
        if not isinstance(keywords, list):
            return "ERROR: jd_keywords_json must be a JSON array of strings."
    except Exception:
        return "ERROR: jd_keywords_json must be a JSON array of strings."

    bullet_lower = bullet.lower()
    matched = [kw for kw in keywords if kw.lower() in bullet_lower]
    has_metric = bool(re.search(r"\d+", bullet))
    has_action_verb = bool(_ACTION_VERB_RE.match(bullet.strip()))

    keyword_score = len(matched) / max(len(keywords), 1)
    metric_bonus = 0.15 if has_metric else 0.0
    verb_bonus = 0.10 if has_action_verb else 0.0
    total_score = min(keyword_score + metric_bonus + verb_bonus, 1.0)

    result = {
        "score": round(total_score, 2),
        "matched_keywords": matched,
        "has_quantified_metric": has_metric,
        "starts_with_action_verb": has_action_verb,
        "suggestion": (
            "Strong bullet."
            if total_score >= 0.7
            else (
                "Consider: "
                + (", ".join(keywords[:3]) if keywords else "")
                + (" | add a metric" if not has_metric else "")
                + (" | start with action verb" if not has_action_verb else "")
            ).strip(" |"),
        ),
    }
    return json.dumps(result, indent=2)


# ─────────────────────────────────────────────────────────────────────────────
# Coordinator tools
# ─────────────────────────────────────────────────────────────────────────────


@tool
def read_pdf_resume(pdf_path: str) -> str:
    """
    Extract text content from a PDF resume file. Returns the full text content
    that can be passed to ingest_resume_text for parsing.

    Args:
        pdf_path: Path to the PDF file (e.g. 'CV.pdf' or absolute path).
                  Relative paths are resolved from the current working directory.
    """
    try:
        # Handle relative paths from current working directory
        if not os.path.isabs(pdf_path):
            pdf_path = str(WORKSPACE_ROOT / pdf_path)
        
        if not os.path.exists(pdf_path):
            return f"ERROR: PDF file not found at {pdf_path}"
        
        reader = PdfReader(pdf_path)
        text_parts = []
        for page in reader.pages:
            # Try multiple extraction methods for better quality
            text = page.extract_text(extraction_mode="layout")
            if text:
                text_parts.append(text)
        
        full_text = "\n".join(text_parts)
        
        # Clean up common PDF extraction artifacts
        full_text = re.sub(r'\s+', ' ', full_text)  # Normalize whitespace
        full_text = re.sub(r' +', ' ', full_text)   # Remove multiple spaces
        full_text = full_text.replace(' \n', '\n')  # Clean line breaks
        full_text = re.sub(r'\n{3,}', '\n\n', full_text)  # Max 2 consecutive newlines

        if not full_text.strip():
            return f"ERROR: No text could be extracted from {pdf_path}"
        
        return full_text
    except Exception as exc:
        return f"ERROR reading PDF {pdf_path}: {exc}"


@tool
def ingest_resume_text(raw_text: str) -> str:
    """
    Parse raw resume text into a ResumeSchema JSON using LLM-based extraction.
    Extracts structured information including name, contact, education, experience,
    projects, skills, and awards from the raw text.

    Args:
        raw_text: Plain text content of the resume (pasted or extracted from PDF).
    """
    try:
        from langchain_openai import ChatOpenAI
        
        # Initialize LLM for parsing
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        
        # Create a detailed parsing prompt
        parsing_prompt = f"""You are a resume parsing expert. Extract structured information from the following resume text and return ONLY a valid JSON object conforming to this schema:

{{
  "basics": {{
    "name": "Full Name",
    "email": "email@example.com",
    "phone": "+1234567890",
    "location": "City, Country",
    "links": ["https://github.com/username", "https://linkedin.com/in/username"]
  }},
  "summary": "Brief professional summary or objective (if present)",
  "skills": ["Skill 1", "Skill 2", "Skill 3"],
  "experience": [
    {{
      "company": "Company Name",
      "title": "Job Title",
      "start_date": "Month Year",
      "end_date": "Month Year or Present",
      "bullets": ["Achievement 1", "Achievement 2"]
    }}
  ],
  "projects": [
    {{
      "name": "Project Name",
      "description": "Brief description",
      "bullets": ["Detail 1", "Detail 2"]
    }}
  ],
  "education": [
    {{
      "school": "University Name",
      "degree": "Degree Type in Field",
      "year": "Graduation Year or Year Range",
      "gpa": "GPA if mentioned",
      "coursework": ["Course 1", "Course 2"]
    }}
  ],
  "awards": [
    {{
      "title": "Award Name",
      "year": "Year",
      "description": "Description if available"
    }}
  ]
}}

IMPORTANT INSTRUCTIONS:
1. Extract ALL information accurately from the resume text
2. Preserve exact company names, job titles, dates, and achievements
3. For experience bullets, keep the original wording but clean up formatting
4. Extract ALL skills mentioned (programming languages, frameworks, tools, etc.)
5. Include ALL projects with their descriptions and details
6. Return ONLY valid JSON, no markdown formatting, no explanations
7. If a field is not present in the resume, use an empty string or empty array
8. For dates, preserve the original format from the resume

Resume Text:
{raw_text}

Return the JSON now:"""

        # Call LLM to parse
        response = llm.invoke(parsing_prompt)
        
        # Extract JSON from response
        response_text = response.content if hasattr(response, 'content') else str(response)
        
        # Try to extract JSON if it's wrapped in markdown code blocks
        json_match = re.search(r'```(?:json)?\s*(\{.*\})\s*```', response_text, re.DOTALL)
        if json_match:
            response_text = json_match.group(1)
        
        # Validate the parsed JSON against ResumeSchema
        parsed_resume = json.loads(response_text)
        validated = ResumeSchema.model_validate(parsed_resume)
        
        return validated.model_dump_json(indent=2)
        
    except Exception as exc:
        # Fallback to basic parsing if LLM fails
        print(f"Warning: LLM parsing failed ({exc}), using fallback basic parsing")
        base = ResumeSchema(summary=raw_text[:1000])
        return base.model_dump_json(indent=2)


@tool
def render_resume_to_markdown(resume_json: str) -> str:
    """
    Render a ResumeSchema JSON into a clean, well-structured Markdown document
    suitable for saving to ./resume/master.md.

    Args:
        resume_json: JSON string conforming to ResumeSchema.
    """
    try:
        resume = json.loads(resume_json)
    except Exception as exc:
        return f"ERROR: Could not parse resume JSON: {exc}"

    lines: list[str] = []
    basics = resume.get("basics", {})

    name = basics.get("name") or "Your Name"
    lines.append(f"# {name}")

    contact_parts = [
        v for v in [basics.get("email"), basics.get("phone"), basics.get("location")] if v
    ]
    if contact_parts:
        lines.append(" | ".join(contact_parts))
    if basics.get("links"):
        # Handle both string links and dict links (e.g., {"url": "...", "label": "..."})
        link_strs = []
        for link in basics["links"]:
            if isinstance(link, str):
                link_strs.append(link)
            elif isinstance(link, dict):
                link_strs.append(link.get("url", "") or link.get("label", "") or str(link))
            else:
                link_strs.append(str(link))
        if link_strs:
            lines.append(" | ".join(link_strs))
    lines.append("")

    if resume.get("summary"):
        lines += ["## Summary", resume["summary"], ""]

    if resume.get("skills"):
        # Handle both string skills and dict skills
        skill_strs = []
        for skill in resume["skills"]:
            if isinstance(skill, str):
                skill_strs.append(skill)
            elif isinstance(skill, dict):
                skill_strs.append(skill.get("name", "") or str(skill))
            else:
                skill_strs.append(str(skill))
        if skill_strs:
            lines += ["## Skills", ", ".join(skill_strs), ""]

    if resume.get("experience"):
        lines.append("## Experience")
        for exp in resume["experience"]:
            start = exp.get("start_date", "")
            end = exp.get("end_date", "Present")
            date_range = f"{start} – {end}" if start else end
            lines.append(f"### {exp.get('title', 'Role')} at {exp.get('company', 'Company')}")
            lines.append(f"*{date_range}*")
            for bullet in exp.get("bullets", []):
                lines.append(f"- {bullet}")
            lines.append("")

    if resume.get("projects"):
        lines.append("## Projects")
        for proj in resume.get("projects", []):
            if isinstance(proj, dict):
                lines.append(f"### {proj.get('name', 'Project')}")
                if proj.get("description"):
                    lines.append(proj["description"])
                for bullet in proj.get("bullets", []):
                    lines.append(f"- {bullet}")
                lines.append("")

    if resume.get("education"):
        lines.append("## Education")
        for edu in resume.get("education", []):
            if isinstance(edu, dict):
                degree = edu.get("degree", "")
                school = edu.get("school", "")
                year = edu.get("year", "")
                entry = f"**{degree}** — {school}" if degree else f"**{school}**"
                if year:
                    entry += f" ({year})"
                lines.append(entry)
        lines.append("")

    if resume.get("awards"):
        lines.append("## Awards & Achievements")
        for award in resume.get("awards", []):
            if isinstance(award, dict):
                lines.append(f"- {award.get('title', str(award))}")
        lines.append("")

    return "\n".join(lines)


@tool
def compute_gap_analysis(resume_json: str, jd_json: str) -> str:
    """
    Compute a keyword-and-coverage gap analysis between the parsed resume and the
    normalized job description. Returns a GapAnalysisSchema JSON string.

    Args:
        resume_json: JSON string conforming to ResumeSchema.
        jd_json: JSON string conforming to JobDescriptionSchema.
    """
    try:
        resume = json.loads(resume_json)
        jd = json.loads(jd_json)
    except Exception as exc:
        return f"ERROR: Could not parse inputs: {exc}"

    resume_text = json.dumps(resume).lower()
    jd_keywords = [kw.lower() for kw in jd.get("keywords", [])]

    matched = [kw for kw in jd_keywords if kw in resume_text]
    missing = [kw for kw in jd_keywords if kw not in resume_text]

    experience_entries = resume.get("experience", [])
    has_metrics = any(
        re.search(r"\d+", b)
        for exp in experience_entries
        for b in exp.get("bullets", [])
    )

    gaps: list[str] = []
    if missing:
        gaps.append(f"Missing JD keywords: {', '.join(missing[:12])}")
    if not has_metrics:
        gaps.append("Experience bullets lack quantified metrics.")
    if not resume.get("summary"):
        gaps.append("Resume summary/headline is absent.")

    sections_to_strengthen: list[str] = []
    if not has_metrics:
        sections_to_strengthen.append("experience")
    if not resume.get("summary"):
        sections_to_strengthen.append("summary")
    if not resume.get("skills"):
        sections_to_strengthen.append("skills")

    inferred_focus = jd.get("inferred_focus_areas", [])
    risk_areas = []
    for area in inferred_focus:
        if isinstance(area, dict):
            area_text = str(area.get("name", "") or area.get("area", "") or area)
        else:
            area_text = str(area)
        if area_text and area_text.lower() not in resume_text:
            risk_areas.append(area_text)

    gap = GapAnalysisSchema(
        matched_strengths=matched[:15],
        coverage_gaps=gaps,
        missing_keywords=missing[:20],
        resume_sections_to_strengthen=sections_to_strengthen,
        interview_risk_areas=risk_areas[:10],
    )
    return gap.model_dump_json(indent=2)


@tool
def apply_resume_patch(patch_json: str, resume_md: str) -> str:
    """
    Apply a validated patch JSON to the resume markdown and return the updated content.
    Runs validation internally and refuses to apply invalid patches.
    Always present the result to the user for review before writing to disk.

    Args:
        patch_json: JSON string of ResumePatch (operations, rationale, confidence).
        resume_md: Current content of ./resume/master.md.
    """
    ok, msg = _validate_patch(patch_json, resume_md)
    if not ok:
        return f"ERROR — Patch validation failed: {msg}\nNo changes applied."
    return _apply_patch_to_md(patch_json, resume_md)


@tool
def validate_question_bank(questions_json: str) -> str:
    """
    Validate a generated question bank JSON string against the QuestionBank schema.
    Returns a VALID/INVALID status with details (count, missing fields, etc.).

    Args:
        questions_json: JSON string conforming to QuestionBank schema.
    """
    try:
        bank = QuestionBank.model_validate_json(questions_json)
    except Exception as exc:
        return f"INVALID: Question bank parse error: {exc}"
    if not bank.questions:
        return "INVALID: Question bank is empty."
    missing_q = [i for i, q in enumerate(bank.questions) if not q.question]
    if missing_q:
        return f"INVALID: Questions at indices {missing_q} have empty 'question' fields."
    return f"VALID: {len(bank.questions)} questions in mode={bank.mode}, difficulty={bank.difficulty}."


@tool
def generate_resume_changelog_entry(patch_json: str, session_id: str) -> str:
    """
    Generate a Markdown changelog entry for a patch operation. The coordinator
    should append this to ./resume/changelog.md after applying a patch.

    Args:
        patch_json: JSON string of the applied patch (for op summary and rationale).
        session_id: Current session identifier (for traceability).
    """
    try:
        patch = json.loads(patch_json)
    except Exception:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M")
        return f"## {ts} | Session: {session_id}\n\nPatch applied.\n\n---\n\n"

    ops = patch.get("operations", [])
    rationale = patch.get("rationale", [])
    confidence = patch.get("confidence", 0.0)
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")

    lines = [
        f"## {ts} | Session: {session_id}",
        f"**{len(ops)} operation(s) applied** | Confidence: {confidence:.0%}",
        "",
    ]
    for op in ops:
        lines.append(f"- `{op.get('op')}` → `{op.get('target')}`")
    if rationale:
        lines += ["", "**Rationale:**"]
        for r in rationale:
            lines.append(f"- {r}")
    lines += ["", "---", ""]
    return "\n".join(lines)


@tool
def generate_session_brief(resume_json: str, jd_json: str, gap_analysis_json: str) -> str:
    """
    Generate a concise session brief Markdown document that summarises the candidate,
    target role, match strengths, gaps, and recommended next steps.
    Save the result to ./session/brief.md.

    Args:
        resume_json: JSON string conforming to ResumeSchema.
        jd_json: JSON string conforming to JobDescriptionSchema.
        gap_analysis_json: JSON string conforming to GapAnalysisSchema.
    """
    try:
        resume = json.loads(resume_json)
        jd = json.loads(jd_json)
        gap = json.loads(gap_analysis_json)
    except Exception as exc:
        return f"ERROR: {exc}"

    name = resume.get("basics", {}).get("name") or "Candidate"
    role = jd.get("job_title") or "Unknown Role"
    company = jd.get("company") or "Unknown Company"
    matched = gap.get("matched_strengths", [])
    gaps = gap.get("coverage_gaps", [])
    missing_kw = gap.get("missing_keywords", [])
    risk_areas = gap.get("interview_risk_areas", [])

    lines = [
        "# Session Brief",
        f"**Candidate:** {name}",
        f"**Target Role:** {role} @ {company}",
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        "## Matched Strengths",
    ]
    lines += [f"- {s}" for s in matched[:8]] or ["- None identified yet."]
    lines += ["", "## Coverage Gaps"]
    lines += [f"- {g}" for g in gaps] or ["- None identified."]
    lines += ["", "## Missing Keywords"]
    lines.append(", ".join(missing_kw[:15]) if missing_kw else "None")
    lines += ["", "## Interview Risk Areas"]
    lines += [f"- {r}" for r in risk_areas[:6]] or ["- None identified."]
    lines += [
        "",
        "## Recommended Next Steps",
        "1. Tailor resume bullets to address coverage gaps and embed missing keywords.",
        "2. Quantify achievements in experience entries where metrics are absent.",
        "3. Generate and review behavioral + technical interview questions.",
        "4. Practice answers for the identified risk areas.",
    ]
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Backend and persistence setup
# ─────────────────────────────────────────────────────────────────────────────

checkpointer = MemorySaver()
store = InMemoryStore()


def make_backend(runtime):
    """
    Composite backend:
      default paths → StateBackend  (ephemeral, thread-scoped)
      /memories/    → StoreBackend  (persistent across threads)
    """
    return CompositeBackend(
        default=StateBackend(runtime),
        routes={"/memories/": StoreBackend(runtime)},
    )


# ─────────────────────────────────────────────────────────────────────────────
# Subagent definitions
# ─────────────────────────────────────────────────────────────────────────────

job_researcher = {
    "name": "job-researcher",
    "description": (
        "Fetches and normalizes a job posting from a URL into a structured JobDescriptionSchema, "
        "then researches the company and gathers interview experiences from PTT, Dcard, and the web. "
        "Use whenever the user provides a job posting URL that needs to be ingested. "
        "Saves jd_normalized.json, jd_raw.md, source_url.txt, and company_research.json to ./job/ subdirectory."
    ),
    "system_prompt": (
        "You are a precise job posting analyst and company researcher. Your workflow:\n\n"
        "=== PART 1: JOB DESCRIPTION (REQUIRED) ===\n"
        "1. Extract the job posting URL from the incoming message (look for https:// or http://).\n"
        "2. Fetch the job posting with fetch_url_content(url).\n"
        "3. Call extract_jd_keywords to get an initial keyword set.\n"
        "4. Enrich the JSON with all available structured fields:\n"
        "   - job_title, company, location, employment_type, seniority\n"
        "   - responsibilities (list, extracted verbatim where possible)\n"
        "   - required_qualifications and preferred_qualifications (separate lists)\n"
        "   - keywords (comprehensive: tools, skills, languages, frameworks)\n"
        "   - inferred_focus_areas (what the role really prioritises — label clearly as inferred)\n"
        "5. Save the enriched JSON to ./job/jd_normalized.json (use get_workspace_path('job', 'jd_normalized.json')).\n"
        "6. Save the raw page text to ./job/jd_raw.md (use get_workspace_path('job', 'jd_raw.md')).\n"
        "7. Save the URL string to ./job/source_url.txt (use get_workspace_path('job', 'source_url.txt')).\n\n"
        "=== PART 2: COMPANY RESEARCH (REQUIRED - but keep it focused) ===\n"
        "8. Run a SINGLE focused search to gather interview intelligence:\n"
        "   - web_search('{company} {job_title} interview experience 2026')\n"
        "9. If the company is Taiwan-based (TSMC, MediaTek, etc.), also run:\n"
        "   - fetch_ptt_posts('{company}', board='Salary', max_posts=3)\n"
        "   - fetch_dcard_posts('{company}', forum='job', max_posts=3)\n"
        "10. From all search results (web + PTT + Dcard), identify the TOP 1 most relevant interview experience post.\n"
        "11. Fetch that single post with fetch_url_content(url) to extract specific questions.\n"
        "12. Synthesise findings into a CompanyResearchSchema JSON and save to ./job/company_research.json:\n"
        "   {\n"
        '     "company": "...",\n'
        '     "industry": "...",\n'
        '     "tech_stack": ["..."],\n'
        '     "interview_process_steps": ["..."],\n'
        '     "common_interview_questions": ["..."],\n'
        '     "interview_experiences": [{"source": "...", "url": "...", "questions_mentioned": ["..."]}],\n'
        '     "sources": ["..."],\n'
        '     "last_updated": "YYYY-MM-DD HH:MM"\n'
        "   }\n\n"
        "OPTIMIZATION RULES:\n"
        "- Do NOT run more than 3 searches total (1 web search + PTT + Dcard for Taiwan companies, or just 1 web search for others)\n"
        "- Fetch full content from ONLY 1 URL (the most relevant one across all sources)\n"
        "- Focus on extracting actual interview questions, not general company info\n"
        "- If no good interview experience is found, save a minimal research file and move on\n"
        "- Prioritize sources in this order: Dcard > PTT > Web (Dcard posts are usually more detailed)\n\n"
        "Return a concise summary: role title, company, seniority, top 5 requirements, "
        "inferred focus areas, AND a brief company research highlight (2-3 bullets max). "
        "Always label inferences distinctly from explicit requirements."
    ),
    "tools": [fetch_url_content, extract_jd_keywords, web_search, fetch_ptt_posts, fetch_dcard_posts, get_workspace_path, write_file, read_file],
}

resume_editor = {
    "name": "resume-editor",
    "description": (
        "Tailors and revises the candidate's canonical resume for the active target job. "
        "Use for: full tailoring passes, bullet rewrites, summary updates, targeted section edits, "
        "and adding or removing specific content. "
        "Produces structured patch operations saved to ./session/pending_patch.json."
    ),
    "system_prompt": (
        "You are a senior technical resume editor. Your workflow:\n"
        "1. Read ./resume/master.md (use get_workspace_path('resume', 'master.md')).\n"
        "2. Read ./job/jd_normalized.json (use get_workspace_path('job', 'jd_normalized.json')).\n"
        "3. Read ./job/gap_analysis.json if it exists (use get_workspace_path('job', 'gap_analysis.json')).\n"
        "4. Use score_bullet_against_jd to measure current bullet alignment before editing.\n"
        "5. Produce a structured patch JSON:\n"
        "   {\n"
        '     "operations": [{\n'
        '       "op": "replace_section|insert_after|delete_section|append_bullet",\n'
        '       "target": "e.g. experience.company_x.bullets[1]",\n'
        '       "before": "exact original text (required for replace/delete/insert_after)",\n'
        '       "after": "new text"\n'
        "     }],\n"
        '     "rationale": ["reason 1 per op"],\n'
        '     "confidence": 0.85\n'
        "   }\n"
        "6. Validate the patch with validate_resume_patch before saving.\n"
        "7. Save the validated patch to ./session/pending_patch.json (use get_workspace_path('session', 'pending_patch.json')).\n\n"
        "Editing rules:\n"
        "- Prefer precise, targeted edits — do NOT rewrite the full resume unless explicitly asked.\n"
        "- Every bullet must start with a strong action verb.\n"
        "- Add quantified metrics only where they exist or can be reasonably inferred.\n"
        "- Align language and keywords with the JD — but NEVER invent achievements.\n"
        "- Mark speculative additions as [DRAFT — PLEASE VERIFY].\n"
        "- Provide one concise rationale item per operation.\n\n"
        "Return to the coordinator: a plain-language summary of changes, "
        "the patch file path, and the alignment score delta (before vs after)."
    ),
    "tools": [score_bullet_against_jd, validate_resume_patch, get_workspace_path, write_file, read_file],
}

interview_designer = {
    "name": "interview-designer",
    "description": (
        "Generates structured behavioral, technical, and whiteboard interview questions "
        "grounded in both the active resume and the target job description. "
        "Also searches for real interview questions reported by other candidates via PTT, Dcard, "
        "Glassdoor, Blind, and the web. "
        "Use whenever the user asks for interview preparation, likely questions, "
        "coding challenges, or system design prompts."
    ),
    "system_prompt": (
        "You are an expert interview preparation coach. Your workflow:\n\n"
        "=== STEP 1: LOAD CONTEXT ===\n"
        "1. Read ./resume/master.md (use get_workspace_path('resume', 'master.md')).\n"
        "2. Read ./job/jd_normalized.json (use get_workspace_path('job', 'jd_normalized.json')).\n"
        "3. Read ./job/gap_analysis.json if present (use get_workspace_path('job', 'gap_analysis.json')).\n"
        "4. Read ./job/company_research.json if present (use get_workspace_path('job', 'company_research.json')) to get:\n"
        "   - Company-specific interview rounds and format\n"
        "   - Real questions previously reported by other candidates\n"
        "   - Company tech stack and culture signals\n\n"
        "=== STEP 2: GATHER ADDITIONAL INTERVIEW INTELLIGENCE (if company_research.json is absent or stale) ===\n"
        "5. Search for real interview experiences:\n"
        "   a. web_search('{company} {job_title} interview questions site:glassdoor.com')\n"
        "   b. web_search('{company} {job_title} interview questions site:blind.com OR site:reddit.com')\n"
        "   c. fetch_ptt_posts('{company} 面試', board='Salary') — also try Soft_Job or Tech_Job\n"
        "   d. fetch_dcard_posts('{company} 面試', forum='job')\n"
        "6. For the 2-3 most relevant results, fetch the full content with fetch_url_content.\n"
        "7. Extract any specific questions or topics others have been asked and incorporate them.\n\n"
        "=== STEP 3: GENERATE QUESTIONS ===\n"
        "8. Generate questions in the requested mode and difficulty level.\n"
        "9. **CRITICAL**: Format the output as a valid QuestionBank JSON conforming to this EXACT schema:\n"
        "   {\n"
        '     "mode": "behavioral" | "technical" | "whiteboard" | "mixed",\n'
        '     "difficulty": "easy" | "medium" | "hard",\n'
        '     "questions": [\n'
        "       {\n"
        '         "question": "the interview question text",\n'
        '         "why_this_might_be_asked": "role/resume/company-specific grounding",\n'
        '         "signals": ["competency 1", "competency 2"],\n'
        '         "follow_ups": ["follow-up question 1", "follow-up question 2"],\n'
        '         "ideal_answer_points": ["key point 1", "key point 2"]\n'
        "       }\n"
        "     ]\n"
        "   }\n"
        "10. Validate the output with validate_question_bank(questions_json).\n"
        "11. If validation fails, fix the JSON structure and re-validate until it passes.\n"
        "12. Save to ./session/question_bank.json (use get_workspace_path('session', 'question_bank.json')).\n\n"
        "IMPORTANT SCHEMA REQUIREMENTS:\n"
        "- The JSON MUST have exactly 3 top-level keys: 'mode', 'difficulty', 'questions'\n"
        "- 'mode' must be one of: behavioral, technical, whiteboard, mixed\n"
        "- 'difficulty' must be one of: easy, medium, hard\n"
        "- 'questions' must be a list of question objects\n"
        "- Each question object MUST have all 5 fields: question, why_this_might_be_asked, signals, follow_ups, ideal_answer_points\n"
        "- DO NOT add extra top-level fields like 'meta', 'top_priority', 'behavioral', etc.\n"
        "- DO NOT nest questions under categories — put ALL questions in the single 'questions' array\n\n"
        "Question generation rules:\n"
        "- Behavioral: use STAR-format probes; reference specific resume projects and roles.\n"
        "- Technical: ground in the JD's required tech stack and the candidate's actual work.\n"
        "- Whiteboard: design problems at the scale/domain relevant to the target company.\n"
        "- If real interview questions were found from PTT/Dcard/Glassdoor/Blind, include them "
        "  verbatim (marked '[Reported by candidates]' in the question text) alongside your generated questions.\n"
        "- Avoid generic filler — use the resume as evidence for deeper, personalised probes.\n"
        "- Flag interview risk areas from gap_analysis.json with targeted follow-ups.\n"
        "- Generate 8-12 questions total (mix of behavioral, technical, and domain-specific).\n\n"
        "Return to the coordinator: a formatted, readable preview of the top questions, "
        "clearly separating AI-generated questions from real questions reported by past candidates."
    ),
    "tools": [validate_question_bank, web_search, fetch_ptt_posts, fetch_dcard_posts, fetch_url_content, get_workspace_path, write_file, read_file],
}


# ─────────────────────────────────────────────────────────────────────────────
# Coordinator system prompt
# ─────────────────────────────────────────────────────────────────────────────

COORDINATOR_SYSTEM_PROMPT = """\
You are Career Copilot, an expert career assistant that helps candidates tailor \
their resumes and prepare for job interviews.

CANONICAL STATE (all paths relative to current working directory):
- ONE canonical resume   → ./resume/master.md
- ONE active job context → ./job/jd_normalized.json
- Company research       → ./job/company_research.json
- Session brief          → ./session/brief.md
- Pending patch          → ./session/pending_patch.json
- User preferences       → /memories/writing_preferences.md

IMPORTANT - FILE PATH USAGE:
- ALWAYS use get_workspace_path(subdir, filename) when saving files to ensure directories exist.
- Example: get_workspace_path('resume', 'master.md') returns './resume/master.md' and creates ./resume/ if needed.
- The helper automatically creates subdirectories, so you never need to check if they exist.

WORKFLOWS:

A. INGESTION (user provides job URL):
   *** CRITICAL: If user message contains http:// or https://, this is workflow A. Follow these steps in order: ***
   
   1. FIRST, call read_pdf_resume('CV.pdf') to extract resume text from the PDF file.
   
   2. THEN, delegate to job-researcher with a message that includes the FULL URL from the user's message.
      - Extract the complete URL (including https://) from the user's message.
      - Delegate with: "Please fetch and normalize this job posting: [FULL_URL_HERE]"
      - DO NOT try to fetch the URL yourself using fetch_url_content.
      - DO NOT claim URLs are inaccessible or blocked.
      - DO NOT ask the user to paste the job description manually.
      - ALWAYS let job-researcher attempt the fetch first - it will report actual errors if any.
   
   3. After job-researcher completes, call ingest_resume_text with the extracted resume text.
   
   4. Call render_resume_to_markdown and save to ./resume/master.md (use get_workspace_path('resume', 'master.md')).
   
   5. Save parsed resume JSON to ./resume/parsed.json (use get_workspace_path('resume', 'parsed.json')).
   
   6. Call compute_gap_analysis → save to ./job/gap_analysis.json (use get_workspace_path('job', 'gap_analysis.json')).
   
   7. Call generate_session_brief → save to ./session/brief.md (use get_workspace_path('session', 'brief.md')).
   
   8. Report to user: role match, top 3 gaps, company research highlights, and next steps.

B. RESUME TAILORING (user asks to tailor or improve):
   1. Delegate to resume-editor with current resume + JD.
   2. Resume-editor produces ./session/pending_patch.json.
   3. Read the patch; present changes as a clear before/after diff with rationale.
   4. After user approves: call apply_resume_patch; save updated markdown to master.md.
   5. Call generate_resume_changelog_entry; append to ./resume/changelog.md (use get_workspace_path('resume', 'changelog.md')).

C. TARGETED EDITS ("add X", "remove Y", "rewrite Z"):
   1. Identify the affected section and intent.
   2. Delegate to resume-editor with specific scoped instructions.
   3. Apply approved patch and update changelog.

D. INTERVIEW PREP (user asks for questions):
   1. Identify: mode (behavioral/technical/whiteboard/mixed), difficulty (easy/medium/hard).
   2. Delegate to interview-designer.
   3. Present the structured question bank in a clean, readable format.

E. PREFERENCE LEARNING (user states a stable preference):
   1. Append to /memories/writing_preferences.md.
   2. Acknowledge and apply in all subsequent operations.

F. COMPANY RESEARCH (user explicitly asks for MORE company research):
   1. Check if ./job/company_research.json already exists (use get_workspace_path('job', 'company_research.json')).
   2. If user wants deeper research, run additional searches:
      a. web_search('{company} engineering culture tech stack')
      b. fetch_ptt_posts('{company}', board='Soft_Job') for Taiwan companies
      c. fetch_dcard_posts('{company}', forum='tech') for Taiwan companies
   3. Fetch top 1-2 additional results with fetch_url_content(url).
   4. Merge findings into existing company_research.json.
   5. Present new findings.

G. INTERVIEW EXPERIENCE SEARCH (user asks what others experienced at this company):
   1. Use web_search('{company} {role} interview experience'), fetch_ptt_posts, fetch_dcard_posts.
   2. Retrieve full posts with fetch_url_content for top 1-2 results only.
   3. Summarise: interview rounds reported, specific questions asked, difficulty, and outcomes.
   4. Save findings to ./job/company_research.json (use get_workspace_path('job', 'company_research.json')).

PRINCIPLES:
- ALWAYS delegate resume editing to resume-editor.
- ALWAYS delegate interview question generation to interview-designer.
- ALWAYS delegate job URL ingestion AND basic company research to job-researcher.
  * CRITICAL: When you see ANY URL in the user's message (http/https), IMMEDIATELY delegate to job-researcher.
  * DO NOT try to fetch URLs yourself. DO NOT claim URLs are inaccessible or blocked.
  * Let job-researcher attempt the fetch - it has the same tools and will report if there's an actual error.
  * Job-researcher will do focused company research automatically (1-2 searches, 1 URL fetch).
- When the user provides a job URL, read the resume from CV.pdf BEFORE delegating to job-researcher.
- ALWAYS use get_workspace_path(subdir, filename) when reading or writing workspace files.
- Prefer patch-based edits over full rewrites.
- NEVER invent resume content not present in the user's original text.
- Always explain changes in plain language before writing files.
- Ask for confirmation before applying patches or overwriting files.

FORMATTING:
- Show patches as clear before/after diffs, not raw JSON.
- Show question banks as numbered lists with rationale.
- Keep coordinator responses concise and action-oriented.
"""

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


# ─────────────────────────────────────────────────────────────────────────────
# Interactive CLI with human-in-the-loop interrupt handling
# ─────────────────────────────────────────────────────────────────────────────


def _extract_last_message(result) -> str:
    """Pull the last assistant message text out of a result object."""
    messages = result.get("messages", []) if isinstance(result, dict) else []
    if not messages:
        return ""
    last = messages[-1]
    if hasattr(last, "content"):
        content = last.content
        if isinstance(content, list):
            parts = [
                block.get("text", "")
                for block in content
                if isinstance(block, dict) and block.get("type") == "text"
            ]
            return "\n".join(parts)
        return str(content)
    return str(last)


def _handle_interrupt(interrupt_value: dict) -> list[dict]:
    """
    Present pending tool-call interrupts to the user and collect approve/edit/reject
    decisions. Returns a list of decision dicts for Command(resume={"decisions": [...]}).
    """
    action_requests: list[dict] = interrupt_value.get("action_requests", [])
    review_configs: list[dict] = interrupt_value.get("review_configs", [])

    # Build a lookup from tool name → allowed decisions
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

        print(f"\n[{i}/{len(action_requests)}] Tool: {tool_name}")
        print("  Arguments:")
        for k, v in tool_args.items():
            val_str = str(v)
            if len(val_str) > 200:
                val_str = val_str[:200] + "…"
            print(f"    {k}: {val_str}")
        print(f"  Allowed decisions: {', '.join(allowed)}")

        while True:
            choice = input(f"\n  Decision [{'/'.join(allowed)}]: ").strip().lower()
            if choice in allowed:
                break
            print(f"  Please enter one of: {', '.join(allowed)}")

        if choice == "approve":
            decisions.append({"type": "approve"})
        elif choice == "reject":
            decisions.append({"type": "reject"})
        elif choice == "edit":
            print("  Enter edited arguments as key=value pairs (one per line).")
            print("  Press Enter on an empty line when done.")
            edited_args = dict(tool_args)  # start from originals
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


def run_interactive() -> None:
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


if __name__ == "__main__":
    run_interactive()
