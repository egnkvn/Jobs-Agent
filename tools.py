"""
Tool functions for Career Copilot.

This module contains all the tool functions used by the coordinator and subagents:
- Workspace path management
- File I/O operations
- Job research tools (URL fetching, keyword extraction)
- Company research tools (web search, PTT, Dcard)
- Resume processing tools (PDF reading, parsing, rendering)
- Resume editing tools (patch validation, bullet scoring, gap analysis)
- Interview preparation tools (question bank validation)
- Session utilities (changelog, session brief)
"""

import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Literal

import requests
from bs4 import BeautifulSoup
from langchain.tools import tool
from pydantic import BaseModel, Field
from pypdf import PdfReader


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


# ─────────────────────────────────────────────────────────────────────────────
# File I/O tools
# ─────────────────────────────────────────────────────────────────────────────

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
        if not os.path.isabs(file_path):
            file_path = str(WORKSPACE_ROOT / file_path)
        
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        
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
        if not os.path.isabs(file_path):
            file_path = str(WORKSPACE_ROOT / file_path)
        
        if not os.path.exists(file_path):
            return f"ERROR: File not found at {file_path}"
        
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        if old_content not in content:
            return f"ERROR: old_content not found in {file_path}. File may have changed."
        
        updated_content = content.replace(old_content, new_content, 1)
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(updated_content)
        
        return f"SUCCESS: Edited {file_path} (replaced {len(old_content)} chars with {len(new_content)} chars)"
    except Exception as exc:
        return f"ERROR editing {file_path}: {exc}"


# ─────────────────────────────────────────────────────────────────────────────
# Data schemas
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
    source: str = ""
    url: str = ""
    title: str = ""
    date: str = ""
    content_summary: str = ""
    questions_mentioned: list[str] = []
    difficulty: str = ""
    outcome: str = ""


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
# Internal validation helpers
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
# Resume processing tools
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
        if not os.path.isabs(pdf_path):
            pdf_path = str(WORKSPACE_ROOT / pdf_path)
        
        if not os.path.exists(pdf_path):
            return f"ERROR: PDF file not found at {pdf_path}"
        
        reader = PdfReader(pdf_path)
        text_parts = []
        for page in reader.pages:
            text = page.extract_text(extraction_mode="layout")
            if text:
                text_parts.append(text)
        
        full_text = "\n".join(text_parts)
        
        full_text = re.sub(r'\s+', ' ', full_text)
        full_text = re.sub(r' +', ' ', full_text)
        full_text = full_text.replace(' \n', '\n')
        full_text = re.sub(r'\n{3,}', '\n\n', full_text)

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
        
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        
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

        response = llm.invoke(parsing_prompt)
        
        response_text = response.content if hasattr(response, 'content') else str(response)
        
        json_match = re.search(r'```(?:json)?\s*(\{.*\})\s*```', response_text, re.DOTALL)
        if json_match:
            response_text = json_match.group(1)
        
        parsed_resume = json.loads(response_text)
        validated = ResumeSchema.model_validate(parsed_resume)
        
        return validated.model_dump_json(indent=2)
        
    except Exception as exc:
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


# ─────────────────────────────────────────────────────────────────────────────
# Resume editing tools
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


# ─────────────────────────────────────────────────────────────────────────────
# Interview preparation tools
# ─────────────────────────────────────────────────────────────────────────────

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


# ─────────────────────────────────────────────────────────────────────────────
# Session utilities
# ─────────────────────────────────────────────────────────────────────────────

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
