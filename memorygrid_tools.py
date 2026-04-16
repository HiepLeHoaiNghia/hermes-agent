"""
MemoryGrid Integration Tools for Hermes MCP Server.

This module contains all MemoryGrid-specific tools that extend the Hermes MCP
server with task queries, context loading, daily notes, and a learning bridge
(lessons, patterns, mistakes).

Separated from mcp_serve.py to minimise conflict surface when syncing with
upstream hermes-agent. Upstream does not (and should not) contain this file.

Usage in mcp_serve.py:
    from memorygrid_tools import register_memorygrid_tools
    register_memorygrid_tools(mcp)   # adds 6 MemoryGrid tools
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def _get_hermes_home() -> Path:
    """Resolve HERMES_HOME from constants or environment."""
    try:
        from hermes_constants import get_hermes_home
        return get_hermes_home()
    except ImportError:
        return Path(os.environ.get("HERMES_HOME", Path.home() / ".hermes"))


def _get_memorygrid_root() -> Path:
    """Resolve MemoryGrid root from HERMES_HOME or environment.

    HERMES_HOME is typically ~/MemoryGrid/extension-data/hermes/,
    so MemoryGrid root is HERMES_HOME.parent.parent.
    """
    mg_root = os.environ.get("MEMORYGRID_HOME")
    if mg_root:
        return Path(mg_root)

    hermes_home = _get_hermes_home()

    if hermes_home.name == "hermes" and hermes_home.parent.name == "extension-data":
        return hermes_home.parent.parent

    return Path.home() / "MemoryGrid"


def _get_lessons_file() -> Path:
    """Return path to the MemoryGrid lessons store in Hermes memories."""
    return _get_hermes_home() / "memories" / "memorygrid-lessons.md"


# ---------------------------------------------------------------------------
# Holographic memory bridge
# ---------------------------------------------------------------------------

_holo_store = None


def _get_holographic_store():
    """Lazily initialise and return the holographic MemoryStore.

    Returns None if the holographic plugin is not available (e.g., missing
    numpy dependency). All callers should handle None gracefully.
    """
    global _holo_store
    if _holo_store is not None:
        return _holo_store

    try:
        from plugins.memory.holographic.store import MemoryStore
        db_path = _get_hermes_home() / "memory_store.db"
        _holo_store = MemoryStore(
            db_path=str(db_path),
            default_trust=0.5,
            hrr_dim=1024,
        )
        logger.info("Holographic memory store initialised at %s", db_path)
        return _holo_store
    except Exception as exc:
        logger.warning("Holographic memory unavailable: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def _read_frontmatter(content: str) -> tuple:
    """Parse YAML frontmatter from markdown content.

    Returns (frontmatter_dict, body_text).
    """
    if not content.startswith("---"):
        return {}, content

    parts = content.split("---", 2)
    if len(parts) < 3:
        return {}, content

    fm_text = parts[1].strip()
    body = parts[2].strip()

    fm = {}
    for line in fm_text.split("\n"):
        line = line.strip()
        if ":" in line and not line.startswith("-"):
            key, _, value = line.partition(":")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if value.startswith("[") and value.endswith("]"):
                value = [
                    v.strip().strip('"').strip("'")
                    for v in value[1:-1].split(",") if v.strip()
                ]
            fm[key] = value

    return fm, body


def _find_task_files(mg_root: Path, project: Optional[str] = None) -> List[dict]:
    """Scan MemoryGrid task files and return parsed metadata.

    Each task file is at: 08-projects/<project>/epics/<epic>/tasks/<task>.md
    """
    tasks = []
    projects_dir = mg_root / "08-projects"

    if not projects_dir.exists():
        return tasks

    project_dirs = []
    if project:
        p_dir = projects_dir / project
        if p_dir.exists():
            project_dirs.append(p_dir)
    else:
        for p in projects_dir.iterdir():
            if p.is_dir() and not p.name.startswith("00-"):
                project_dirs.append(p)

    for p_dir in project_dirs:
        epics_dir = p_dir / "epics"
        if not epics_dir.exists():
            continue
        for epic_dir in epics_dir.iterdir():
            if not epic_dir.is_dir():
                continue
            tasks_dir = epic_dir / "tasks"
            if not tasks_dir.exists():
                continue
            for task_file in tasks_dir.glob("*.md"):
                try:
                    content = task_file.read_text(encoding="utf-8")
                    fm, _ = _read_frontmatter(content)
                    tasks.append({
                        "id": fm.get("id", task_file.stem),
                        "status": fm.get("status", "unknown"),
                        "progress": fm.get("progress", 0),
                        "project": p_dir.name,
                        "epic": epic_dir.name,
                        "description": fm.get("description", ""),
                        "file": str(task_file.relative_to(mg_root)),
                    })
                except Exception:
                    continue

    return tasks


def _parse_lessons(content: str) -> List[dict]:
    """Parse structured lesson entries from the lessons file.

    Format:
        ### LESSON-N | type | project | date
        **Tags:** tag1, tag2
        **Content:** lesson text
        **Source:** session_id or task_id
    """
    lessons = []
    current = None
    for line in content.split("\n"):
        if line.startswith("### LESSON-"):
            if current:
                lessons.append(current)
            parts = line.strip("# ").split(" | ")
            lesson_id = parts[0].strip() if len(parts) > 0 else ""
            lesson_type = parts[1].strip().lower() if len(parts) > 1 else ""
            project = parts[2].strip().lower() if len(parts) > 2 else ""
            date = parts[3].strip() if len(parts) > 3 else ""
            current = {
                "id": lesson_id,
                "type": lesson_type,
                "project": project,
                "date": date,
                "tags": [],
                "content": "",
                "source": "",
            }
        elif current and line.startswith("**Tags:**"):
            tags_str = line.replace("**Tags:**", "").strip()
            current["tags"] = [
                t.strip().lower()
                for t in tags_str.split(",")
                if t.strip()
            ]
        elif current and line.startswith("**Content:**"):
            current["content"] = line.replace("**Content:**", "").strip()
        elif current and line.startswith("**Source:**"):
            current["source"] = line.replace("**Source:**", "").strip()
        elif current and line.strip() and not line.startswith("#"):
            # Continuation of content
            current["content"] += " " + line.strip()

    if current:
        lessons.append(current)
    return lessons


# ---------------------------------------------------------------------------
# Tool registration
# ---------------------------------------------------------------------------

def register_memorygrid_tools(mcp) -> None:
    """Register all MemoryGrid tools on the given FastMCP server.

    Call this once from create_mcp_server() in mcp_serve.py:

        from memorygrid_tools import register_memorygrid_tools
        register_memorygrid_tools(mcp)
    """

    # -- Read-only tools ------------------------------------------------------

    @mcp.tool()
    def memorygrid_get_context() -> str:
        """Get MemoryGrid bootstrap context for agent sessions.

        Returns working style hard rules, current focus handoff, and active
        project resolution. Replaces reading 3 separate files with one MCP call.
        """
        mg_root = _get_memorygrid_root()
        result = {"memorygrid_root": str(mg_root), "files_loaded": []}

        # 1. Working Style — HARD RULES only
        ws_file = mg_root / "02-working-style.md"
        if ws_file.exists():
            try:
                content = ws_file.read_text(encoding="utf-8")
                hard_rules = ""
                in_hard_rules = False
                for line in content.split("\n"):
                    if "HARD RULES" in line:
                        in_hard_rules = True
                    elif in_hard_rules and line.startswith("## ") and "HARD RULES" not in line:
                        break
                    if in_hard_rules:
                        hard_rules += line + "\n"
                result["working_style_hard_rules"] = hard_rules.strip()
                result["files_loaded"].append("02-working-style.md")
            except Exception as e:
                result["working_style_error"] = str(e)

        # 2. Current Focus — Handoff + Global Top 3
        cf_file = mg_root / "03-current-focus.md"
        if cf_file.exists():
            try:
                content = cf_file.read_text(encoding="utf-8")
                fm, body = _read_frontmatter(content)

                handoff = ""
                global_top = ""
                section = None
                for line in body.split("\n"):
                    if "## Handoff" in line:
                        section = "handoff"
                    elif "## Global Top 3" in line:
                        section = "global_top"
                    elif line.startswith("## ") and section in ("handoff", "global_top"):
                        section = None

                    if section == "handoff":
                        handoff += line + "\n"
                    elif section == "global_top":
                        global_top += line + "\n"

                result["current_focus"] = {
                    "active_project_id": fm.get("active_project_id", ""),
                    "handoff": handoff.strip(),
                    "global_top_3": global_top.strip(),
                }
                result["files_loaded"].append("03-current-focus.md")
            except Exception as e:
                result["current_focus_error"] = str(e)

        return json.dumps(result, indent=2)

    @mcp.tool()
    def memorygrid_query_tasks(
        project: Optional[str] = None,
        status: Optional[str] = None,
        query: Optional[str] = None,
    ) -> str:
        """Query MemoryGrid task status across projects and epics.

        Returns task metadata (id, status, progress, project, epic, description)
        matching the given filters. All filters are optional.

        Args:
            project: Filter by project ID (e.g., "memorygrid", "seopul")
            status: Filter by status (open, in progress, in review, resolved, done, cancelled)
            query: Text search across task ID and description
        """
        mg_root = _get_memorygrid_root()
        tasks = _find_task_files(mg_root, project=project)

        if status:
            status_lower = status.lower()
            tasks = [t for t in tasks if t["status"].lower() == status_lower]

        if query:
            query_lower = query.lower()
            tasks = [
                t for t in tasks
                if query_lower in t["id"].lower()
                or query_lower in t.get("description", "").lower()
            ]

        return json.dumps({"count": len(tasks), "tasks": tasks}, indent=2)

    @mcp.tool()
    def memorygrid_get_daily_notes(
        date: Optional[str] = None,
        days: int = 1,
    ) -> str:
        """Get MemoryGrid daily note(s) with session summaries.

        Returns daily note content for the specified date, or summaries for
        the most recent N days if days > 1.

        Args:
            date: Date in YYYY-MM-DD format (default: today)
            days: Number of recent days to return (default 1, max 7)
        """
        from datetime import datetime, timedelta

        mg_root = _get_memorygrid_root()
        daily_dir = mg_root / "06-daily"

        if not daily_dir.exists():
            return json.dumps({"error": "Daily notes directory not found"})

        days = min(max(1, days), 7)

        if date:
            try:
                start_date = datetime.strptime(date, "%Y-%m-%d")
            except ValueError:
                return json.dumps(
                    {"error": f"Invalid date format: {date}. Use YYYY-MM-DD"}
                )
        else:
            start_date = datetime.now()

        notes = []
        for i in range(days):
            check_date = start_date - timedelta(days=i)
            date_str = check_date.strftime("%Y-%m-%d")
            note_file = daily_dir / f"{date_str}.md"

            if note_file.exists():
                try:
                    content = note_file.read_text(encoding="utf-8")
                    fm, body = _read_frontmatter(content)

                    session_count = body.count("## Session —")
                    sessions = []
                    for line in body.split("\n"):
                        if line.startswith("## Session —"):
                            sessions.append(line.strip("# ").strip())

                    notes.append({
                        "date": date_str,
                        "day_of_week": fm.get("day_of_week", ""),
                        "session_count": session_count,
                        "sessions": sessions,
                        "content_length": len(body),
                        "content": body if days == 1 else None,
                    })
                except Exception as e:
                    notes.append({"date": date_str, "error": str(e)})
            else:
                notes.append({"date": date_str, "exists": False})

        return json.dumps({"count": len(notes), "notes": notes}, indent=2)

    # -- Learning Bridge tools -------------------------------------------------

    @mcp.tool()
    def memorygrid_save_lesson(
        type: str,
        project: str,
        content: str,
        tags: Optional[str] = None,
        source: Optional[str] = None,
    ) -> str:
        """Save a lesson, pattern, or mistake to Hermes persistent memory.

        Agents call this when they learn something worth remembering: mistakes to
        avoid, domain patterns, project-specific conventions, or user preferences.

        Dual-write: saves to both markdown file (portable backup) and holographic
        SQLite store (semantic search, trust scoring, entity resolution).

        Args:
            type: Lesson type — "mistake", "pattern", "domain", "preference", "correction"
            project: Project ID (e.g., "memorygrid", "seopul", "snw")
            content: The lesson text — be specific and actionable
            tags: Comma-separated tags for filtering (e.g., "python,api,celery")
            source: Optional source reference (task ID, session ID, or context)
        """
        from datetime import datetime

        lessons_file = _get_lessons_file()

        # Ensure directory exists
        lessons_file.parent.mkdir(parents=True, exist_ok=True)

        # Read existing lessons to determine next ID
        existing_count = 0
        if lessons_file.exists():
            existing_content = lessons_file.read_text(encoding="utf-8")
            existing_count = existing_content.count("### LESSON-")

        lesson_id = f"LESSON-{existing_count + 1:03d}"
        date_str = datetime.now().strftime("%Y-%m-%d")
        tags_str = tags if tags else ""
        source_str = source if source else ""

        entry = (
            f"\n### {lesson_id} | {type} | {project} | {date_str}\n"
            f"**Tags:** {tags_str}\n"
            f"**Content:** {content}\n"
            f"**Source:** {source_str}\n"
        )

        # Create file with header if new
        if not lessons_file.exists():
            header = (
                "# MemoryGrid Lessons\n\n"
                "Persistent knowledge store for MemoryGrid agents.\n"
                "Lessons are saved by agents and queried at session start.\n"
            )
            lessons_file.write_text(header + entry, encoding="utf-8")
        else:
            with open(lessons_file, "a", encoding="utf-8") as f:
                f.write(entry)

        # --- Dual-write: holographic SQLite store ---
        holo_tags = ",".join(
            t.strip()
            for t in (tags_str or "").split(",")
            if t.strip()
        )
        holo = _get_holographic_store()
        holo_ok = False
        if holo:
            try:
                holo.add_fact(
                    content=f"[{lesson_id}] {content}",
                    category=type,
                    tags=holo_tags,
                )
                holo_ok = True
            except Exception as exc:
                logger.warning("Holographic write failed: %s", exc)

        return json.dumps({
            "status": "saved",
            "lesson_id": lesson_id,
            "type": type,
            "project": project,
            "date": date_str,
            "holographic": holo_ok,
        }, indent=2)

    @mcp.tool()
    def memorygrid_query_lessons(
        project: Optional[str] = None,
        type: Optional[str] = None,
        query: Optional[str] = None,
        tags: Optional[str] = None,
        limit: int = 10,
    ) -> str:
        """Query accumulated lessons from Hermes persistent memory.

        Returns lessons matching the given filters. All filters are optional.
        When holographic memory is available, uses FTS5 full-text search with
        trust-weighted ranking. Falls back to markdown file parsing otherwise.

        Args:
            project: Filter by project ID (e.g., "memorygrid", "seopul")
            type: Filter by type (mistake, pattern, domain, preference, correction)
            query: Text search across lesson content
            tags: Comma-separated tags to match (any tag matches)
            limit: Maximum number of results (default 10, max 50)
        """
        limit = min(max(1, limit), 50)

        # --- Try holographic store first ---
        holo = _get_holographic_store()
        if holo and (query or tags or type):
            try:
                search_q = query or ""
                if tags:
                    search_q = f"{search_q} {tags}".strip()

                category = type if type else None
                results = holo.search_facts(
                    query=search_q,
                    category=category,
                    min_trust=0.0,
                    limit=limit,
                )
                lessons = [
                    {
                        "id": f"FACT-{r['fact_id']}",
                        "type": r.get("category", "general"),
                        "content": r["content"],
                        "tags": r.get("tags", "").split(",") if r.get("tags") else [],
                        "trust": round(r.get("trust_score", 0.5), 2),
                    }
                    for r in results
                ]
                return json.dumps({
                    "count": len(lessons),
                    "lessons": lessons,
                    "source": "holographic",
                }, indent=2)
            except Exception as exc:
                logger.warning("Holographic query failed, falling back: %s", exc)

        # --- Fallback: markdown file parsing ---
        lessons_file = _get_lessons_file()

        if not lessons_file.exists():
            return json.dumps({"count": 0, "lessons": [], "note": "No lessons file found"})

        try:
            content = lessons_file.read_text(encoding="utf-8")
        except Exception as e:
            return json.dumps({"error": f"Failed to read lessons: {e}"})

        lessons = _parse_lessons(content)

        # Apply filters
        if project:
            project_lower = project.lower()
            lessons = [l for l in lessons if l["project"] == project_lower]

        if type:
            type_lower = type.lower()
            lessons = [l for l in lessons if l["type"] == type_lower]

        if query:
            query_lower = query.lower()
            lessons = [
                l for l in lessons
                if query_lower in l["content"].lower()
                or query_lower in l["id"].lower()
            ]

        if tags:
            filter_tags = {t.strip().lower() for t in tags.split(",") if t.strip()}
            lessons = [
                l for l in lessons
                if filter_tags & set(l["tags"])
            ]

        lessons = lessons[:limit]

        return json.dumps({"count": len(lessons), "lessons": lessons}, indent=2)

    @mcp.tool()
    def memorygrid_record_session(
        task_id: str,
        outcome: str,
        mistakes: Optional[str] = None,
        patterns: Optional[str] = None,
        tokens_spent: Optional[str] = None,
    ) -> str:
        """Record a session outcome and optionally save lessons.

        Call at session end to persist session metadata. If mistakes or patterns
        are provided, they are automatically saved as lessons.

        Args:
            task_id: The task identifier (e.g., "hermes-learning-bridge")
            outcome: Session outcome — "success", "partial", "blocked", "failed"
            mistakes: Semicolon-separated mistake descriptions to save
            patterns: Semicolon-separated pattern descriptions to save
            tokens_spent: Optional token count or cost string
        """
        from datetime import datetime

        results = {"task_id": task_id, "outcome": outcome, "lessons_saved": []}

        # Auto-save mistakes as lessons
        if mistakes:
            for i, mistake in enumerate(mistakes.split(";")):
                mistake = mistake.strip()
                if not mistake:
                    continue
                memorygrid_save_lesson(
                    type="mistake",
                    project="auto",
                    content=mistake,
                    tags="auto-recorded,mistake",
                    source=task_id,
                )
                results["lessons_saved"].append("mistake")

        # Auto-save patterns as lessons
        if patterns:
            for i, pattern in enumerate(patterns.split(";")):
                pattern = pattern.strip()
                if not pattern:
                    continue
                memorygrid_save_lesson(
                    type="pattern",
                    project="auto",
                    content=pattern,
                    tags="auto-recorded,pattern",
                    source=task_id,
                )
                results["lessons_saved"].append("pattern")

        # Append session record to lessons file
        lessons_file = _get_lessons_file()
        if lessons_file.exists():
            date_str = datetime.now().strftime("%Y-%m-%d %H:%M")
            entry = (
                f"\n### SESSION-{datetime.now().strftime('%Y%m%d-%H%M')} | session | auto | {date_str}\n"
                f"**Tags:** session-record,{outcome}\n"
                f"**Content:** Task {task_id} — outcome: {outcome}"
                f"{f' | tokens: {tokens_spent}' if tokens_spent else ''}\n"
                f"**Source:** auto-recorded\n"
            )
            with open(lessons_file, "a", encoding="utf-8") as f:
                f.write(entry)

        # Dual-write session to holographic
        holo = _get_holographic_store()
        if holo:
            try:
                session_content = f"Task {task_id} — outcome: {outcome}"
                if tokens_spent:
                    session_content += f" | tokens: {tokens_spent}"
                holo.add_fact(
                    content=session_content,
                    category="session",
                    tags=f"session-record,{outcome}",
                )
            except Exception as exc:
                logger.warning("Holographic session write failed: %s", exc)

        results["timestamp"] = datetime.now().isoformat()

        return json.dumps(results, indent=2)

    # -- Bootstrap / seed tool ------------------------------------------------

    @mcp.tool()
    def memorygrid_seed_holographic() -> str:
        """Import existing markdown lessons into the holographic SQLite store.

        Idempotent — lessons already present (by content) are skipped due to
        the UNIQUE constraint on facts.content. Safe to call multiple times.

        Use this after a fresh install to populate the holographic DB from the
        tracked memorygrid-lessons.md file.
        """
        lessons_file = _get_lessons_file()
        holo = _get_holographic_store()

        if not holo:
            return json.dumps({
                "status": "unavailable",
                "message": "Holographic store not initialised",
            })

        if not lessons_file.exists():
            return json.dumps({
                "status": "no_source",
                "message": "No lessons markdown file found",
            })

        try:
            content = lessons_file.read_text(encoding="utf-8")
        except Exception as e:
            return json.dumps({"status": "error", "message": str(e)})

        lessons = _parse_lessons(content)
        imported = 0
        skipped = 0

        for lesson in lessons:
            if lesson["type"] == "session":
                skipped += 1
                continue
            try:
                holo.add_fact(
                    content=f"[{lesson['id']}] {lesson['content']}",
                    category=lesson["type"],
                    tags=",".join(lesson["tags"]),
                )
                imported += 1
            except Exception:
                skipped += 1

        return json.dumps({
            "status": "seeded",
            "total_in_file": len(lessons),
            "imported": imported,
            "skipped": skipped,
        }, indent=2)
