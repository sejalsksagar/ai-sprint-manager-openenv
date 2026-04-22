"""
Project data loader — Round 2.

Loads multi-sprint scenario data from data/project_data.json.
Uses module-level cache so disk is read exactly once per process.
Mirrors the pattern of sprint_env/data_loader.py exactly.

Public API
----------
    load_project_data(data_path=None)  → raw dict
    get_project_scenario_names()       → list[str]
    build_project_scenario(name)       → (tasks, developers, instructions, absences, meta)
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

from sprint_env.tasks import Task, Developer, TaskType, TaskStatus

# ── Path resolution ────────────────────────────────────────────────────────────

DEFAULT_DATA_PATH = Path(__file__).parent.parent / "data" / "project_data.json"

# Module-level cache — loaded once, reused forever (same as R1)
_PROJECT_DATA_CACHE: Optional[dict] = None


# ── Loader ─────────────────────────────────────────────────────────────────────

def load_project_data(data_path: Optional[str] = None) -> dict:
    """
    Load project_data.json, caching in memory after the first read.

    Args:
        data_path: Override path. Falls back to PROJECT_DATA_PATH env var,
                   then to data/project_data.json relative to this file.

    Returns:
        Full parsed JSON dict.

    Raises:
        FileNotFoundError: If the JSON file cannot be found.
    """
    global _PROJECT_DATA_CACHE
    if _PROJECT_DATA_CACHE is not None:
        return _PROJECT_DATA_CACHE

    path = Path(data_path or os.getenv("PROJECT_DATA_PATH", str(DEFAULT_DATA_PATH)))
    if not path.exists():
        raise FileNotFoundError(
            f"Project data file not found: {path}\n"
            "Expected at data/project_data.json relative to repo root."
        )

    with open(path, "r", encoding="utf-8") as f:
        _PROJECT_DATA_CACHE = json.load(f)

    return _PROJECT_DATA_CACHE


def invalidate_cache() -> None:
    """
    Clear the in-memory cache. Useful in tests that swap data files.
    Not needed in normal production use.
    """
    global _PROJECT_DATA_CACHE
    _PROJECT_DATA_CACHE = None


# ── Scenario helpers ───────────────────────────────────────────────────────────

def get_project_scenario_names(data_path: Optional[str] = None) -> list[str]:
    """Return the list of available scenario names from project_data.json."""
    return list(load_project_data(data_path)["scenarios"].keys())


def build_project_scenario(
    scenario_name: str,
    data_path: Optional[str] = None,
) -> tuple[list[Task], list[Developer], list[dict], list[dict], dict]:
    """
    Build typed Task and Developer objects for a named scenario.

    Args:
        scenario_name: e.g. "project_easy", "project_medium", "project_hard"
        data_path:     Optional override for the JSON path.

    Returns:
        tasks        : List[Task]  — all tasks, status=BACKLOG, with R2 metadata
        developers   : List[Developer]
        instructions : List[dict] — raw instruction dicts from JSON
        absences     : List[dict] — scheduled absence windows (may be empty)
        meta         : dict       — description, difficulty, num_sprints, days_per_sprint

    Raises:
        ValueError: If scenario_name is not found in the data file.
    """
    data = load_project_data(data_path)
    scenarios = data["scenarios"]

    if scenario_name not in scenarios:
        available = list(scenarios.keys())
        raise ValueError(
            f"Unknown scenario '{scenario_name}'. "
            f"Available: {available}"
        )

    scenario = scenarios[scenario_name]

    meta = {
        "description":    scenario.get("description", ""),
        "difficulty":     scenario.get("difficulty", "unknown"),
        "num_sprints":    scenario.get("num_sprints", 6),
        "days_per_sprint": scenario.get("days_per_sprint", 10),
    }

    developers: list[Developer] = [
        Developer(
            id=d["id"],
            name=d["name"],
            skill=d["skill"],
            capacity=d["capacity"],
            productivity=d.get("productivity", 1.0),
        )
        for d in scenario["developers"]
    ]

    tasks: list[Task] = []
    for t in scenario["tasks"]:
        task = Task(
            id=t["id"],
            name=t["name"],
            task_type=TaskType(t["task_type"]),
            priority=t["priority"],
            effort=t["effort"],
            deadline=t["deadline_day"],      # absolute day 1-60
            required_skill=t["required_skill"],
            status=TaskStatus.BACKLOG,
        )
        # Store R2-specific metadata (sprint assignment, dependencies)
        task.metadata = {
            "sprint":       t["sprint"],
            "deadline_day": t["deadline_day"],
            "depends_on":   t.get("depends_on", []),
            "tech_debt":    False,
        }
        tasks.append(task)

    instructions: list[dict] = scenario.get("instructions", [])
    absences: list[dict]     = scenario.get("absences", [])

    return tasks, developers, instructions, absences, meta