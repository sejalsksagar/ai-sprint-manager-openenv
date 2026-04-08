"""
Loads sprint scenario data from JSON file.
Uses module-level cache so disk is read only once.
"""
from __future__ import annotations
import json
import os
from pathlib import Path
from typing import Optional
from functools import lru_cache

from sprint_env.tasks import Task, Developer, TaskType, TaskStatus

DEFAULT_DATA_PATH = Path(__file__).parent.parent / "data" / "sprint_data.json"

# Module-level cache — loaded once, reused forever
_DATA_CACHE: Optional[dict] = None


def load_sprint_data(data_path: Optional[str] = None) -> dict:
    global _DATA_CACHE
    if _DATA_CACHE is not None:
        return _DATA_CACHE
    path = Path(data_path or os.getenv("SPRINT_DATA_PATH", DEFAULT_DATA_PATH))
    if not path.exists():
        raise FileNotFoundError(f"Sprint data file not found: {path}")
    with open(path, "r") as f:
        _DATA_CACHE = json.load(f)
    return _DATA_CACHE


def get_scenario_names(data_path: Optional[str] = None) -> list[str]:
    return list(load_sprint_data(data_path)["scenarios"].keys())


def build_scenario(
    scenario_name: str, data_path: Optional[str] = None
) -> tuple[list[Task], list[Developer], dict]:
    data = load_sprint_data(data_path)
    scenarios = data["scenarios"]

    if scenario_name not in scenarios:
        available = list(scenarios.keys())
        raise ValueError(f"Unknown scenario '{scenario_name}'. Available: {available}")

    scenario = scenarios[scenario_name]
    meta = {
        "description": scenario.get("description", ""),
        "difficulty": scenario.get("difficulty", "unknown"),
    }

    developers = [
        Developer(
            id=d["id"], name=d["name"], skill=d["skill"],
            capacity=d["capacity"], productivity=d.get("productivity", 1.0),
        )
        for d in scenario["developers"]
    ]

    tasks = [
        Task(
            id=t["id"], name=t["name"], task_type=TaskType(t["task_type"]),
            priority=t["priority"], effort=t["effort"], deadline=t["deadline"],
            required_skill=t["required_skill"], status=TaskStatus.BACKLOG,
        )
        for t in scenario["tasks"]
    ]

    return tasks, developers, meta