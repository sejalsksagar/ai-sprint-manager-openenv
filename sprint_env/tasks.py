"""
Task and Developer data classes for the Sprint Manager environment.
Data is loaded from data/sprint_data.json via data_loader.py.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum


class TaskType(str, Enum):
    FEATURE     = "feature"
    BUG         = "bug"
    URGENT_BUG  = "urgent_bug"
    TECH_DEBT   = "tech_debt"


class TaskStatus(str, Enum):
    BACKLOG     = "backlog"
    IN_PROGRESS = "in_progress"
    BLOCKED     = "blocked"
    DONE        = "done"
    MISSED      = "missed"


@dataclass
class Task:
    id: str
    name: str
    task_type: TaskType
    priority: int           # 1 (highest) → 5 (lowest)
    effort: int             # story points 1–8
    deadline: int           # sprint day deadline 1–10
    required_skill: str     # "frontend" | "backend" | "devops" | "fullstack"
    status: TaskStatus = TaskStatus.BACKLOG
    assigned_to: Optional[str] = None
    progress: float = 0.0   # 0.0 → 1.0
    days_in_progress: int = 0
    created_day: int = 1

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "task_type": self.task_type.value,
            "priority": self.priority,
            "effort": self.effort,
            "deadline": self.deadline,
            "required_skill": self.required_skill,
            "status": self.status.value,
            "assigned_to": self.assigned_to,
            "progress": round(self.progress, 2),
            "days_in_progress": self.days_in_progress,
            "created_day": self.created_day,
        }


@dataclass
class Developer:
    id: str
    name: str
    skill: str              # "frontend" | "backend" | "devops" | "fullstack"
    capacity: int           # max story points per day
    current_load: int = 0
    assigned_tasks: list = field(default_factory=list)
    is_available: bool = True
    productivity: float = 1.0

    def can_take_task(self, task: Task) -> bool:
        if not self.is_available:
            return False
        if self.current_load >= self.capacity:
            return False
        return (
            self.skill == task.required_skill
            or self.skill == "fullstack"
            or task.required_skill == "fullstack"
        )

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "skill": self.skill,
            "capacity": self.capacity,
            "current_load": self.current_load,
            "assigned_tasks": list(self.assigned_tasks),
            "is_available": self.is_available,
            "productivity": round(self.productivity, 2),
        }