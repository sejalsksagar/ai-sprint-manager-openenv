"""
OpenEnv-compliant Pydantic models for Round 2 — multi-sprint project environment.

Extends the R1 models (SprintAction, SprintObservation, SprintState) with three
new classes:

    ProjectAction      — adds "sprint_plan" action type; backwards-compatible
                         with SprintAction (same fields, stricter validator)
    ProjectObservation — all R1 fields + current_sprint, instruction_queue,
                         instruction_following_score, tech_debt, sprint_rewards
    ProjectState       — all R1 fields + full project-scope tracking fields

R1 models are NOT modified. Import both from their respective modules.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


# ─── Instruction model ────────────────────────────────────────────────────────

class ProjectInstruction(BaseModel):
    """
    A time-released instruction delivered to the agent during the project.
    Mirrors the structure in project_data.json.
    """

    id: str = Field(description="Instruction ID, e.g. 'I01'")
    release_day: int = Field(description="Absolute project day the instruction becomes visible")
    text: str = Field(description="Natural-language instruction text")
    target_sprint: int = Field(description="Sprint number this instruction applies to (1-6)")
    affects_tasks: List[str] = Field(
        default_factory=list,
        description="Task IDs the instruction references",
    )
    followed: bool = Field(
        default=False,
        description="True once the agent has acted on at least one affected task",
    )

    @field_validator("release_day")
    @classmethod
    def release_day_positive(cls, v: int) -> int:
        if v < 1:
            raise ValueError("release_day must be >= 1")
        return v

    @field_validator("target_sprint")
    @classmethod
    def sprint_in_range(cls, v: int) -> int:
        if not (1 <= v <= 6):
            raise ValueError("target_sprint must be 1-6")
        return v


# ─── Action ───────────────────────────────────────────────────────────────────

# Valid action types for R2 (superset of R1)
_VALID_ACTION_TYPES = {"assign", "reassign", "reprioritize", "skip", "unblock", "sprint_plan"}


class ProjectAction(BaseModel):
    """
    R2 action — backwards-compatible with SprintAction.

    action_type options (R1 + new):
        "assign"       : Assign a backlog task to a developer
        "reassign"     : Move an in-progress task to a different developer
        "reprioritize" : Change a task's priority (1=highest, 5=lowest)
        "unblock"      : Clear a blocked task so it can be assigned
        "skip"         : Do nothing this step
        "sprint_plan"  : NEW — submit a batch sprint plan for the next sprint
                         (task_ids lists tasks to prioritise; agent earns a
                         small bonus for planning ahead of sprint boundary)

    Fields:
        task_id      : Single task ID for assign / reassign / reprioritize / unblock
        dev_id       : Developer ID for assign / reassign
        new_priority : New priority value 1-5 for reprioritize
        task_ids     : NEW — list of task IDs for sprint_plan batch action
        notes        : NEW — optional free-text reasoning (logged, not scored)
    """

    action_type: str = Field(
        default="skip",
        description="One of: assign, reassign, reprioritize, skip, unblock, sprint_plan",
    )
    task_id: Optional[str] = Field(default=None, description="Task ID to act on")
    dev_id: Optional[str] = Field(default=None, description="Developer ID to assign to")
    new_priority: Optional[int] = Field(
        default=None, description="New priority 1-5 (for reprioritize)"
    )
    # R2-only fields
    task_ids: Optional[List[str]] = Field(
        default=None,
        description="R2: list of task IDs for sprint_plan batch action",
    )
    notes: Optional[str] = Field(
        default=None,
        description="R2: optional agent reasoning / rationale (logged only)",
    )

    @field_validator("action_type")
    @classmethod
    def action_type_valid(cls, v: str) -> str:
        normalised = v.lower().strip()
        if normalised not in _VALID_ACTION_TYPES:
            raise ValueError(
                f"action_type '{v}' not recognised. "
                f"Must be one of: {sorted(_VALID_ACTION_TYPES)}"
            )
        return normalised

    @field_validator("new_priority")
    @classmethod
    def priority_in_range(cls, v: Optional[int]) -> Optional[int]:
        if v is not None and v not in range(1, 6):
            raise ValueError("new_priority must be between 1 and 5 inclusive")
        return v

    @model_validator(mode="after")
    def cross_field_checks(self) -> "ProjectAction":
        atype = self.action_type
        if atype == "assign" and (self.task_id is None or self.dev_id is None):
            raise ValueError("assign requires both task_id and dev_id")
        if atype == "reassign" and (self.task_id is None or self.dev_id is None):
            raise ValueError("reassign requires both task_id and dev_id")
        if atype == "reprioritize" and self.task_id is None:
            raise ValueError("reprioritize requires task_id")
        if atype == "reprioritize" and self.new_priority is None:
            raise ValueError("reprioritize requires new_priority")
        if atype == "unblock" and self.task_id is None:
            raise ValueError("unblock requires task_id")
        if atype == "sprint_plan" and not self.task_ids:
            raise ValueError("sprint_plan requires task_ids (non-empty list)")
        return self

    def to_sprint_action_dict(self) -> Dict[str, Any]:
        """
        Returns a dict compatible with the R1 SprintAction for use in the
        existing environment's _apply_action without any import changes.
        """
        return {
            "action_type": self.action_type,
            "task_id": self.task_id,
            "dev_id": self.dev_id,
            "new_priority": self.new_priority,
        }


# ─── Observation ─────────────────────────────────────────────────────────────

class ProjectObservation(BaseModel):
    """
    What the agent sees after each step in the multi-sprint environment.

    Includes all R1 SprintObservation fields plus R2 project-scope fields.
    current_day now counts 1-60 (not 1-10).
    """

    # ── Core fields (identical to R1 SprintObservation) ──────────────────────
    current_day: int = Field(description="Absolute project day (1-60)")
    sprint_length: int = Field(description="Total project length in days (60)")
    task_id: Optional[str] = Field(default=None, description="Active scenario name")

    developers: List[Dict[str, Any]] = Field(description="Developer states")
    tasks: List[Dict[str, Any]] = Field(description="All task states")

    reward: float = Field(default=0.0, description="Step reward")
    cumulative_reward: float = Field(default=0.0, description="Total reward so far")

    tasks_completed: int = Field(default=0)
    tasks_missed: int = Field(default=0)
    tasks_in_progress: int = Field(default=0)
    tasks_backlog: int = Field(default=0)
    workload_balance_score: float = Field(
        default=0.0, description="0=unbalanced, 1=perfectly balanced"
    )

    events: List[str] = Field(default_factory=list, description="Events this step")
    done: bool = Field(default=False)
    info: Dict[str, Any] = Field(default_factory=dict)

    # ── R2 extension fields ───────────────────────────────────────────────────
    current_sprint: int = Field(
        default=1, description="Current sprint number (1-6)"
    )
    instruction_queue: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="All instructions released up to current_day",
    )
    instruction_following_score: float = Field(
        default=1.0,
        description="Fraction of released instructions acted on (0.01-0.99)",
    )
    tech_debt: List[str] = Field(
        default_factory=list,
        description="Task IDs that became tech debt (missed at sprint boundary)",
    )
    sprint_rewards: List[float] = Field(
        default_factory=list,
        description="Reward earned at each sprint boundary (one entry per completed sprint)",
    )

    @field_validator("current_day")
    @classmethod
    def day_in_range(cls, v: int) -> int:
        if not (1 <= v <= 61):   # 61 allows the post-termination state
            raise ValueError(f"current_day must be 1-61, got {v}")
        return v

    @field_validator("current_sprint")
    @classmethod
    def sprint_in_range(cls, v: int) -> int:
        if not (1 <= v <= 6):
            raise ValueError(f"current_sprint must be 1-6, got {v}")
        return v

    @field_validator("instruction_following_score")
    @classmethod
    def inst_score_clamped(cls, v: float) -> float:
        return max(0.01, min(0.99, v))

    @field_validator("workload_balance_score")
    @classmethod
    def balance_clamped(cls, v: float) -> float:
        return max(0.0, min(1.0, v))

    def active_instructions(self) -> List[Dict[str, Any]]:
        """Return instructions that have been released but not yet followed."""
        return [i for i in self.instruction_queue if not i.get("followed", False)]

    def current_sprint_tasks(self) -> List[Dict[str, Any]]:
        """Return tasks belonging to the current sprint."""
        return [
            t for t in self.tasks
            if t.get("metadata", {}).get("sprint") == self.current_sprint
        ]

    def days_remaining_in_sprint(self) -> int:
        """Days left before the current sprint boundary."""
        sprint_end = self.current_sprint * 10
        return max(0, sprint_end - self.current_day + 1)


# ─── State ────────────────────────────────────────────────────────────────────

class ProjectState(BaseModel):
    """
    Full internal state snapshot for the /project/state endpoint.

    Includes all R1 SprintState fields plus full project-scope tracking.
    """

    # ── Core fields (mirrors R1 SprintState) ─────────────────────────────────
    episode_id: str
    task_name: str
    current_day: int
    sprint_length: int = Field(description="Total days in episode (60)")
    step_count: int
    tasks: List[Dict[str, Any]]
    developers: List[Dict[str, Any]]
    cumulative_reward: float
    done: bool
    events_log: List[str]

    # ── R2 extension fields ───────────────────────────────────────────────────
    current_sprint: int = Field(default=1, description="Current sprint (1-6)")
    num_sprints: int = Field(default=6, description="Total sprints in project")
    days_per_sprint: int = Field(default=10)

    released_instructions: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="All instructions released to the agent so far",
    )
    followed_instructions: List[str] = Field(
        default_factory=list,
        description="Instruction IDs the agent has acted on",
    )
    instruction_following_score: float = Field(
        default=1.0,
        description="Fraction of released instructions followed (0.01-0.99)",
    )

    tech_debt: List[str] = Field(
        default_factory=list,
        description="Task IDs carried as tech debt across sprint boundaries",
    )
    sprint_rewards: List[float] = Field(
        default_factory=list,
        description="Reward at each sprint boundary (max 6 entries)",
    )

    # Derived summary — populated on creation
    tasks_completed: int = Field(default=0)
    tasks_missed: int = Field(default=0)
    tasks_in_progress: int = Field(default=0)
    tasks_backlog: int = Field(default=0)

    @model_validator(mode="after")
    def compute_task_counts(self) -> "ProjectState":
        """Auto-populate task count fields from the tasks list."""
        status_map: Dict[str, int] = {
            "done": 0, "missed": 0, "in_progress": 0, "backlog": 0
        }
        for t in self.tasks:
            s = t.get("status", "backlog")
            if s in status_map:
                status_map[s] += 1
        self.tasks_completed = status_map["done"]
        self.tasks_missed = status_map["missed"]
        self.tasks_in_progress = status_map["in_progress"]
        self.tasks_backlog = status_map["backlog"]
        return self

    @field_validator("instruction_following_score")
    @classmethod
    def inst_score_clamped(cls, v: float) -> float:
        return max(0.01, min(0.99, v))

    def sprint_progress_pct(self) -> float:
        """Fraction of the project completed (0.0-1.0)."""
        return min(1.0, (self.current_day - 1) / max(1, self.sprint_length))

    def pending_instructions(self) -> List[Dict[str, Any]]:
        """Released instructions not yet followed."""
        followed_set = set(self.followed_instructions)
        return [i for i in self.released_instructions if i["id"] not in followed_set]