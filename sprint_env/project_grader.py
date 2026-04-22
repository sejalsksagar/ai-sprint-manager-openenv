"""
Project graders — Round 2 multi-sprint scoring functions.

Three grading tiers, one per difficulty:

    grade_project_easy(...)
    grade_project_medium(...)
    grade_project_hard(...)

Each grader operates at three levels and returns a GradeResult:

    step_score    — per-step quality signal (0.01-0.99)
                    Same intent as R1 graders: rewards good assignments,
                    penalises overload / wrong-skill assignments.

    sprint_score  — called at every sprint boundary (days 10,20,30,40,50,60).
                    Delivery rate × instruction-following × team health.

    project_score — final score at day 60.
                    delivery_rate × instruction_following × team_health,
                    weighted differently per difficulty.
                    This is the number judges see in evaluate_r2.py.

All scores clamped to (0.01, 0.99) — exactly 0.0 or 1.0 would fail the
OpenEnv validator.

Usage in project_environment.py:
    from sprint_env.project_grader import (
        grade_project_easy, grade_project_medium, grade_project_hard,
        grade_step, grade_sprint, GradeResult,
    )
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from sprint_env.tasks import Task, TaskStatus, TaskType


# ─── Clamp helper ─────────────────────────────────────────────────────────────

def _clamp(v: float) -> float:
    """Clamp to (0.01, 0.99) — never exactly 0 or 1."""
    return round(max(0.01, min(0.99, v)), 4)


# ─── Return type ──────────────────────────────────────────────────────────────

@dataclass
class GradeResult:
    """
    Structured grading output returned by all project graders.

    step_score    : per-step signal (call after every action)
    sprint_score  : boundary signal (call at end of each sprint)
    project_score : final signal (call once at day 60)

    breakdown     : dict of named sub-scores for logging / UI display
    """
    step_score: float = 0.01
    sprint_score: float = 0.01
    project_score: float = 0.01
    breakdown: Dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.step_score    = _clamp(self.step_score)
        self.sprint_score  = _clamp(self.sprint_score)
        self.project_score = _clamp(self.project_score)


# ─── Shared sub-scorers (used by all three graders) ───────────────────────────

def _delivery_rate(tasks: List[Task], sprint: Optional[int] = None) -> float:
    """
    Fraction of tasks that are DONE.
    If sprint is given, only considers tasks for that sprint number.
    """
    subset = tasks
    if sprint is not None:
        subset = [t for t in tasks if t.metadata.get("sprint") == sprint]
    if not subset:
        return 1.0
    done = sum(1 for t in subset if t.status == TaskStatus.DONE)
    return done / len(subset)


def _priority_weighted_delivery(
    tasks: List[Task],
    weights: Dict[int, float],
    sprint: Optional[int] = None,
    late_multiplier: float = 0.5,
    partial_multiplier: float = 0.25,
) -> float:
    """
    Weighted delivery rate giving higher credit to high-priority tasks.
    Partial credit for in-progress tasks, reduced credit for late completions.
    """
    subset = tasks
    if sprint is not None:
        subset = [t for t in tasks if t.metadata.get("sprint") == sprint]
    if not subset:
        return 1.0

    total_w = sum(weights.get(t.priority, 1.0) for t in subset)
    if total_w == 0:
        return 0.0

    earned = 0.0
    for t in subset:
        w = weights.get(t.priority, 1.0)
        if t.status == TaskStatus.DONE:
            on_time = t.days_in_progress <= (t.deadline - t.created_day + 1)
            earned += w if on_time else w * late_multiplier
        elif t.status == TaskStatus.IN_PROGRESS:
            earned += w * t.progress * partial_multiplier

    return earned / total_w


def _team_balance(developers: list) -> float:
    """
    1.0 = perfectly balanced load; 0.0 = maximally unbalanced.
    Mirrors R1 grader logic.
    """
    ratios = [
        d.current_load / d.capacity
        for d in developers
        if getattr(d, "capacity", 0) > 0
    ]
    if len(ratios) < 2:
        return 1.0
    mean = sum(ratios) / len(ratios)
    variance = sum((r - mean) ** 2 for r in ratios) / len(ratios)
    return max(0.0, 1.0 - variance ** 0.5)


def _instruction_following(
    released: List[dict], followed: List[str]
) -> float:
    """Fraction of released instructions that have been acted on."""
    if not released:
        return 1.0
    return len(followed) / len(released)


def _tech_debt_drag(tech_debt: List[str]) -> float:
    """
    Productivity drag from accumulated tech debt.
    Each missed task costs 2%, capped at 40%.
    """
    return min(0.40, len(tech_debt) * 0.02)


def _urgent_bug_penalty(tasks: List[Task]) -> float:
    """0.20 penalty for every missed urgent bug."""
    missed_urgent = sum(
        1 for t in tasks
        if t.status == TaskStatus.MISSED and t.task_type == TaskType.URGENT_BUG
    )
    return min(0.60, missed_urgent * 0.20)


def _burnout_penalty(developers: list) -> float:
    """0.06 penalty per dev working over 120% capacity."""
    return min(0.30, sum(
        0.06 for d in developers
        if getattr(d, "current_load", 0) > getattr(d, "capacity", 1) * 1.2
    ))


def _dependency_violation_penalty(tasks: List[Task]) -> float:
    """
    Penalty for tasks that were started before their dependencies were done.
    Detected by checking DONE tasks whose depends_on contains non-DONE tasks.
    """
    task_map = {t.id: t for t in tasks}
    violations = 0
    for t in tasks:
        if t.status == TaskStatus.DONE:
            for dep_id in t.metadata.get("depends_on", []):
                dep = task_map.get(dep_id)
                if dep and dep.status != TaskStatus.DONE:
                    violations += 1
    return min(0.30, violations * 0.05)


# ─── Step grader (shared across all difficulties) ────────────────────────────

def grade_step(
    action_type: str,
    task: Optional[Task],
    dev,
    tasks: List[Task],
    developers: list,
) -> float:
    """
    Per-step quality signal — called after every agent action.

    Rewards:
      + skill match on assign
      + assigning high-priority tasks first
      + following an active instruction

    Penalises:
      - wrong-skill assignment
      - skip / no-op
      - assigning to overloaded dev
    """
    atype = (action_type or "skip").lower()

    if atype == "skip":
        return _clamp(-0.05)

    if task is None or dev is None:
        return _clamp(-0.15)

    reward = 0.0

    if atype in ("assign", "reassign"):
        skill_bonus  = 0.30 if getattr(dev, "skill", "") == task.required_skill else 0.05
        prio_bonus   = (6 - task.priority) * 0.08   # priority 1 → +0.40, priority 5 → +0.08
        overload_pen = -0.15 if (
            getattr(dev, "current_load", 0) + task.effort > getattr(dev, "capacity", 1) * 1.5
        ) else 0.0
        reward = 0.40 + skill_bonus + prio_bonus + overload_pen

    elif atype == "reprioritize":
        reward = 0.08

    elif atype == "unblock":
        reward = 0.25

    elif atype == "sprint_plan":
        reward = 0.15   # small planning bonus

    return _clamp(reward)


# ─── Sprint grader (shared, parameterised per difficulty) ─────────────────────

def grade_sprint(
    tasks: List[Task],
    developers: list,
    released_instructions: List[dict],
    followed_instructions: List[str],
    tech_debt: List[str],
    completed_sprint: int,
    *,
    delivery_weight: float = 0.50,
    inst_weight: float     = 0.30,
    health_weight: float   = 0.20,
    priority_weights: Optional[Dict[int, float]] = None,
) -> float:
    """
    Sprint-boundary score. Called at days 10, 20, 30, 40, 50, 60.

    score = delivery × w_d + instruction_following × w_i + team_health × w_h
    All weights must sum to 1.0 (enforced via normalisation).
    """
    if priority_weights is None:
        priority_weights = {1: 3.0, 2: 2.0, 3: 1.5, 4: 1.0, 5: 0.5}

    delivery  = _priority_weighted_delivery(tasks, priority_weights, sprint=completed_sprint)
    inst      = _instruction_following(released_instructions, followed_instructions)
    balance   = _team_balance(developers)
    debt_drag = _tech_debt_drag(tech_debt)

    # Normalise weights in case caller passes non-summing values
    total_w = delivery_weight + inst_weight + health_weight
    w_d = delivery_weight / total_w
    w_i = inst_weight     / total_w
    w_h = health_weight   / total_w

    team_health = max(0.0, balance - debt_drag)

    score = delivery * w_d + inst * w_i + team_health * w_h
    return _clamp(score)


# ─── Easy project grader ──────────────────────────────────────────────────────

def grade_project_easy(
    tasks: List[Task],
    developers: list,
    released_instructions: List[dict],
    followed_instructions: List[str],
    tech_debt: List[str],
    sprint_scores: List[float],
) -> GradeResult:
    """
    Easy project grader.

    project_score = delivery_rate × instruction_following × team_health
    Weights: delivery 60%, instructions 25%, health 15%.
    No urgent-bug or dependency-violation penalties.
    """
    delivery  = _delivery_rate(tasks)
    inst      = _instruction_following(released_instructions, followed_instructions)
    balance   = _team_balance(developers)
    debt_drag = _tech_debt_drag(tech_debt)
    team_health = max(0.0, balance - debt_drag)

    project_score = (
        0.60 * delivery
        + 0.25 * inst
        + 0.15 * team_health
    )

    # Sprint consistency bonus: reward low variance across sprint scores
    if len(sprint_scores) > 1:
        mean_s = sum(sprint_scores) / len(sprint_scores)
        variance_s = sum((s - mean_s) ** 2 for s in sprint_scores) / len(sprint_scores)
        consistency = max(0.0, 1.0 - variance_s ** 0.5)
        project_score += 0.05 * consistency  # small bonus up to 5%

    # Step score: simple delivery snapshot
    step_score = _clamp(delivery)

    # Sprint score: most recent sprint or default
    sprint_score = _clamp(sprint_scores[-1]) if sprint_scores else _clamp(0.5)

    breakdown = {
        "delivery_rate":            round(delivery, 4),
        "instruction_following":    round(inst, 4),
        "team_health":              round(team_health, 4),
        "tech_debt_drag":           round(debt_drag, 4),
        "workload_balance":         round(balance, 4),
        "sprint_consistency_bonus": round(
            0.05 * max(0.0, 1.0 - (
                (sum((s - sum(sprint_scores)/len(sprint_scores))**2
                     for s in sprint_scores) / len(sprint_scores)) ** 0.5
                if len(sprint_scores) > 1 else 0.0
            )), 4
        ) if sprint_scores else 0.0,
    }

    return GradeResult(
        step_score=step_score,
        sprint_score=sprint_score,
        project_score=project_score,
        breakdown=breakdown,
    )


# ─── Medium project grader ────────────────────────────────────────────────────

def grade_project_medium(
    tasks: List[Task],
    developers: list,
    released_instructions: List[dict],
    followed_instructions: List[str],
    tech_debt: List[str],
    sprint_scores: List[float],
) -> GradeResult:
    """
    Medium project grader.

    project_score = weighted_delivery × inst_following × team_health
    Weights: delivery 55%, instructions 30%, health 15%.
    Adds: burnout penalty, dependency-violation penalty.
    Priority weights give more importance to P1/P2 tasks.
    """
    priority_weights = {1: 3.0, 2: 2.0, 3: 1.5, 4: 1.0, 5: 0.5}

    delivery    = _priority_weighted_delivery(tasks, priority_weights, late_multiplier=0.5)
    inst        = _instruction_following(released_instructions, followed_instructions)
    balance     = _team_balance(developers)
    debt_drag   = _tech_debt_drag(tech_debt)
    burnout_pen = _burnout_penalty(developers)
    dep_pen     = _dependency_violation_penalty(tasks)

    team_health = max(0.0, balance - debt_drag)

    project_score = (
        0.55 * delivery
        + 0.30 * inst
        + 0.15 * team_health
        - burnout_pen
        - dep_pen
    )

    # Sprint trajectory bonus: reward improving trend across sprints
    trajectory_bonus = 0.0
    if len(sprint_scores) >= 3:
        # Simple linear trend: positive slope = improving
        n = len(sprint_scores)
        xs = list(range(n))
        mean_x = sum(xs) / n
        mean_y = sum(sprint_scores) / n
        num   = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, sprint_scores))
        denom = sum((x - mean_x) ** 2 for x in xs) or 1
        slope = num / denom
        trajectory_bonus = min(0.08, max(0.0, slope * 4))   # up to 8% bonus

    project_score += trajectory_bonus

    step_score   = _clamp(delivery * 0.6 + balance * 0.4)
    sprint_score = _clamp(sprint_scores[-1]) if sprint_scores else _clamp(0.3)

    breakdown = {
        "weighted_delivery":         round(delivery, 4),
        "instruction_following":     round(inst, 4),
        "team_health":               round(team_health, 4),
        "tech_debt_drag":            round(debt_drag, 4),
        "burnout_penalty":           round(burnout_pen, 4),
        "dependency_violation_pen":  round(dep_pen, 4),
        "trajectory_bonus":          round(trajectory_bonus, 4),
    }

    return GradeResult(
        step_score=step_score,
        sprint_score=sprint_score,
        project_score=project_score,
        breakdown=breakdown,
    )


# ─── Hard project grader ──────────────────────────────────────────────────────

def grade_project_hard(
    tasks: List[Task],
    developers: list,
    released_instructions: List[dict],
    followed_instructions: List[str],
    tech_debt: List[str],
    sprint_scores: List[float],
) -> GradeResult:
    """
    Hard project grader.

    project_score = weighted_delivery × inst_following × team_health
    Weights: delivery 50%, instructions 35%, health 15%.
    Adds: urgent-bug penalty, burnout penalty, dependency-violation penalty.
    Requires near-perfect execution to score above 0.75.
    Priority weights heavily favour P1 tasks.
    """
    priority_weights = {1: 5.0, 2: 3.0, 3: 1.5, 4: 0.8, 5: 0.3}

    delivery    = _priority_weighted_delivery(
        tasks, priority_weights, late_multiplier=0.35, partial_multiplier=0.15
    )
    inst        = _instruction_following(released_instructions, followed_instructions)
    balance     = _team_balance(developers)
    debt_drag   = _tech_debt_drag(tech_debt)
    burnout_pen = _burnout_penalty(developers)
    urgent_pen  = _urgent_bug_penalty(tasks)
    dep_pen     = _dependency_violation_penalty(tasks)

    team_health = max(0.0, balance - debt_drag)

    project_score = (
        0.50 * delivery
        + 0.35 * inst
        + 0.15 * team_health
        - burnout_pen
        - urgent_pen
        - dep_pen
    )

    # Hard: late-sprint recovery bonus — if last two sprints score better than average
    recovery_bonus = 0.0
    if len(sprint_scores) >= 4:
        early_avg = sum(sprint_scores[:-2]) / (len(sprint_scores) - 2)
        late_avg  = sum(sprint_scores[-2:]) / 2
        if late_avg > early_avg:
            recovery_bonus = min(0.06, (late_avg - early_avg) * 0.5)

    project_score += recovery_bonus

    step_score   = _clamp(delivery * 0.5 + balance * 0.3 + inst * 0.2)
    sprint_score = _clamp(sprint_scores[-1]) if sprint_scores else _clamp(0.2)

    breakdown = {
        "weighted_delivery":         round(delivery, 4),
        "instruction_following":     round(inst, 4),
        "team_health":               round(team_health, 4),
        "tech_debt_drag":            round(debt_drag, 4),
        "burnout_penalty":           round(burnout_pen, 4),
        "urgent_bug_penalty":        round(urgent_pen, 4),
        "dependency_violation_pen":  round(dep_pen, 4),
        "recovery_bonus":            round(recovery_bonus, 4),
    }

    return GradeResult(
        step_score=step_score,
        sprint_score=sprint_score,
        project_score=project_score,
        breakdown=breakdown,
    )


# ─── Convenience dispatcher ───────────────────────────────────────────────────

_GRADER_MAP = {
    "project_easy":   grade_project_easy,
    "project_medium": grade_project_medium,
    "project_hard":   grade_project_hard,
}


def grade_project(
    scenario_name: str,
    tasks: List[Task],
    developers: list,
    released_instructions: List[dict],
    followed_instructions: List[str],
    tech_debt: List[str],
    sprint_scores: List[float],
) -> GradeResult:
    """
    Dispatcher — call with scenario_name to get the right grader automatically.

    >>> result = grade_project("project_hard", tasks, devs, released, followed, debt, sprints)
    >>> result.project_score   # final judge score
    >>> result.breakdown       # sub-score dict for UI display
    """
    grader = _GRADER_MAP.get(scenario_name, grade_project_easy)
    return grader(tasks, developers, released_instructions, followed_instructions,
                  tech_debt, sprint_scores)