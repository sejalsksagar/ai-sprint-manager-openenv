"""
Project Manager RL Environment — Round 2 multi-sprint logic.

Extends the single-sprint SprintManagerEnv to manage a full 6-sprint project
(60 days). State carries over between sprints: missed tasks become tech debt,
instructions are released on schedule, and a project-level reward is computed
at day 60.

Episode lifecycle:
    reset(task_name)  → day 1 of sprint 1
    step(action)      → agent acts, day advances (same action space as R1)
    state()           → full project state including cross-sprint fields
"""

from __future__ import annotations

import json
import random
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from sprint_env.tasks import Task, Developer, TaskStatus, TaskType
from sprint_env.models import SprintAction
from sprint_env.project_data_loader import load_project_data
# ---------------------------------------------------------------------------
# Data path
# ---------------------------------------------------------------------------

_DATA_PATH = Path(__file__).parent.parent / "data" / "project_data.json"

# Module-level cache — reads JSON once (mirrors R1 data_loader pattern)
_PROJECT_DATA: Optional[Dict[str, Any]] = None


def _load_project_data() -> Dict[str, Any]:
    global _PROJECT_DATA
    if _PROJECT_DATA is None:
        with open(_DATA_PATH, "r", encoding="utf-8") as f:
            _PROJECT_DATA = json.load(f)
    return _PROJECT_DATA


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DAYS_PER_SPRINT = 10
NUM_SPRINTS = 6
TOTAL_DAYS = DAYS_PER_SPRINT * NUM_SPRINTS  # 60

VALID_PROJECT_TASK_NAMES = ["project_easy", "project_medium", "project_hard"]

# Tech-debt penalty multiplier per missed task carried forward
TECH_DEBT_PENALTY = 0.05

# ---------------------------------------------------------------------------
# Helper: build tasks and developers from project_data.json scenario
# ---------------------------------------------------------------------------


def _build_project_scenario(
    scenario_name: str,
) -> Tuple[List[Task], List[Developer], List[Dict], List[Dict]]:
    """
    Returns (tasks, developers, instructions, absences) for a named scenario.
    Tasks are constructed as R1-compatible Task objects with the extra fields
    (sprint, deadline_day, depends_on) stored in task.metadata.
    """
    data = load_project_data()
    sc = data["scenarios"][scenario_name]

    developers: List[Developer] = []
    for d in sc["developers"]:
        dev = Developer(
            id=d["id"],
            name=d["name"],
            skill=d["skill"],
            capacity=d["capacity"],
            productivity=d["productivity"],
        )
        developers.append(dev)

    tasks: List[Task] = []
    for t in sc["tasks"]:
        task = Task(
            id=t["id"],
            name=t["name"],
            task_type=TaskType(t["task_type"]),
            priority=t["priority"],
            effort=t["effort"],
            deadline=t["deadline_day"],   # absolute day (1-60)
            required_skill=t["required_skill"],
        )
        # Store R2-specific metadata on the task object
        task.metadata = {
            "sprint": t["sprint"],
            "deadline_day": t["deadline_day"],
            "depends_on": t.get("depends_on", []),
            "tech_debt": False,   # True once it becomes carry-over debt
        }
        tasks.append(task)

    instructions: List[Dict] = sc.get("instructions", [])
    absences: List[Dict] = sc.get("absences", [])

    return tasks, developers, instructions, absences


# ---------------------------------------------------------------------------
# Observation / State dataclasses (plain dicts — Step 3 adds Pydantic models)
# ---------------------------------------------------------------------------


def _make_project_observation(
    *,
    current_day: int,
    current_sprint: int,
    episode_id: str,
    task_name: str,
    tasks: List[Task],
    developers: List[Developer],
    instruction_queue: List[Dict],
    instruction_following_score: float,
    tech_debt: List[str],
    reward: float,
    cumulative_reward: float,
    sprint_rewards: List[float],
    done: bool,
    step_count: int,
    events: List[str],
) -> Dict[str, Any]:
    done_count = sum(1 for t in tasks if t.status == TaskStatus.DONE)
    missed_count = sum(1 for t in tasks if t.status == TaskStatus.MISSED)
    in_prog = sum(1 for t in tasks if t.status == TaskStatus.IN_PROGRESS)
    backlog = sum(1 for t in tasks if t.status == TaskStatus.BACKLOG)

    load_ratios = [d.current_load / d.capacity for d in developers if d.capacity > 0]
    if load_ratios:
        mean = sum(load_ratios) / len(load_ratios)
        variance = sum((r - mean) ** 2 for r in load_ratios) / len(load_ratios)
        balance = max(0.0, 1.0 - variance ** 0.5)
    else:
        balance = 1.0

    return {
        # ── Core fields (same as R1 SprintObservation) ──
        "current_day": current_day,
        "sprint_length": TOTAL_DAYS,
        "task_id": task_name,
        "developers": [d.to_dict() for d in developers],
        "tasks": [t.to_dict() for t in tasks],
        "reward": round(reward, 4),
        "cumulative_reward": round(cumulative_reward, 4),
        "tasks_completed": done_count,
        "tasks_missed": missed_count,
        "tasks_in_progress": in_prog,
        "tasks_backlog": backlog,
        "workload_balance_score": round(balance, 4),
        "events": events,
        "done": done,
        "info": {"episode_id": episode_id, "step": step_count},
        # ── R2 extension fields ──
        "current_sprint": current_sprint,
        "instruction_queue": instruction_queue,
        "instruction_following_score": round(instruction_following_score, 4),
        "tech_debt": tech_debt,
        "sprint_rewards": sprint_rewards,
    }


# ---------------------------------------------------------------------------
# Main environment class
# ---------------------------------------------------------------------------


class ProjectManagerEnv:
    """
    OpenEnv-compliant multi-sprint Project Manager environment.

    Uses the same SprintAction action space as R1 (assign / reassign /
    reprioritize / unblock / skip), so the same LLM policy works across both
    environments.

    New R2 state fields exposed in every observation:
        current_sprint            int   1-6
        instruction_queue         list  instructions released up to current_day
        instruction_following_score float 0.0-1.0
        tech_debt                 list  task IDs carried as debt
        sprint_rewards            list  reward earned per completed sprint
    """

    def __init__(self) -> None:
        self._episode_id: str = ""
        self._task_name: str = "project_easy"
        self._current_day: int = 1
        self._current_sprint: int = 1
        self._step_count: int = 0
        self._done: bool = False

        self._tasks: List[Task] = []
        self._developers: List[Developer] = []
        self._all_instructions: List[Dict] = []
        self._absences: List[Dict] = []

        self._released_instructions: List[Dict] = []   # released so far
        self._followed_instructions: List[str] = []    # instruction IDs acted on
        self._tech_debt: List[str] = []                # task IDs that are tech debt

        self._cumulative_reward: float = 0.0
        self._sprint_rewards: List[float] = []         # reward per completed sprint
        self._events_log: List[str] = []

    # ── Reset ────────────────────────────────────────────────────────────────

    def reset(
        self,
        task_name: str = "project_easy",
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        if seed is not None:
            random.seed(seed)

        self._episode_id = episode_id or str(uuid.uuid4())
        self._task_name = (
            task_name if task_name in VALID_PROJECT_TASK_NAMES else "project_easy"
        )

        self._current_day = 1
        self._current_sprint = 1
        self._step_count = 0
        self._done = False
        self._cumulative_reward = 0.0
        self._sprint_rewards = []
        self._events_log = []

        self._released_instructions = []
        self._followed_instructions = []
        self._tech_debt = []

        self._tasks, self._developers, self._all_instructions, self._absences = (
            _build_project_scenario(self._task_name)
        )

        # Release any day-1 instructions
        self._release_instructions()

        obs = _make_project_observation(
            current_day=self._current_day,
            current_sprint=self._current_sprint,
            episode_id=self._episode_id,
            task_name=self._task_name,
            tasks=self._tasks,
            developers=self._developers,
            instruction_queue=self._released_instructions,
            instruction_following_score=self._instruction_following_score(),
            tech_debt=self._tech_debt,
            reward=0.0,
            cumulative_reward=0.0,
            sprint_rewards=self._sprint_rewards,
            done=False,
            step_count=0,
            events=["Project started! Sprint 1 of 6 begins."],
        )
        return obs

    # ── Step ─────────────────────────────────────────────────────────────────

    def step(
        self, action: SprintAction
    ) -> Tuple[Dict[str, Any], float, bool, Dict]:
        if self._done:
            obs = _make_project_observation(
                current_day=self._current_day,
                current_sprint=self._current_sprint,
                episode_id=self._episode_id,
                task_name=self._task_name,
                tasks=self._tasks,
                developers=self._developers,
                instruction_queue=self._released_instructions,
                instruction_following_score=self._instruction_following_score(),
                tech_debt=self._tech_debt,
                reward=0.0,
                cumulative_reward=self._cumulative_reward,
                sprint_rewards=self._sprint_rewards,
                done=True,
                step_count=self._step_count,
                events=["Episode already done."],
            )
            return obs, 0.0, True, {}

        self._step_count += 1
        events: List[str] = []
        step_reward = 0.0

        # 1. Apply developer absences for today
        self._apply_absences(events)

        # 2. Apply agent action
        action_reward, action_events = self._apply_action(action)
        step_reward += action_reward
        events.extend(action_events)

        # 3. Check if action follows an active instruction → bonus
        inst_reward, inst_events = self._check_instruction_follow(action)
        step_reward += inst_reward
        events.extend(inst_events)

        # 4. Simulate one day of work
        day_reward, day_events = self._simulate_day()
        step_reward += day_reward
        events.extend(day_events)

        # 5. Advance day counter
        self._current_day += 1

        # 6. Release new instructions for today
        self._release_instructions()

        # 7. Sprint boundary: every DAYS_PER_SPRINT days
        if (self._current_day - 1) % DAYS_PER_SPRINT == 0 and self._current_day > 1:
            sprint_bonus, sprint_events = self._end_of_sprint_review()
            step_reward += sprint_bonus
            events.extend(sprint_events)
            if self._current_sprint < NUM_SPRINTS:
                self._current_sprint += 1
                events.append(
                    f"🏃 Sprint {self._current_sprint} of {NUM_SPRINTS} begins."
                )

        # 8. Check project termination
        if self._current_day > TOTAL_DAYS:
            self._done = True
            final_reward, final_events = self._end_of_project_score()
            step_reward += final_reward
            events.extend(final_events)

        self._cumulative_reward += step_reward
        self._events_log.extend(events)

        obs = _make_project_observation(
            current_day=self._current_day,
            current_sprint=self._current_sprint,
            episode_id=self._episode_id,
            task_name=self._task_name,
            tasks=self._tasks,
            developers=self._developers,
            instruction_queue=self._released_instructions,
            instruction_following_score=self._instruction_following_score(),
            tech_debt=self._tech_debt,
            reward=step_reward,
            cumulative_reward=self._cumulative_reward,
            sprint_rewards=self._sprint_rewards,
            done=self._done,
            step_count=self._step_count,
            events=events,
        )

        info: Dict[str, Any] = {
            "current_sprint": self._current_sprint,
            "tech_debt_count": len(self._tech_debt),
            "instruction_following_score": self._instruction_following_score(),
        }
        return obs, step_reward, self._done, info

    # ── State property ────────────────────────────────────────────────────────

    @property
    def state(self) -> Dict[str, Any]:
        return {
            "episode_id": self._episode_id,
            "task_name": self._task_name,
            "current_day": self._current_day,
            "current_sprint": self._current_sprint,
            "total_days": TOTAL_DAYS,
            "sprint_length": DAYS_PER_SPRINT,
            "step_count": self._step_count,
            "tasks": [t.to_dict() for t in self._tasks],
            "developers": [d.to_dict() for d in self._developers],
            "released_instructions": self._released_instructions,
            "followed_instructions": self._followed_instructions,
            "instruction_following_score": round(
                self._instruction_following_score(), 4
            ),
            "tech_debt": self._tech_debt,
            "sprint_rewards": self._sprint_rewards,
            "cumulative_reward": round(self._cumulative_reward, 4),
            "done": self._done,
            "events_log": self._events_log[-20:],
        }

    # ── Instruction helpers ───────────────────────────────────────────────────

    def _release_instructions(self) -> None:
        """Move instructions whose release_day <= current_day into the queue."""
        already_released = {i["id"] for i in self._released_instructions}
        for inst in self._all_instructions:
            if inst["id"] not in already_released and inst["release_day"] <= self._current_day:
                self._released_instructions.append(inst)

    def _instruction_following_score(self) -> float:
        """Fraction of released instructions that have been followed."""
        if not self._released_instructions:
            return 1.0
        score = len(self._followed_instructions) / len(self._released_instructions)
        return max(0.01, min(0.99, score))

    def _check_instruction_follow(
        self, action: SprintAction
    ) -> Tuple[float, List[str]]:
        """
        If the action's task_id appears in an active instruction's affects_tasks
        and hasn't been credited yet, mark it followed and grant a bonus.
        """
        reward = 0.0
        events = []
        task_id = action.task_id or ""
        atype = (action.action_type or "").lower()

        if atype == "skip" or not task_id:
            return reward, events

        for inst in self._released_instructions:
            if (
                inst["id"] not in self._followed_instructions
                and task_id in inst.get("affects_tasks", [])
            ):
                self._followed_instructions.append(inst["id"])
                reward += 0.4
                events.append(
                    f"📋 Instruction {inst['id']} followed: '{inst['text'][:60]}…'"
                )
        return reward, events

    # ── Sprint boundary ───────────────────────────────────────────────────────

    def _end_of_sprint_review(self) -> Tuple[float, List[str]]:
        """
        Called at the end of each sprint (days 10, 20, 30, 40, 50).
        - Scores delivery rate for the sprint's tasks.
        - Converts unfinished sprint tasks to tech_debt for the next sprint.
        - Returns a sprint-level bonus reward.
        """
        reward = 0.0
        events = []
        sprint = self._current_sprint

        sprint_tasks = [
            t for t in self._tasks
            if t.metadata.get("sprint") == sprint
        ]

        done = [t for t in sprint_tasks if t.status == TaskStatus.DONE]
        missed = [
            t for t in sprint_tasks
            if t.status in (TaskStatus.MISSED, TaskStatus.BACKLOG, TaskStatus.IN_PROGRESS)
        ]

        delivery_rate = len(done) / len(sprint_tasks) if sprint_tasks else 1.0

        # Sprint bonus: 0-2.0 scaled by delivery, instruction following, team health
        inst_score = self._instruction_following_score()
        team_health = self._team_health_score()
        sprint_score = delivery_rate * 0.5 + inst_score * 0.3 + team_health * 0.2
        sprint_bonus = max(0.01, min(0.99, sprint_score)) * 2.0
        reward += sprint_bonus
        self._sprint_rewards.append(round(sprint_bonus, 4))

        events.append(
            f"📊 Sprint {sprint} review: {len(done)}/{len(sprint_tasks)} tasks done "
            f"(delivery={delivery_rate:.0%}, inst={inst_score:.2f}, "
            f"health={team_health:.2f}) → bonus {sprint_bonus:.2f}"
        )

        # Carry unfinished tasks as tech debt into the next sprint
        for t in missed:
            if t.id not in self._tech_debt:
                self._tech_debt.append(t.id)
                # Mark still-in-progress tasks as missed for scoring
                if t.status == TaskStatus.IN_PROGRESS:
                    t.status = TaskStatus.MISSED
                    dev = self._find_dev(t.assigned_to)
                    if dev:
                        if t.id in dev.assigned_tasks:
                            dev.assigned_tasks.remove(t.id)
                        dev.current_load = max(0, dev.current_load - t.effort)
                events.append(
                    f"🔴 Tech debt: {t.id} ({t.name}) carried to sprint {sprint + 1}"
                )
                # Apply tech-debt productivity drag to all devs
                for dev in self._developers:
                    dev.productivity = max(
                        0.5, dev.productivity - TECH_DEBT_PENALTY
                    )

        return reward, events

    # ── Project final score ───────────────────────────────────────────────────

    def _end_of_project_score(self) -> Tuple[float, List[str]]:
        """
        Final project reward at day 60.
        Score = delivery_rate × instruction_following × team_health
        Clamped to (0.01, 0.99), scaled to a 5.0 point bonus.
        """
        events = []
        total = len(self._tasks)
        done = sum(1 for t in self._tasks if t.status == TaskStatus.DONE)

        delivery_rate = done / total if total else 0.0
        inst_score = self._instruction_following_score()
        team_health = self._team_health_score()

        raw_score = delivery_rate * inst_score * team_health
        final_score = max(0.01, min(0.99, raw_score))
        final_bonus = final_score * 5.0

        events.append(
            f"🏁 Project complete! {done}/{total} tasks delivered. "
            f"delivery={delivery_rate:.0%} inst={inst_score:.2f} "
            f"health={team_health:.2f} → final_score={final_score:.3f} "
            f"bonus={final_bonus:.2f}"
        )
        return final_bonus, events

    # ── Team health ──────────────────────────────────────────────────────────

    def _team_health_score(self) -> float:
        """
        Simple team health: fraction of devs with current_load <= capacity.
        Degraded by tech debt count.
        """
        if not self._developers:
            return 1.0
        healthy = sum(
            1 for d in self._developers if d.current_load <= d.capacity
        )
        base = healthy / len(self._developers)
        debt_drag = min(0.5, len(self._tech_debt) * 0.02)
        return max(0.01, base - debt_drag)

    # ── Absences ─────────────────────────────────────────────────────────────

    def _apply_absences(self, events: List[str]) -> None:
        """Apply scheduled absences from project_data.json."""
        day = self._current_day
        for absence in self._absences:
            dev = self._find_dev(absence["dev_id"])
            if dev is None:
                continue
            if absence["day_start"] <= day <= absence["day_end"]:
                if dev.is_available:
                    dev.is_available = False
                    events.append(
                        f"🏖️  {dev.name} absent today ({absence['reason']})."
                    )
            else:
                # Restore availability once absence window ends
                if not dev.is_available:
                    dev.is_available = True
                    events.append(f"💪 {dev.name} is back from {absence['reason']}.")

    # ── Action handler (mirrors R1 _apply_action exactly) ────────────────────

    def _apply_action(self, action: SprintAction) -> Tuple[float, List[str]]:
        reward = 0.0
        events = []

        atype = (action.action_type or "skip").lower()
        task = self._find_task(action.task_id)
        dev = self._find_dev(action.dev_id)

        if atype == "assign":
            if task is None:
                events.append(f"Invalid assign: task {action.task_id} not found.")
                reward -= 0.2
            elif dev is None:
                events.append(f"Invalid assign: dev {action.dev_id} not found.")
                reward -= 0.2
            elif task.status != TaskStatus.BACKLOG:
                events.append(
                    f"Task {task.id} not in backlog (status={task.status.value})."
                )
                reward -= 0.1
            elif not dev.can_take_task(task):
                events.append(
                    f"Dev {dev.id} can't take task {task.id} (capacity/skill)."
                )
                reward -= 0.15
            else:
                # Dependency check: don't assign if dependencies incomplete
                unmet = self._unmet_dependencies(task)
                if unmet:
                    events.append(
                        f"⛔ {task.id} blocked: dependencies not done {unmet}."
                    )
                    task.status = TaskStatus.BLOCKED
                    reward -= 0.1
                else:
                    task.status = TaskStatus.IN_PROGRESS
                    task.assigned_to = dev.id
                    dev.assigned_tasks.append(task.id)
                    dev.current_load += task.effort
                    skill_bonus = 0.3 if dev.skill == task.required_skill else 0.1
                    priority_bonus = (6 - task.priority) * 0.1
                    reward += 0.5 + skill_bonus + priority_bonus
                    events.append(f"Assigned {task.id} ({task.name}) → {dev.name}")

        elif atype == "reassign":
            if task is None or dev is None:
                events.append("Invalid reassign: task or dev not found.")
                reward -= 0.2
            elif task.status not in (TaskStatus.IN_PROGRESS, TaskStatus.BLOCKED):
                events.append(
                    f"Cannot reassign {task.id} (status={task.status.value})."
                )
                reward -= 0.1
            else:
                old_dev = self._find_dev(task.assigned_to)
                if old_dev:
                    if task.id in old_dev.assigned_tasks:
                        old_dev.assigned_tasks.remove(task.id)
                    old_dev.current_load = max(0, old_dev.current_load - task.effort)
                task.assigned_to = dev.id
                task.status = TaskStatus.IN_PROGRESS
                dev.assigned_tasks.append(task.id)
                dev.current_load += task.effort
                reward += 0.2
                events.append(f"Reassigned {task.id} → {dev.name}")

        elif atype == "reprioritize":
            if task is None:
                events.append(f"Task {action.task_id} not found.")
                reward -= 0.1
            elif action.new_priority not in range(1, 6):
                events.append(f"Invalid priority {action.new_priority}.")
                reward -= 0.05
            else:
                old_p = task.priority
                task.priority = action.new_priority
                reward += 0.1
                events.append(
                    f"Reprioritized {task.id}: {old_p} → {action.new_priority}"
                )

        elif atype == "unblock":
            if task is None:
                events.append(f"Task {action.task_id} not found.")
                reward -= 0.1
            elif task.status != TaskStatus.BLOCKED:
                events.append(f"Task {task.id} is not blocked.")
            else:
                unmet = self._unmet_dependencies(task)
                if unmet:
                    events.append(
                        f"Still can't unblock {task.id}: {unmet} not done."
                    )
                    reward -= 0.05
                else:
                    task.status = TaskStatus.IN_PROGRESS
                    reward += 0.3
                    events.append(f"Unblocked task {task.id}")

        elif atype == "skip":
            reward -= 0.05
            events.append("Agent chose to skip (no action).")

        else:
            events.append(f"Unknown action type: {atype}")
            reward -= 0.2

        return reward, events

    # ── Day simulation (mirrors R1 _simulate_day) ─────────────────────────────

    def _simulate_day(self) -> Tuple[float, List[str]]:
        reward = 0.0
        events = []

        for task in self._tasks:
            if task.status != TaskStatus.IN_PROGRESS:
                continue
            dev = self._find_dev(task.assigned_to)
            if dev is None or not dev.is_available:
                continue

            daily_progress = (dev.productivity * dev.capacity) / (
                task.effort * TOTAL_DAYS / 2
            )
            daily_progress = min(daily_progress, 0.5)
            task.progress = min(1.0, task.progress + daily_progress)
            task.days_in_progress += 1

            if task.progress >= 1.0:
                task.status = TaskStatus.DONE
                task.progress = 1.0
                on_time = self._current_day <= task.deadline
                if on_time:
                    reward += (6 - task.priority) * 0.5 + 0.5
                    events.append(f"✅ {task.name} completed on time!")
                else:
                    reward += 0.1
                    events.append(f"⚠️  {task.name} completed LATE.")
                dev.assigned_tasks = [
                    tid for tid in dev.assigned_tasks if tid != task.id
                ]
                dev.current_load = max(0, dev.current_load - task.effort)

            elif self._current_day > task.deadline and task.status == TaskStatus.IN_PROGRESS:
                task.status = TaskStatus.MISSED
                reward -= (6 - task.priority) * 0.4
                events.append(f"❌ {task.name} MISSED deadline (day {task.deadline})!")
                dev_obj = self._find_dev(task.assigned_to)
                if dev_obj:
                    if task.id in dev_obj.assigned_tasks:
                        dev_obj.assigned_tasks.remove(task.id)
                    dev_obj.current_load = max(0, dev_obj.current_load - task.effort)

        # Backlog tasks past absolute deadline expire
        for task in self._tasks:
            if task.status == TaskStatus.BACKLOG and self._current_day > task.deadline:
                task.status = TaskStatus.MISSED
                reward -= (6 - task.priority) * 0.3
                events.append(f"❌ {task.name} expired in backlog!")

        return reward, events

    # ── Dependency check ──────────────────────────────────────────────────────

    def _unmet_dependencies(self, task: Task) -> List[str]:
        """Return list of dependency task IDs that are not yet DONE."""
        deps = task.metadata.get("depends_on", [])
        unmet = []
        for dep_id in deps:
            dep_task = self._find_task(dep_id)
            if dep_task is None or dep_task.status != TaskStatus.DONE:
                unmet.append(dep_id)
        return unmet

    # ── Finders ───────────────────────────────────────────────────────────────

    def _find_task(self, task_id: Optional[str]) -> Optional[Task]:
        if not task_id:
            return None
        return next((t for t in self._tasks if t.id == task_id), None)

    def _find_dev(self, dev_id: Optional[str]) -> Optional[Developer]:
        if not dev_id:
            return None
        return next((d for d in self._developers if d.id == dev_id), None)