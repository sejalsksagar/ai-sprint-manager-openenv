"""
Project Manager — OpenEnv Client (Round 2)

Typed HTTP client for the multi-sprint /project/* endpoints.
Mirrors the R1 SprintEnvClient interface exactly, extended for R2.

Usage:
    from project_client import ProjectEnvClient

    client = ProjectEnvClient(base_url="https://sejal-k-ai-sprint-manager.hf.space")
    obs = client.reset(task_name="project_easy", seed=42)

    while not obs["done"]:
        result = client.step({"action_type": "assign", "task_id": "T01", "dev_id": "dev1"})
        obs = result.observation

    client.close()

    # Or as context manager:
    with ProjectEnvClient() as client:
        obs = client.reset("project_hard")
        while not obs["done"]:
            result = client.step({"action_type": "skip"})
            obs = result.observation
"""

from __future__ import annotations

import requests
from typing import Any, Optional


# ── Typed step result ──────────────────────────────────────────────────────────

class ProjectStepResult:
    """
    Typed result from a project step() call.

    Extends R1 StepResult with R2-specific fields surfaced as properties.
    """

    def __init__(self, payload: dict) -> None:
        self.observation: dict  = payload["observation"]
        self.reward: float      = payload["reward"]
        self.done: bool         = payload["done"]
        self.info: dict         = payload.get("info", {})

    # ── R2 convenience accessors ──────────────────────────────────────────────

    @property
    def current_sprint(self) -> int:
        return self.observation.get("current_sprint", 1)

    @property
    def current_day(self) -> int:
        return self.observation.get("current_day", 1)

    @property
    def instruction_queue(self) -> list[dict]:
        """All instructions released up to current_day."""
        return self.observation.get("instruction_queue", [])

    @property
    def active_instructions(self) -> list[dict]:
        """Released instructions not yet followed."""
        return [i for i in self.instruction_queue if not i.get("followed", False)]

    @property
    def tech_debt(self) -> list[str]:
        """Task IDs that became tech debt at sprint boundaries."""
        return self.observation.get("tech_debt", [])

    @property
    def sprint_rewards(self) -> list[float]:
        """Per-sprint reward history."""
        return self.observation.get("sprint_rewards", [])

    @property
    def instruction_following_score(self) -> float:
        return self.observation.get("instruction_following_score", 1.0)

    @property
    def tasks_completed(self) -> int:
        return self.observation.get("tasks_completed", 0)

    @property
    def tasks_missed(self) -> int:
        return self.observation.get("tasks_missed", 0)

    @property
    def cumulative_reward(self) -> float:
        return self.observation.get("cumulative_reward", 0.0)

    def __repr__(self) -> str:
        return (
            f"ProjectStepResult("
            f"reward={self.reward:+.3f}, done={self.done}, "
            f"day={self.current_day}/60, sprint={self.current_sprint}/6, "
            f"completed={self.tasks_completed}, "
            f"inst_score={self.instruction_following_score:.2f}, "
            f"debt={len(self.tech_debt)})"
        )


# ── Client ─────────────────────────────────────────────────────────────────────

class ProjectEnvClient:
    """
    HTTP client for the R2 multi-sprint Project Manager environment.

    Wraps the /project/* REST API into a clean typed Python interface.
    Use this in RL training loops (train_llm.py), evaluation scripts
    (evaluate_r2.py), and notebooks.

    All endpoints mirror the R1 SprintEnvClient interface so training
    code can swap between R1 and R2 with minimal changes.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:7860",
        timeout: int  = 60,          # 60s — R2 steps can be slower than R1
    ) -> None:
        self.base_url  = base_url.rstrip("/")
        self.timeout   = timeout
        self._session  = requests.Session()
        self._prefix   = "/project"

    # ── Core API ──────────────────────────────────────────────────────────────

    def reset(
        self,
        task_name:  str           = "project_easy",
        seed:       Optional[int] = None,
        episode_id: Optional[str] = None,
    ) -> dict:
        """
        Start a new 60-day multi-sprint episode.

        Args:
            task_name:  "project_easy" | "project_medium" | "project_hard"
            seed:       Random seed for reproducibility
            episode_id: Optional custom episode identifier

        Returns:
            Initial observation dict (includes current_sprint, instruction_queue,
            tech_debt, sprint_rewards).
        """
        payload: dict[str, Any] = {"task_name": task_name}
        if seed is not None:
            payload["seed"] = seed
        if episode_id is not None:
            payload["episode_id"] = episode_id

        resp = self._session.post(
            f"{self.base_url}{self._prefix}/reset",
            json=payload,
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return resp.json()

    def step(self, action: dict[str, Any]) -> ProjectStepResult:
        """
        Take one action and advance the project by one day.

        Args:
            action: dict with keys: action_type, task_id, dev_id,
                    new_priority, task_ids (for sprint_plan), notes.
                    Minimum required: {"action_type": "skip"}

        Returns:
            ProjectStepResult with observation, reward, done, info,
            plus R2 convenience properties.
        """
        payload = {"action": action}
        resp = self._session.post(
            f"{self.base_url}{self._prefix}/step",
            json=payload,
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return ProjectStepResult(resp.json())

    def state(self) -> dict:
        """Return the full current internal state snapshot."""
        resp = self._session.get(
            f"{self.base_url}{self._prefix}/state",
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return resp.json()

    def health(self) -> dict:
        """
        Check R2 server health.
        Returns round=2 and current episode summary.
        """
        resp = self._session.get(
            f"{self.base_url}{self._prefix}/health",
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return resp.json()

    def list_tasks(self) -> list[dict]:
        """List all available R2 multi-sprint scenarios."""
        resp = self._session.get(
            f"{self.base_url}{self._prefix}/tasks",
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return resp.json()["tasks"]

    # ── Helpers ───────────────────────────────────────────────────────────────

    def run_episode(
        self,
        task_name: str,
        policy_fn,                     # Callable[[dict], dict]
        seed: Optional[int] = None,
        max_steps: int = 60,
        verbose: bool = False,
    ) -> dict:
        """
        Convenience: run a full episode with a policy function.

        Args:
            task_name:  Scenario name
            policy_fn:  fn(observation: dict) -> action: dict
            seed:       Optional random seed
            max_steps:  Safety cap (default 60 = full project)
            verbose:    Print step summaries to stdout

        Returns:
            dict with keys: steps, cumulative_reward, final_score,
            tasks_completed, tasks_missed, instruction_following_score,
            tech_debt, sprint_rewards, done
        """
        obs = self.reset(task_name=task_name, seed=seed)
        total_reward = 0.0
        steps = 0

        for step_num in range(1, max_steps + 1):
            if obs.get("done", False):
                break
            action = policy_fn(obs)
            result = self.step(action)
            obs = result.observation
            total_reward += result.reward
            steps += 1

            if verbose:
                print(
                    f"  day={obs['current_day']-1:02d}/60 "
                    f"sprint={obs['current_sprint']}/6 "
                    f"done={obs['tasks_completed']} "
                    f"missed={obs['tasks_missed']} "
                    f"debt={len(obs['tech_debt'])} "
                    f"inst={obs['instruction_following_score']:.2f} "
                    f"reward={result.reward:+.3f}",
                    flush=True,
                )

        return {
            "steps":                      steps,
            "cumulative_reward":          round(total_reward, 4),
            "tasks_completed":            obs.get("tasks_completed", 0),
            "tasks_missed":               obs.get("tasks_missed", 0),
            "instruction_following_score": obs.get("instruction_following_score", 0.0),
            "tech_debt":                  obs.get("tech_debt", []),
            "sprint_rewards":             obs.get("sprint_rewards", []),
            "done":                       obs.get("done", False),
        }

    # ── Session lifecycle ─────────────────────────────────────────────────────

    def close(self) -> None:
        """Close the underlying HTTP session."""
        self._session.close()

    def __enter__(self) -> "ProjectEnvClient":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def __repr__(self) -> str:
        return f"ProjectEnvClient(base_url='{self.base_url}', prefix='{self._prefix}')"