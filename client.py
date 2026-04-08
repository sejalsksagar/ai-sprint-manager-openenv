"""
Sprint Manager — OpenEnv Client

This is what RL researchers import in their training code.
It provides a clean typed interface to the environment server.

Usage:
    import requests
    from client import SprintEnvClient, SprintAction

    client = SprintEnvClient(base_url="https://sejal-k-ai-sprint-manager.hf.space")
    obs = client.reset(task_name="easy_sprint")
    result = client.step(SprintAction(action_type="assign", task_id="T1", dev_id="dev1"))
    state = client.state()
    client.close()

    # Or as context manager:
    with SprintEnvClient(base_url="http://localhost:7860") as client:
        obs = client.reset(task_name="medium_sprint", seed=42)
        while not obs["done"]:
            result = client.step(SprintAction(action_type="skip"))
            obs = result["observation"]
"""
from __future__ import annotations
import requests
from typing import Optional, Any
from sprint_env.models import SprintAction


class StepResult:
    """Typed result from a step() call."""
    def __init__(self, payload: dict):
        self.observation: dict = payload["observation"]
        self.reward: float = payload["reward"]
        self.done: bool = payload["done"]
        self.info: dict = payload.get("info", {})

    def __repr__(self):
        return (
            f"StepResult(reward={self.reward:+.2f}, done={self.done}, "
            f"day={self.observation.get('current_day')}, "
            f"completed={self.observation.get('tasks_completed')})"
        )


class SprintEnvClient:
    """
    HTTP client for the Sprint Manager OpenEnv environment.

    Wraps the REST API into a clean Python interface.
    Use this in RL training loops, notebooks, or evaluation scripts.
    """

    def __init__(self, base_url: str = "http://localhost:7860", timeout: int = 30):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._session = requests.Session()

    def reset(
        self,
        task_name: str = "easy_sprint",
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
    ) -> dict:
        """
        Reset the environment and return initial observation.

        Args:
            task_name: One of "easy_sprint", "medium_sprint", "hard_sprint"
            seed: Random seed for reproducibility
            episode_id: Optional episode identifier

        Returns:
            Observation dict
        """
        payload = {"task_name": task_name}
        if seed is not None:
            payload["seed"] = seed
        if episode_id is not None:
            payload["episode_id"] = episode_id

        resp = self._session.post(
            f"{self.base_url}/reset", json=payload, timeout=self.timeout
        )
        resp.raise_for_status()
        return resp.json()

    def step(self, action: SprintAction) -> StepResult:
        """
        Take one action and advance the sprint by one day.

        Args:
            action: SprintAction with action_type, task_id, dev_id, new_priority

        Returns:
            StepResult with observation, reward, done, info
        """
        payload = {"action": action.model_dump()}
        resp = self._session.post(
            f"{self.base_url}/step", json=payload, timeout=self.timeout
        )
        resp.raise_for_status()
        return StepResult(resp.json())

    def state(self) -> dict:
        """Return the full current environment state."""
        resp = self._session.get(f"{self.base_url}/state", timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    def health(self) -> dict:
        """Check server health."""
        resp = self._session.get(f"{self.base_url}/health", timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    def list_tasks(self) -> list[dict]:
        """List all available sprint scenarios."""
        resp = self._session.get(f"{self.base_url}/tasks", timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()["tasks"]

    def close(self):
        """Close the HTTP session."""
        self._session.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def __repr__(self):
        return f"SprintEnvClient(base_url='{self.base_url}')"