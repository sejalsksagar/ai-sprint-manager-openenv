"""
FastAPI server implementing the OpenEnv HTTP API for Sprint Manager.

Endpoints:
  POST /reset   — start new episode
  POST /step    — take an action
  GET  /state   — get current state
  GET  /health  — health check
  GET  /tasks   — list available tasks
"""
from __future__ import annotations
import os
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from sprint_env.environment import SprintManagerEnv
from sprint_env.models import SprintAction, SprintObservation, SprintState

app = FastAPI(
    title="AI Sprint Manager — OpenEnv",
    description="RL environment for agile sprint management and task allocation.",
    version="1.0.0",
)

# Single global env instance (stateful per container, as per OpenEnv pattern)
env = SprintManagerEnv()


# ── Request/Response schemas ──────────────────────────────────────────────────

class ResetRequest(BaseModel):
    task_name: Optional[str] = "easy_sprint"
    seed: Optional[int] = None
    episode_id: Optional[str] = None


class StepRequest(BaseModel):
    action: SprintAction


class StepResponse(BaseModel):
    observation: SprintObservation
    reward: float
    done: bool
    info: Dict[str, Any]


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.post("/reset", response_model=SprintObservation)
def reset(req: ResetRequest = None):
    """Reset the environment and return the initial observation."""
    if req is None:
        req = ResetRequest()
    obs = env.reset(
        task_name=req.task_name or "easy_sprint",
        seed=req.seed,
        episode_id=req.episode_id,
    )
    return obs


@app.post("/step", response_model=StepResponse)
def step(req: StepRequest):
    """Take one action and advance the environment by one day."""
    obs, reward, done, info = env.step(req.action)
    return StepResponse(observation=obs, reward=reward, done=done, info=info)


@app.get("/state", response_model=SprintState)
def state():
    """Return the full current environment state."""
    return env.state


@app.get("/health")
def health():
    """Health check endpoint."""
    return {"status": "ok", "env": "ai-sprint-manager"}


@app.get("/tasks")
def list_tasks():
    """List all available task scenarios."""
    return {
        "tasks": [
            {
                "id": "easy_sprint",
                "name": "Easy Sprint: Small Team, Clear Tasks",
                "difficulty": "easy",
                "description": "3 developers, 5 tasks, no random events. Good for baseline testing.",
            },
            {
                "id": "medium_sprint",
                "name": "Medium Sprint: Bugs & Delays",
                "difficulty": "medium",
                "description": "4 developers, 8 tasks including bugs. Random dev absences may occur.",
            },
            {
                "id": "hard_sprint",
                "name": "Hard Sprint: Cascading Failures",
                "difficulty": "hard",
                "description": "5 developers, 12 tasks. Urgent bugs appear mid-sprint. Tests frontier model capability.",
            },
        ]
    }


@app.get("/")
def root():
    return {
        "name": "AI Sprint Manager OpenEnv",
        "version": "1.0.0",
        "endpoints": ["/reset", "/step", "/state", "/tasks", "/health"],
    }