"""
Round 2 FastAPI server — /project/* endpoints for the multi-sprint environment.

Runs ALONGSIDE the existing R1 server/app.py (not replacing it).
All routes are prefixed /project/ to avoid any conflict with R1's
/reset  /step  /state  /health  /tasks.

Endpoints
---------
POST /project/reset   — start a new 60-day episode
POST /project/step    — advance one day / take one action
GET  /project/state   — full internal state snapshot
GET  /project/health  — liveness check (judges hit this)
GET  /project/tasks   — list available R2 scenarios

Integration with R1 ui.py
--------------------------
Import `project_router` and mount it on the shared FastAPI app:

    from server.project_app import project_router
    app.include_router(project_router)

Or run this file standalone (port 8001) for isolated testing:

    python -m server.project_app
"""

from __future__ import annotations

import traceback
from typing import Any, Dict, Optional

import uvicorn
from fastapi import APIRouter, FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from sprint_env.project_environment import ProjectManagerEnv, VALID_PROJECT_TASK_NAMES
from sprint_env.project_models import ProjectAction

# ─── Module-level environment instance ───────────────────────────────────────
# One shared instance per process — same pattern as R1 server/app.py.
# The /project/reset endpoint reinitialises it for each new episode.

_env = ProjectManagerEnv()

# ─── Request / response schemas ──────────────────────────────────────────────


class ResetRequest(BaseModel):
    task_name: str = Field(
        default="project_easy",
        description="Scenario name: project_easy | project_medium | project_hard",
    )
    seed: Optional[int] = Field(default=None, description="Random seed for reproducibility")
    episode_id: Optional[str] = Field(default=None, description="Optional custom episode ID")


class StepRequest(BaseModel):
    action: Dict[str, Any] = Field(
        default_factory=lambda: {"action_type": "skip"},
        description=(
            "ProjectAction dict. Keys: action_type, task_id, dev_id, "
            "new_priority, task_ids, notes"
        ),
    )


# ─── Router (imported by ui.py / R1 app) ─────────────────────────────────────

project_router = APIRouter(prefix="/project", tags=["Round 2 — Project Manager"])


@project_router.post("/reset")
def project_reset(req: ResetRequest) -> Dict[str, Any]:
    """
    Start a new 60-day multi-sprint episode.

    Returns the initial ProjectObservation dict.
    """
    task_name = req.task_name
    if task_name not in VALID_PROJECT_TASK_NAMES:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Unknown task_name '{task_name}'. "
                f"Must be one of: {VALID_PROJECT_TASK_NAMES}"
            ),
        )
    try:
        obs = _env.reset(
            task_name=task_name,
            seed=req.seed,
            episode_id=req.episode_id,
        )
        return obs
    except Exception as exc:
        raise HTTPException(status_code=500, detail=_fmt_error(exc))


@project_router.post("/step")
def project_step(req: StepRequest) -> Dict[str, Any]:
    """
    Advance the environment by one day.

    The agent submits one action; the environment simulates the day,
    releases any new instructions, checks sprint boundaries, and returns
    the next observation together with the step reward and done flag.

    Returns:
        observation  — full ProjectObservation dict
        reward       — float, step reward (includes any sprint/project bonuses)
        done         — bool, True once day 60 is complete
        info         — dict with current_sprint, tech_debt_count, inst_score
    """
    if _env._done:
        # Soft guard: return terminal state instead of raising
        return {
            "observation": _env.state,
            "reward": 0.0,
            "done": True,
            "info": {"message": "Episode already done. Call /project/reset to start a new one."},
        }

    try:
        action = _parse_action(req.action)
    except (ValueError, TypeError) as exc:
        raise HTTPException(status_code=422, detail=f"Invalid action: {exc}")

    try:
        obs, reward, done, info = _env.step(action)
        return {
            "observation": obs,
            "reward": round(reward, 4),
            "done": done,
            "info": info,
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=_fmt_error(exc))


@project_router.get("/state")
def project_state() -> Dict[str, Any]:
    """
    Full internal state snapshot.

    Returns ProjectState dict including tasks, developers, released
    instructions, tech debt list, sprint rewards, and all counters.
    Useful for debugging and UI polling.
    """
    try:
        return _env.state
    except Exception as exc:
        raise HTTPException(status_code=500, detail=_fmt_error(exc))


@project_router.get("/health")
def project_health() -> Dict[str, Any]:
    """
    Liveness check — judges verify this returns 200.

    Returns environment status and a summary of the current episode.
    """
    return {
        "status": "ok",
        "env": "ai-sprint-manager-r2",
        "round": 2,
        "episode": {
            "id":             _env._episode_id or "not_started",
            "task_name":      _env._task_name,
            "current_day":    _env._current_day,
            "current_sprint": _env._current_sprint,
            "done":           _env._done,
            "cumulative_reward": round(_env._cumulative_reward, 4),
        },
        "valid_scenarios": VALID_PROJECT_TASK_NAMES,
    }


@project_router.get("/tasks")
def project_tasks() -> Dict[str, Any]:
    """List the available R2 multi-sprint scenarios."""
    return {
        "tasks": [
            {
                "id":          "project_easy",
                "difficulty":  "easy",
                "description": "Small stable team, clear backlog, no cascade failures.",
                "num_sprints": 6,
                "total_days":  60,
                "num_tasks":   24,
                "num_instructions": 12,
            },
            {
                "id":          "project_medium",
                "difficulty":  "medium",
                "description": "Mid-size team, dev absences, urgent bugs, conflicting instructions.",
                "num_sprints": 6,
                "total_days":  60,
                "num_tasks":   30,
                "num_instructions": 18,
            },
            {
                "id":          "project_hard",
                "difficulty":  "hard",
                "description": "Large team, cascading failures, scope pivots, hard deadline.",
                "num_sprints": 6,
                "total_days":  60,
                "num_tasks":   37,
                "num_instructions": 25,
            },
        ]
    }


# ─── Helpers ─────────────────────────────────────────────────────────────────

_VALID_ACTIONS = {"assign", "reassign", "reprioritize", "skip", "unblock", "sprint_plan"}


def _parse_action(action_dict: Dict[str, Any]) -> ProjectAction:
    """
    Sanitise and validate an action dict from the LLM into a ProjectAction.

    Handles all common LLM output mistakes to eliminate 422 errors:
    - Unknown action_type → replaced with "skip"
    - "null" string → None
    - sprint_plan without task_ids → replaced with "skip"
    - assign/reassign without task_id or dev_id → replaced with "skip"
    - Extra unknown keys are ignored by Pydantic
    """
    # Deep copy so we don't mutate caller's dict
    d: Dict[str, Any] = dict(action_dict)

    # 1. Normalise action_type
    raw_type = str(d.get("action_type", "skip")).lower().strip()
    if raw_type not in _VALID_ACTIONS:
        raw_type = "skip"
    d["action_type"] = raw_type

    # 2. Convert "null" strings → None (Llama often outputs "null" as a string)
    for key in ("task_id", "dev_id", "new_priority", "task_ids", "notes"):
        if d.get(key) in ("null", "none", "None", "Null", "", "undefined"):
            d[key] = None

    # 3. Convert new_priority to int if it's a numeric string
    if d.get("new_priority") is not None:
        try:
            d["new_priority"] = int(d["new_priority"])
            if d["new_priority"] not in range(1, 6):
                d["new_priority"] = None
        except (ValueError, TypeError):
            d["new_priority"] = None

    # 4. Safety: demote invalid action types to skip before cross-field validation
    atype = d["action_type"]
    if atype == "assign" and (not d.get("task_id") or not d.get("dev_id")):
        d["action_type"] = "skip"
    if atype == "reassign" and (not d.get("task_id") or not d.get("dev_id")):
        d["action_type"] = "skip"
    if atype == "reprioritize" and (not d.get("task_id") or not d.get("new_priority")):
        d["action_type"] = "skip"
    if atype == "unblock" and not d.get("task_id"):
        d["action_type"] = "skip"
    if atype == "sprint_plan" and not d.get("task_ids"):
        d["action_type"] = "skip"  # sprint_plan without task_ids is a no-op anyway

    return ProjectAction(**d)


def _fmt_error(exc: Exception) -> str:
    """Format exception for HTTPException detail — include type and message."""
    return f"{type(exc).__name__}: {exc}\n{traceback.format_exc(limit=3)}"


# ─── Standalone app (for isolated testing on port 8001) ──────────────────────
# This does NOT replace the R1 app. Import project_router into ui.py instead.

_standalone_app = FastAPI(
    title="AI Sprint Manager — Round 2 Project Endpoints",
    version="2.0.0",
    description=(
        "Standalone R2 server for isolated testing. "
        "In production, project_router is mounted on the shared R1 app in ui.py."
    ),
)
_standalone_app.include_router(project_router)


@_standalone_app.get("/health")
def standalone_health() -> Dict[str, Any]:
    """Root health — confirms both R1-style and R2 endpoints are live."""
    return {"status": "ok", "mode": "standalone-r2", "r2_prefix": "/project"}


def main() -> None:
    """Entry point for isolated local testing only."""
    uvicorn.run(_standalone_app, host="0.0.0.0", port=8001)


if __name__ == "__main__":
    main()