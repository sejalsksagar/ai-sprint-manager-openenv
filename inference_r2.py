"""
Inference Script — AI Sprint Manager Round 2 (Multi-Sprint Project Manager)
============================================================================
Runs LLM-driven episodes against the /project/* endpoints.

MANDATORY ENV VARS:
    API_BASE_URL  : LLM endpoint  (e.g. https://router.huggingface.co/v1)
    MODEL_NAME    : Model identifier
    HF_TOKEN      : HuggingFace / API key
    ENV_BASE_URL  : Running environment server

OUTPUT FORMAT (required by OpenEnv validator):
    [START] printed once at episode start      — flush=True
    [STEP]  printed after every step           — flush=True
    [END]   printed once at episode end        — flush=True

Run locally (no LLM — rule-based fallback):
    python inference_r2.py

Run with LLM:
    HF_TOKEN=hf_... MODEL_NAME=Qwen/Qwen2.5-1.5B-Instruct python inference_r2.py
"""

from __future__ import annotations

import json
import os
import sys
import time

import requests
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
API_KEY      = os.getenv("HF_TOKEN") or os.getenv("API_KEY", "dummy")
MODEL_NAME   = os.getenv("MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct")
ENV_BASE_URL = os.getenv("ENV_BASE_URL", "https://sejal-k-ai-sprint-manager.hf.space")

MAX_STEPS   = 60        # full 60-day project
TEMPERATURE = 0.2
MAX_TOKENS  = 80        # action JSON is tiny — 80 tokens is plenty

TASKS = ["project_easy", "project_medium", "project_hard"]

client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

# ── System prompt ─────────────────────────────────────────────────────────────
# R2 additions over R1:
#   - "ACTIVE INSTRUCTIONS" section — agent must follow these
#   - "TECH DEBT" section — missed tasks drag productivity
#   - sprint context (current sprint, days remaining)
#   - sprint_plan action type for batch planning

R2_SYSTEM_PROMPT = """You are an Engineering Manager. Output ONLY a JSON action each step.

Schema: {"action_type":"<assign|reassign|reprioritize|unblock|skip>","task_id":"<id or null>","dev_id":"<id or null>","new_priority":<1-5 or null>}

Rules:
- Follow ACTIVE INSTRUCTIONS first — assign their tasks immediately
- assign: only if task deps (✓) are satisfied and dev is available (✓)
- unblock: only if task status is "blocked"
- skip: last resort only

Output ONLY the JSON. No explanation."""


# ── Prompt builder ─────────────────────────────────────────────────────────────

def build_user_prompt(obs: dict) -> str:
    """
    Compact per-step prompt for R2. Kept short to avoid HF router token limits.
    Qwen2.5-1.5B context via HF router: ~2048 tokens total (system + user + completion).
    Target: user prompt under 600 tokens.
    """
    current_day    = obs["current_day"]
    current_sprint = obs.get("current_sprint", 1)
    days_left      = max(0, current_sprint * 10 - current_day + 1)

    # Active instructions — most critical, show up to 3
    active_insts = [i for i in obs.get("instruction_queue", []) if not i.get("followed", False)]
    if active_insts:
        inst_lines = " | ".join(
            f"[{i['id']}] {i['text'][:50]}" for i in active_insts[:3]
        )
        inst_section = f"FOLLOW NOW: {inst_lines}"
    else:
        inst_section = "No pending instructions."

    # Tech debt — just count and IDs
    tech_debt = obs.get("tech_debt", [])
    debt_str = f"DEBT({len(tech_debt)}): {','.join(tech_debt[:5])}" if tech_debt else "No debt."

    # Tasks — compact format, top 6 backlog only
    tasks    = obs["tasks"]
    done_ids = {t["id"] for t in tasks if t["status"] == "done"}
    backlog  = sorted(
        [t for t in tasks if t["status"] == "backlog"],
        key=lambda t: (t["priority"], t["deadline"])
    )
    in_prog  = [t for t in tasks if t["status"] == "in_progress"]

    def fmt(t: dict) -> str:
        deps = t.get("metadata", {}).get("depends_on", [])
        ok = "✓" if all(d in done_ids for d in deps) else "✗"
        return f"[{t['id']}]P{t['priority']} {t['required_skill']} {ok} due=D{t['deadline']}"

    backlog_str  = " ".join(fmt(t) for t in backlog[:6])
    if len(backlog) > 6:
        backlog_str += f" +{len(backlog)-6}more"
    inprog_str   = " ".join(f"[{t['id']}]→{t['assigned_to']}" for t in in_prog) or "none"
    missed_str   = ",".join(t["id"] for t in tasks if t["status"] == "missed") or "none"

    # Developers — compact
    devs_str = " | ".join(
        f"[{d['id']}]{d['name']}({d['skill']}) {d['current_load']}/{d['capacity']} {'✓' if d['is_available'] else '✗'}"
        for d in obs["developers"]
    )

    return (
        f"D{current_day}/60 S{current_sprint}/6 ({days_left}d left) "
        f"done={obs['tasks_completed']} missed={obs['tasks_missed']} inst={obs.get('instruction_following_score',1):.2f}\n"
        f"{inst_section}\n"
        f"{debt_str}\n"
        f"BACKLOG(✓=deps_ok): {backlog_str}\n"
        f"IN_PROGRESS: {inprog_str}\n"
        f"MISSED: {missed_str}\n"
        f"DEVS: {devs_str}\n"
        f"Output JSON action:"
    )


# ── Environment HTTP helpers ───────────────────────────────────────────────────

def call_env(endpoint: str, payload: dict | None = None, method: str = "POST") -> dict:
    url = f"{ENV_BASE_URL}/project/{endpoint}"
    if method == "GET":
        resp = requests.get(url, timeout=60)
    else:
        resp = requests.post(url, json=payload or {}, timeout=60)
    resp.raise_for_status()
    return resp.json()


# ── Rule-based fallback ────────────────────────────────────────────────────────

def get_rule_based_action(obs: dict) -> str:
    """
    Deterministic fallback policy for when the LLM is unavailable.

    Priority order:
    1. Act on the highest-priority active instruction's task
    2. Assign highest-priority backlog task with deps satisfied + available dev
    3. Unblock tasks that are genuinely BLOCKED (status==blocked) with deps done
    4. Skip — never loop on the same failing action
    """
    tasks = obs.get("tasks", [])
    devs  = obs.get("developers", [])

    done_ids   = {t["id"] for t in tasks if t["status"] == "done"}
    in_prog_ids = {t["id"] for t in tasks if t["status"] == "in_progress"}
    available  = [d for d in devs if d["is_available"] and d["current_load"] < d["capacity"] * 2]

    def best_dev(task: dict):
        skill_match = [d for d in available if d["skill"] == task["required_skill"] or d["skill"] == "fullstack"]
        return skill_match[0] if skill_match else (available[0] if available else None)

    skip_action = json.dumps({
        "action_type": "skip",
        "task_id": None, "dev_id": None,
        "new_priority": None,
    })

    # 1. Instructions first — assign their backlog tasks if deps met
    active_insts = [i for i in obs.get("instruction_queue", []) if not i.get("followed", False)]
    for inst in sorted(active_insts, key=lambda i: i.get("target_sprint", 99)):
        for tid in inst.get("affects_tasks", []):
            task = next((t for t in tasks if t["id"] == tid), None)
            if task and task["status"] == "backlog":
                deps = task.get("metadata", {}).get("depends_on", [])
                if all(d in done_ids for d in deps):
                    dev = best_dev(task)
                    if dev:
                        return json.dumps({
                            "action_type": "assign",
                            "task_id": task["id"],
                            "dev_id": dev["id"],
                            "new_priority": None,
                        })

    # 2. Highest-priority backlog task with deps satisfied
    backlog = sorted(
        [t for t in tasks if t["status"] == "backlog"],
        key=lambda t: (t["priority"], t["deadline"])
    )
    for task in backlog:
        deps = task.get("metadata", {}).get("depends_on", [])
        if all(d in done_ids for d in deps):
            dev = best_dev(task)
            if dev:
                return json.dumps({
                    "action_type": "assign",
                    "task_id": task["id"],
                    "dev_id": dev["id"],
                    "new_priority": None,
                })

    # 3. Unblock tasks that are genuinely BLOCKED and have deps done now
    for task in tasks:
        if task["status"] == "blocked":
            deps = task.get("metadata", {}).get("depends_on", [])
            if all(d in done_ids for d in deps):
                return json.dumps({
                    "action_type": "unblock",
                    "task_id": task["id"],
                    "dev_id": None,
                    "new_priority": None,
                })

    # 4. Nothing actionable — skip cleanly
    return skip_action


# ── JSON parser ────────────────────────────────────────────────────────────────

_VALID_ACTIONS = {"assign", "reassign", "reprioritize", "skip", "unblock"}
_NULL_STRINGS  = {"null", "none", "None", "Null", "", "undefined", "N/A"}


def parse_action(text: str) -> dict:
    """
    Parse LLM output into a clean action dict the server will accept without 422.
    Handles: markdown fences, "null" strings, sprint_plan, missing fields.
    """
    text = text.strip()

    # Strip markdown code fences
    if "```" in text:
        lines = [l for l in text.split("\n") if not l.strip().startswith("```")]
        text = "\n".join(lines).strip()

    # Parse JSON
    d = None
    try:
        d = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end   = text.rfind("}") + 1
        if start >= 0 and end > start:
            try:
                d = json.loads(text[start:end])
            except json.JSONDecodeError:
                pass

    if d is None:
        return {"action_type": "skip", "task_id": None, "dev_id": None, "new_priority": None}

    # Normalise action_type — sprint_plan and unknowns → skip
    raw = str(d.get("action_type", "skip")).lower().strip()
    if raw not in _VALID_ACTIONS:
        raw = "skip"
    d["action_type"] = raw

    # Convert "null" strings → None
    for key in ("task_id", "dev_id", "new_priority"):
        if str(d.get(key, "")).strip() in _NULL_STRINGS:
            d[key] = None

    # Convert new_priority to int
    if d.get("new_priority") is not None:
        try:
            d["new_priority"] = int(d["new_priority"])
            if d["new_priority"] not in range(1, 6):
                d["new_priority"] = None
        except (ValueError, TypeError):
            d["new_priority"] = None

    # Demote actions missing required fields → skip (avoids server 422)
    atype = d["action_type"]
    if atype in ("assign", "reassign") and (not d.get("task_id") or not d.get("dev_id")):
        d["action_type"] = "skip"
    if atype == "reprioritize" and (not d.get("task_id") or not d.get("new_priority")):
        d["action_type"] = "skip"
    if atype == "unblock" and not d.get("task_id"):
        d["action_type"] = "skip"

    return {
        "action_type":  d["action_type"],
        "task_id":      d.get("task_id"),
        "dev_id":       d.get("dev_id"),
        "new_priority": d.get("new_priority"),
    }


# ── Episode runner ─────────────────────────────────────────────────────────────

def run_episode(task_name: str) -> float:
    """
    Run one complete 60-day episode and return the final project score.

    Emits [START], [STEP]×N, [END] to stdout with flush=True.
    This is the format required by the OpenEnv validator.
    """

    # ── [START] ───────────────────────────────────────────────────────────────
    print(f"[START] task={task_name}", flush=True)

    obs = call_env("reset", {"task_name": task_name, "seed": 42})
    final_score = 0.01
    step_num    = 0

    for step_num in range(1, MAX_STEPS + 1):
        if obs.get("done", False):
            break

        # ── LLM call with rule-based fallback ─────────────────────────────────
        try:
            # First try: system + user roles
            messages = [
                {"role": "system", "content": R2_SYSTEM_PROMPT},
                {"role": "user",   "content": build_user_prompt(obs)},
            ]
            try:
                completion = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=messages,
                    temperature=TEMPERATURE,
                    max_tokens=MAX_TOKENS,
                )
            except Exception as e1:
                # Log the real error detail for debugging
                err_detail = getattr(e1, 'response', None)
                err_text = err_detail.text if err_detail else str(e1)
                if step_num == 1:  # only log on first step to avoid spam
                    print(f"[DEBUG] LLM error detail: {err_text[:200]}", flush=True)

                # Retry: merge system + user into single user message
                # (some HF endpoints don't support system role)
                merged = f"{R2_SYSTEM_PROMPT}\n\n{build_user_prompt(obs)}"
                completion = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[{"role": "user", "content": merged}],
                    temperature=TEMPERATURE,
                    max_tokens=MAX_TOKENS,
                )

            response_text = completion.choices[0].message.content or ""
        except Exception as e:
            if step_num == 1:
                print(f"[WARN] LLM unavailable ({type(e).__name__}: {str(e)[:100]}), using rule-based fallback", flush=True)
            else:
                print(f"[WARN] LLM unavailable ({type(e).__name__}), using rule-based fallback", flush=True)
            response_text = get_rule_based_action(obs)

        action = parse_action(response_text)

        # ── Call environment ──────────────────────────────────────────────────
        result = call_env("step", {"action": action})
        obs    = result["observation"]
        reward = result["reward"]
        done   = result["done"]
        info   = result.get("info", {})

        # ── [STEP] ────────────────────────────────────────────────────────────
        print(
            f"[STEP] task={task_name} step={step_num} "
            f"day={obs.get('current_day', '?')} "
            f"sprint={obs.get('current_sprint', '?')} "
            f"action={action.get('action_type')} "
            f"task={action.get('task_id')} "
            f"reward={reward:.4f} "
            f"cumulative={obs.get('cumulative_reward', 0):.4f} "
            f"inst_score={obs.get('instruction_following_score', 0):.3f} "
            f"debt={len(obs.get('tech_debt', []))} "
            f"done={done}",
            flush=True,
        )

        if done:
            # Compute final project score using grader formula
            # Matches grade_project_easy/medium/hard weights (delivery×inst×health approximation)
            tasks_total   = len(obs.get("tasks", [])) or 1
            tasks_done    = obs.get("tasks_completed", 0)
            inst_score    = obs.get("instruction_following_score", 0.01)
            delivery_rate = tasks_done / tasks_total
            debt_count    = len(obs.get("tech_debt", []))
            team_health   = max(0.01, 1.0 - debt_count * 0.02)
            # Weighted: delivery 55%, inst 30%, health 15% (medium weights — safe middle ground)
            raw = delivery_rate * 0.55 + inst_score * 0.30 + team_health * 0.15
            final_score = max(0.01, min(0.99, raw))
            break

    # ── [END] ─────────────────────────────────────────────────────────────────
    print(
        f"[END] task={task_name} score={final_score:.4f} steps={step_num} "
        f"completed={obs.get('tasks_completed', 0)} "
        f"missed={obs.get('tasks_missed', 0)} "
        f"inst_score={obs.get('instruction_following_score', 0):.3f} "
        f"debt={len(obs.get('tech_debt', []))}",
        flush=True,
    )

    return final_score


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    print(f"[INFO] model={MODEL_NAME} server={ENV_BASE_URL}", flush=True)

    # Health check
    try:
        health = call_env("health", method="GET")
        print(f"[INFO] health={health}", flush=True)
        assert health.get("round") == 2, "Expected R2 health endpoint"
    except Exception as e:
        print(f"[ERROR] Cannot reach R2 env server: {e}", flush=True)
        print(f"[ERROR] Make sure ENV_BASE_URL points to a running server and /project/health returns 200", flush=True)
        sys.exit(1)

    scores: dict[str, float] = {}
    start_time = time.time()

    for task in TASKS:
        try:
            score = run_episode(task)
            scores[task] = score
        except Exception as e:
            print(f"[ERROR] task={task} error={e}", flush=True)
            scores[task] = 0.01

    elapsed = time.time() - start_time

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 62, flush=True)
    print(" ROUND 2 — BASELINE SCORES", flush=True)
    print("=" * 62, flush=True)
    for task, score in scores.items():
        bar = "█" * int(score * 20)
        print(f"  {task:<22} {score:.4f}  {bar}", flush=True)
    avg = sum(scores.values()) / len(scores) if scores else 0.0
    print(f"  {'AVERAGE':<22} {avg:.4f}", flush=True)
    print(f"\n  Runtime: {elapsed:.1f}s", flush=True)
    print("=" * 62, flush=True)


if __name__ == "__main__":
    main()