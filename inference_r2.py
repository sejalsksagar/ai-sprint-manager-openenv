"""
Inference Script — AI Sprint Manager Round 2 (Multi-Sprint Project Manager)
============================================================================
Key improvements over previous version:
  - 3-retry backoff before rule-based fallback (was 1 try → immediate fallback)
  - Batched LLM calls: only call LLM every N steps, reuse for simple repeats
  - No repeated assignments: episode-level memory tracks assigned/in-progress tasks
  - Smart fallback: instruction-aware, dep-aware, never repeats same failing action
  - Rate-limit-aware: exponential back-off (1s, 2s, 4s) between retries
  - Compact prompt: ~250 tokens (was ~350+), avoids hitting HF free-tier quota

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
    HF_TOKEN=hf_... python inference_r2.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from typing import Optional

import requests
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
API_KEY      = os.getenv("HF_TOKEN") or os.getenv("API_KEY", "dummy")
MODEL_NAME   = os.getenv("MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct")
ENV_BASE_URL = os.getenv("ENV_BASE_URL", "https://sejal-k-ai-sprint-manager.hf.space")

MAX_STEPS    = 60        # full 60-day project
TEMPERATURE  = 0.15     # low: we want deterministic JSON not creative writing
MAX_TOKENS   = 80       # action JSON is tiny — 80 tokens is plenty

# Retry / rate-limit config
MAX_RETRIES  = 3        # attempts before falling back to rule-based
RETRY_DELAYS = [1, 2, 4]  # seconds between retries (exponential-ish)

# LLM call batching: only call LLM every LLM_CALL_EVERY steps.
# Between calls, repeat the last valid action if it worked, else fallback.
# Set to 1 to call every step (best quality, most tokens).
# Set to 2-3 to halve/third token usage (helps stay under HF free-tier quota).
LLM_CALL_EVERY = 1  # call every step — adjust to 2 if rate limits are severe

TASKS = ["project_easy", "project_medium", "project_hard"]

client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

# ── System prompt ─────────────────────────────────────────────────────────────
# Kept deliberately short. Every token in the system prompt burns quota.
# Rules that matter: follow instructions, check deps, don't re-assign in_progress.

R2_SYSTEM_PROMPT = """You are an Engineering Manager. Output ONLY a JSON action each step.

Schema: {"action_type":"<assign|reassign|reprioritize|unblock|skip>","task_id":"<id or null>","dev_id":"<id or null>","new_priority":<1-5 or null>}

Rules:
- Follow ACTIVE INSTRUCTIONS first — assign their tasks immediately
- Only assign BACKLOG tasks (not in_progress or done)
- Only assign if deps marked ✓ and developer is available (✓)
- unblock: only for blocked tasks
- skip: last resort

Output ONLY the JSON. No explanation."""


# ── Prompt builder ─────────────────────────────────────────────────────────────

def build_user_prompt(obs: dict, assigned_this_episode: set) -> str:
    """
    Compact per-step prompt. Target: under 250 tokens.
    Shows only actionable backlog tasks (deps met, not already assigned this episode).
    """
    current_day    = obs["current_day"]
    current_sprint = obs.get("current_sprint", 1)
    days_left      = max(0, current_sprint * 10 - current_day + 1)

    tasks    = obs["tasks"]
    done_ids = {t["id"] for t in tasks if t["status"] == "done"}

    # Active unfollowed instructions — top 2 only to save tokens
    active_insts = [i for i in obs.get("instruction_queue", []) if not i.get("followed", False)]
    if active_insts:
        inst_lines = " | ".join(
            f"[{i['id']}] {i['text'][:45]}" for i in active_insts[:2]
        )
        inst_section = f"⚡FOLLOW: {inst_lines}"
    else:
        inst_section = "No instructions."

    # Tech debt — just count
    debt_count = len(obs.get("tech_debt", []))
    debt_str   = f"debt={debt_count}" if debt_count else "debt=0"

    # Backlog tasks — only actionable ones (deps met), exclude already-assigned
    backlog = sorted(
        [t for t in tasks if t["status"] == "backlog"],
        key=lambda t: (t["priority"], t["deadline"])
    )

    def fmt(t: dict) -> str:
        deps = t.get("metadata", {}).get("depends_on", [])
        dep_ok = "✓" if all(d in done_ids for d in deps) else "✗"
        already = "★" if t["id"] in assigned_this_episode else ""
        return f"[{t['id']}]P{t['priority']} {t['required_skill'][:4]} {dep_ok}{already} D{t['deadline']}"

    # Prioritise: actionable (✓ deps, not already assigned), then rest
    actionable = [t for t in backlog
                  if all(d in done_ids for d in t.get("metadata", {}).get("depends_on", []))
                  and t["id"] not in assigned_this_episode]
    blocked_deps = [t for t in backlog if t not in actionable]

    shown = actionable[:5] + blocked_deps[:2]
    backlog_str = " ".join(fmt(t) for t in shown)
    if len(backlog) > len(shown):
        backlog_str += f" +{len(backlog)-len(shown)}"

    in_prog = [t for t in tasks if t["status"] == "in_progress"]
    inprog_str = " ".join(f"[{t['id']}]→{t['assigned_to']}" for t in in_prog) or "none"

    # Developers — available ones first
    avail_devs = [d for d in obs["developers"] if d["is_available"]]
    busy_devs  = [d for d in obs["developers"] if not d["is_available"]]
    def fmt_dev(d):
        return f"[{d['id']}]{d['name'][:4]}({d['skill'][:3]}) {d['current_load']}/{d['capacity']}"
    devs_str = " ".join(fmt_dev(d) for d in avail_devs)
    if busy_devs:
        devs_str += " BUSY:" + " ".join(fmt_dev(d) for d in busy_devs)

    return (
        f"D{current_day}/60 S{current_sprint}/6 {days_left}d "
        f"done={obs['tasks_completed']} miss={obs['tasks_missed']} "
        f"inst={obs.get('instruction_following_score',0):.2f} {debt_str}\n"
        f"{inst_section}\n"
        f"BACKLOG(✓=ok ★=assigned): {backlog_str}\n"
        f"IN_PROG: {inprog_str}\n"
        f"DEVS(avail): {devs_str}\n"
        f"JSON:"
    )


# ── Environment HTTP helpers ───────────────────────────────────────────────────

def call_env(endpoint: str, payload: Optional[dict] = None, method: str = "POST") -> dict:
    url = f"{ENV_BASE_URL}/project/{endpoint}"
    if method == "GET":
        resp = requests.get(url, timeout=60)
    else:
        resp = requests.post(url, json=payload or {}, timeout=60)
    resp.raise_for_status()
    return resp.json()


# ── Rule-based fallback ────────────────────────────────────────────────────────

def get_rule_based_action(obs: dict, assigned_this_episode: set,
                          last_failed_task: Optional[str] = None) -> dict:
    """
    Deterministic fallback. Priority:
    1. Active instruction's tasks (if deps met, not already assigned)
    2. Highest-priority backlog (deps met, not already assigned, not last failed)
    3. Unblock genuinely blocked tasks
    4. Skip
    """
    tasks     = obs.get("tasks", [])
    devs      = obs.get("developers", [])
    done_ids  = {t["id"] for t in tasks if t["status"] == "done"}
    available = [d for d in devs if d["is_available"] and d["current_load"] < d["capacity"] * 2]

    def best_dev(task: dict):
        skill_match = [d for d in available
                       if d["skill"] == task["required_skill"] or d["skill"] == "fullstack"]
        return skill_match[0] if skill_match else (available[0] if available else None)

    def can_assign(task: dict) -> bool:
        if task["status"] != "backlog":
            return False
        if task["id"] in assigned_this_episode:
            return False
        if task["id"] == last_failed_task:
            return False
        deps = task.get("metadata", {}).get("depends_on", [])
        return all(d in done_ids for d in deps)

    skip = {"action_type": "skip", "task_id": None, "dev_id": None, "new_priority": None}

    # 1. Instructions first
    active_insts = sorted(
        [i for i in obs.get("instruction_queue", []) if not i.get("followed", False)],
        key=lambda i: i.get("target_sprint", 99)
    )
    for inst in active_insts:
        for tid in inst.get("affects_tasks", []):
            task = next((t for t in tasks if t["id"] == tid), None)
            if task and can_assign(task):
                dev = best_dev(task)
                if dev:
                    return {"action_type": "assign", "task_id": task["id"],
                            "dev_id": dev["id"], "new_priority": None}

    # 2. Highest-priority backlog (deps met)
    backlog = sorted(
        [t for t in tasks if t["status"] == "backlog"],
        key=lambda t: (t["priority"], t["deadline"])
    )
    for task in backlog:
        if can_assign(task):
            dev = best_dev(task)
            if dev:
                return {"action_type": "assign", "task_id": task["id"],
                        "dev_id": dev["id"], "new_priority": None}

    # 3. Unblock
    for task in tasks:
        if task["status"] == "blocked":
            deps = task.get("metadata", {}).get("depends_on", [])
            if all(d in done_ids for d in deps):
                return {"action_type": "unblock", "task_id": task["id"],
                        "dev_id": None, "new_priority": None}

    return skip


# ── JSON parser ────────────────────────────────────────────────────────────────

_VALID_ACTIONS = {"assign", "reassign", "reprioritize", "skip", "unblock"}
_NULL_STRINGS  = {"null", "none", "None", "Null", "", "undefined", "N/A", "n/a"}


def parse_action(text: str) -> dict:
    """
    Parse LLM output into a clean action dict.
    Handles: markdown fences, "null" strings, sprint_plan, missing fields.
    """
    text = text.strip()

    # Strip markdown code fences
    if "```" in text:
        lines = [l for l in text.split("\n") if not l.strip().startswith("```")]
        text  = "\n".join(lines).strip()

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

    # Normalise action_type
    raw = str(d.get("action_type", "skip")).lower().strip()
    if raw not in _VALID_ACTIONS:
        raw = "skip"
    d["action_type"] = raw

    # Convert "null" strings → None
    for key in ("task_id", "dev_id", "new_priority"):
        if str(d.get(key, "")).strip() in _NULL_STRINGS:
            d[key] = None

    # new_priority → int
    if d.get("new_priority") is not None:
        try:
            d["new_priority"] = int(d["new_priority"])
            if d["new_priority"] not in range(1, 6):
                d["new_priority"] = None
        except (ValueError, TypeError):
            d["new_priority"] = None

    # Demote invalid actions → skip
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


# ── LLM call with retry ────────────────────────────────────────────────────────

def call_llm(obs: dict, assigned_this_episode: set, step_num: int) -> tuple[str, bool]:
    """
    Call LLM with up to MAX_RETRIES retries and exponential backoff.
    Returns (response_text, used_llm: bool).
    If all retries fail, returns ("", False) — caller uses rule-based fallback.
    """
    prompt = build_user_prompt(obs, assigned_this_episode)
    messages_sys = [
        {"role": "system", "content": R2_SYSTEM_PROMPT},
        {"role": "user",   "content": prompt},
    ]
    messages_usr = [
        {"role": "user", "content": f"{R2_SYSTEM_PROMPT}\n\n{prompt}"},
    ]

    last_err = None
    for attempt in range(MAX_RETRIES):
        try:
            # First attempt: system+user. Subsequent: merged (some endpoints reject system role)
            msgs = messages_sys if attempt == 0 else messages_usr
            completion = client.chat.completions.create(
                model=MODEL_NAME,
                messages=msgs,
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
            )
            return completion.choices[0].message.content or "", True

        except Exception as e:
            last_err = e
            err_name = type(e).__name__
            if attempt < MAX_RETRIES - 1:
                wait = RETRY_DELAYS[attempt]
                print(
                    f"[WARN] LLM attempt {attempt+1}/{MAX_RETRIES} failed "
                    f"({err_name}), retrying in {wait}s...",
                    flush=True,
                )
                time.sleep(wait)
            else:
                print(
                    f"[WARN] LLM unavailable after {MAX_RETRIES} attempts "
                    f"({err_name}), using rule-based fallback",
                    flush=True,
                )

    return "", False


# ── Episode runner ─────────────────────────────────────────────────────────────

def run_episode(task_name: str) -> float:
    """
    Run one complete 60-day episode and return the final project score.

    Episode-level state:
      assigned_this_episode: set of task IDs the LLM has chosen to assign.
        Prevents the agent looping on the same in_progress task.
      last_failed_task: most recent task that triggered a negative reward
        (used by fallback to avoid repeating the same mistake).
      llm_fail_streak: consecutive steps where LLM was unavailable.
        After 5+ consecutive failures we switch entirely to rule-based
        for the rest of the episode to stop burning retry time.

    Emits [START], [STEP]×N, [END] to stdout with flush=True.
    """
    print(f"[START] task={task_name}", flush=True)

    obs = call_env("reset", {"task_name": task_name, "seed": 42})
    final_score = 0.01

    # ── Episode memory ────────────────────────────────────────────────────────
    assigned_this_episode: set[str] = set()   # tasks we've sent assign for
    last_failed_task: Optional[str] = None    # last task that got negative reward
    llm_fail_streak: int            = 0       # consecutive LLM failures
    LLM_ABANDON_AFTER               = 5       # give up on LLM after this many consecutive failures

    step_num = 0
    for step_num in range(1, MAX_STEPS + 1):
        if obs.get("done", False):
            break

        # ── Choose action ─────────────────────────────────────────────────────
        use_llm = (llm_fail_streak < LLM_ABANDON_AFTER)

        # Optional batching: only call LLM every LLM_CALL_EVERY steps
        # (set LLM_CALL_EVERY = 1 to always call)
        should_call = (step_num % LLM_CALL_EVERY == 1 or LLM_CALL_EVERY == 1)

        if use_llm and should_call:
            response_text, llm_ok = call_llm(obs, assigned_this_episode, step_num)
            if llm_ok:
                llm_fail_streak = 0
                action = parse_action(response_text)

                # ── Validate LLM action and override if needed ────────────────
                # If LLM tries to assign a task that's already in_progress or done,
                # override with smart fallback
                if action["action_type"] == "assign" and action.get("task_id"):
                    tid    = action["task_id"]
                    t_info = next((t for t in obs["tasks"] if t["id"] == tid), None)
                    if t_info and t_info["status"] != "backlog":
                        # Task is in_progress/done — LLM is looping, override
                        action = get_rule_based_action(
                            obs, assigned_this_episode, last_failed_task
                        )
            else:
                llm_fail_streak += 1
                action = get_rule_based_action(obs, assigned_this_episode, last_failed_task)
        else:
            # Either LLM abandoned or between batch intervals
            action = get_rule_based_action(obs, assigned_this_episode, last_failed_task)

        # Track assigned tasks to avoid re-assigning
        if action["action_type"] == "assign" and action.get("task_id"):
            assigned_this_episode.add(action["task_id"])

        # ── Call environment ──────────────────────────────────────────────────
        result = call_env("step", {"action": action})
        obs    = result["observation"]
        reward = result["reward"]
        done   = result["done"]

        # Track which task just failed (for fallback to avoid)
        if reward < -0.5 and action.get("task_id"):
            last_failed_task = action["task_id"]
        elif reward > 0:
            last_failed_task = None  # reset on success

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
            tasks_total   = len(obs.get("tasks", [])) or 1
            tasks_done    = obs.get("tasks_completed", 0)
            inst_score    = obs.get("instruction_following_score", 0.01)
            delivery_rate = tasks_done / tasks_total
            debt_count    = len(obs.get("tech_debt", []))
            team_health   = max(0.01, 1.0 - debt_count * 0.02)
            # Weighted: delivery 55%, inst 30%, health 15%
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
    print(f"[INFO] retry_config=max_retries={MAX_RETRIES} delays={RETRY_DELAYS} "
          f"llm_call_every={LLM_CALL_EVERY}", flush=True)

    # Health check
    try:
        health = call_env("health", method="GET")
        print(f"[INFO] health={health}", flush=True)
        assert health.get("round") == 2, "Expected R2 health endpoint"
    except Exception as e:
        print(f"[ERROR] Cannot reach R2 env server: {e}", flush=True)
        print(f"[ERROR] Make sure ENV_BASE_URL points to a running server", flush=True)
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

    print("\n" + "=" * 62, flush=True)
    print(" ROUND 2 — SCORES", flush=True)
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
