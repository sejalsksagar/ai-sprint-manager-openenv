"""
inference_r2.py — Round 2 LLM agent for AI Sprint Manager
Fixed version: 2024-Q2-patch

Key fixes vs original:
  1. 402 errors no longer retried (billing errors are irrecoverable)
  2. LLM called every LLM_CALL_EVERY steps only (default=3) to stay within quota
  3. smart_fallback actually assigns BACKLOG tasks instead of defaulting to skip
  4. task_id validated with ^T\\d+$ regex before accepting LLM output
  5. Dev rotation in fallback to avoid same dev doing everything
  6. Instruction-aware fallback: prioritises tasks referenced in active instructions
"""

import os
import re
import sys
import time
import json
import random
from typing import Optional
import requests
from openai import OpenAI, APIStatusError, APIConnectionError

# ─── Config ──────────────────────────────────────────────────────────────────
MODEL_NAME   = os.getenv("MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct")
ENV_BASE_URL = "https://sejal-k-ai-sprint-manager.hf.space"
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
HF_TOKEN     = os.getenv("HF_TOKEN", "")

MAX_TOKENS      = 60
MAX_RETRIES     = 2       # Applies only to retryable errors (429, 5xx)
TEMPERATURE     = 0.1
LLM_CALL_EVERY  = 3       # Call LLM every Nth step; use fallback on other steps
BUCKET_MAX      = 35      # Token-bucket calls/min ceiling (informational)

# Regex for valid task IDs — anything else from LLM output is garbage
TASK_ID_RE = re.compile(r"^T\d+$")

# ─── Retryable vs non-retryable HTTP errors ───────────────────────────────────
RETRYABLE_CODES = {429, 500, 502, 503, 504}
# 402 = billing / quota depleted — retrying is pointless, go straight to fallback


def is_retryable(exc: APIStatusError) -> bool:
    return exc.status_code in RETRYABLE_CODES


# ─── OpenAI / HF Router client ───────────────────────────────────────────────
def make_client() -> Optional[OpenAI]:
    if not HF_TOKEN:
        return None
    return OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)


R2_SYSTEM_PROMPT = (
    "You are a tech lead managing a 6-sprint software project. "
    "Each step you receive the current environment state and must output ONE JSON action. "
    "Respond ONLY with valid JSON, no markdown, no explanation."
)


def build_user_prompt(obs: dict) -> str:
    tasks = obs.get("tasks", [])
    backlog = [
        f"{t['id']}({t.get('required_skill','?')},eff={t.get('effort',1)})"
        for t in tasks
        if t.get("status") == "backlog"
    ]
    in_prog = [
        f"{t['id']}({t.get('status','?')})"
        for t in tasks
        if t.get("status") == "in_progress"
    ]
    devs = obs.get("developers", [])
    dev_info = [
        f"{d['id']}({d.get('skill','?')},cap={d.get('remaining_capacity', d.get('capacity',5))})"
        for d in devs
    ]
    instructions = obs.get("instruction_queue", [])
    active_inst = [i.get("text", "")[:80] for i in instructions if not i.get("followed")]

    lines = [
        f"day={obs.get('current_day',0)} sprint={obs.get('current_sprint',0)}",
        f"BACKLOG: {', '.join(backlog[:12]) or 'none'}",
        f"IN_PROGRESS: {', '.join(in_prog[:6]) or 'none'}",
        f"DEVS: {', '.join(dev_info)}",
    ]
    if active_inst:
        lines.append(f"INSTRUCTIONS: {' | '.join(active_inst[:3])}")
    lines.append(
        'Output JSON: {"action_type":"assign","task_id":"T01","dev_id":"dev1"} '
        'Only assign BACKLOG tasks. Valid action_types: assign, reprioritize, unblock, skip.'
    )
    return "\n".join(lines)


# ─── LLM call with smart retry ───────────────────────────────────────────────
def call_llm(client: OpenAI, obs: dict) -> Optional[dict]:
    """
    Call the LLM. Returns parsed action dict or None on failure.
    - 402 (billing): immediate None, no retry
    - 429 / 5xx:     retry up to MAX_RETRIES with exponential backoff
    """
    if client is None:
        return None

    user_msg = build_user_prompt(obs)
    messages = [
        {"role": "system", "content": R2_SYSTEM_PROMPT},
        {"role": "user",   "content": user_msg},
    ]

    last_exc = None
    for attempt in range(1, MAX_RETRIES + 2):  # attempts: 1, 2, 3
        try:
            resp = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                max_tokens=MAX_TOKENS,
                temperature=TEMPERATURE,
            )
            raw = resp.choices[0].message.content or ""
            action = parse_action(raw)
            return action

        except APIStatusError as e:
            last_exc = e
            if not is_retryable(e):
                # 402 or other non-retryable — bail immediately
                print(f"  [LLM] Non-retryable error {e.status_code}, using fallback", flush=True)
                return None
            if attempt <= MAX_RETRIES:
                wait = 2 ** attempt * 1.0 + random.uniform(0, 0.5)
                print(f"  [WARN] LLM attempt {attempt}/{MAX_RETRIES+1} failed "
                      f"({e.status_code}), retry in {wait:.1f}s", flush=True)
                time.sleep(wait)
            else:
                print(f"  [WARN] LLM unavailable after {MAX_RETRIES+1} attempts, "
                      f"using rule-based fallback", flush=True)
                return None

        except (APIConnectionError, Exception) as e:
            last_exc = e
            if attempt <= MAX_RETRIES:
                wait = 2 ** attempt
                print(f"  [WARN] LLM attempt {attempt} connection error, retry in {wait}s", flush=True)
                time.sleep(wait)
            else:
                print(f"  [WARN] LLM connection failed, using fallback", flush=True)
                return None

    return None


# ─── Action parser with strict validation ────────────────────────────────────
def parse_action(raw: str) -> Optional[dict]:
    """
    Parse LLM output into a clean action dict.
    Validates task_id against ^T\\d+$ to reject garbage like
    'auth tasks above all others in sprint 1'.
    """
    if not raw:
        return None

    # Strip markdown fences if present
    raw = raw.strip()
    raw = re.sub(r"^```[a-z]*\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)

    # Try to extract JSON object
    match = re.search(r"\{[^{}]*\}", raw, re.DOTALL)
    if not match:
        return None
    try:
        obj = json.loads(match.group())
    except json.JSONDecodeError:
        return None

    action_type = obj.get("action_type", "")
    valid_types = {"assign", "reassign", "reprioritize", "unblock", "skip", "sprint_plan"}
    if action_type not in valid_types:
        return None

    # Validate task_id — must be T followed by digits (e.g. T01, T12)
    task_id = obj.get("task_id")
    if task_id is not None and not TASK_ID_RE.match(str(task_id)):
        print(f"  [INVALID] task_id={repr(task_id)} rejected (not T##)", flush=True)
        return None

    # assign/reassign need both task_id and dev_id
    if action_type in {"assign", "reassign"}:
        if not task_id or not obj.get("dev_id"):
            return None

    return {
        "action_type": action_type,
        "task_id":     task_id,
        "dev_id":      obj.get("dev_id"),
        "new_priority": obj.get("new_priority"),
        "task_ids":    obj.get("task_ids"),
        "notes":       obj.get("notes"),
    }


# ─── Smart fallback policy ───────────────────────────────────────────────────
def smart_fallback(obs: dict, assigned_this_episode: set, last_dev_idx: list) -> dict:
    """
    Rule-based fallback that actually assigns tasks instead of defaulting to skip.

    Priority order:
      1. BACKLOG tasks referenced in active (unfollowed) instructions
      2. BACKLOG tasks in the current sprint with met dependencies
      3. Any BACKLOG task with met dependencies

    Developer selection:
      - Rotates through available devs (round-robin) to avoid overloading one dev
      - Skips devs with zero remaining capacity

    Falls back to skip only when no assignable tasks exist.
    """
    tasks   = obs.get("tasks", [])
    devs    = obs.get("developers", [])
    instructions = obs.get("instruction_queue", [])
    current_sprint = obs.get("current_sprint", 1)

    # Build sets for quick lookup
    done_statuses = {"done", "missed", "in_progress"}
    completed_ids = {t["id"] for t in tasks if t.get("status") == "done"}

    def deps_met(task):
        return all(dep in completed_ids for dep in task.get("depends_on", []))

    def is_assignable(task):
        # BUG 3 FIX: fallback relies only on real task status from the observation.
        # assigned_this_episode is only for the LLM override check — not here.
        # If the server shows a task as backlog with deps met, we CAN assign it.
        return (
            task.get("status") == "backlog"
            and deps_met(task)
        )

    # Tasks referenced in active instructions
    instruction_task_ids = set()
    for inst in instructions:
        if not inst.get("followed", False):
            for tid in inst.get("affects_tasks", []):
                instruction_task_ids.add(tid)

    # Build candidate list — instruction tasks first, then sprint-targeted, then any backlog
    assignable = [t for t in tasks if is_assignable(t)]
    if not assignable:
        return {"action_type": "skip", "task_id": None, "dev_id": None, "new_priority": None}

    # Sort: instruction-referenced first, then by sprint target, then by priority
    def sort_key(t):
        meta = getattr(t, "metadata", {}) if hasattr(t, "metadata") else {}
        if not isinstance(meta, dict):
            meta = {}
        sprint_target = meta.get("sprint", current_sprint + 99)
        in_inst = 0 if t["id"] in instruction_task_ids else 1
        priority = t.get("priority", 99)
        return (in_inst, sprint_target, priority)

    assignable.sort(key=sort_key)
    task = assignable[0]

    # Pick developer — round-robin rotation, skip zero-capacity devs
    available_devs = [
        d for d in devs
        if d.get("remaining_capacity", d.get("capacity", 1)) > 0
    ]
    if not available_devs:
        available_devs = devs  # all exhausted — assign anyway

    # Prefer skill match, but rotate to avoid always picking same dev
    skill = task.get("required_skill", "")
    skilled_devs = [d for d in available_devs if d.get("skill") == skill]
    pool = skilled_devs if skilled_devs else available_devs

    # Round-robin using last_dev_idx[0] as mutable counter
    idx = last_dev_idx[0] % len(pool)
    dev = pool[idx]
    last_dev_idx[0] = (idx + 1) % len(pool)

    return {
        "action_type": "assign",
        "task_id":     task["id"],
        "dev_id":      dev["id"],
        "new_priority": None,
    }


# ─── Step helper ─────────────────────────────────────────────────────────────
def step(action: dict) -> dict:
    url = f"{ENV_BASE_URL}/project/step"
    resp = requests.post(url, json={"action": action}, timeout=15)
    resp.raise_for_status()
    return resp.json()


def reset(scenario: str, seed: int = 42) -> dict:
    url = f"{ENV_BASE_URL}/project/reset"
    resp = requests.post(url, json={"task_name": scenario, "seed": seed}, timeout=15)
    resp.raise_for_status()
    return resp.json()


def health() -> dict:
    resp = requests.get(f"{ENV_BASE_URL}/project/health", timeout=10)
    resp.raise_for_status()
    return resp.json()


# ─── Episode runner ──────────────────────────────────────────────────────────
def run_episode(scenario: str, client: Optional[OpenAI], seed: int = 42) -> dict:
    obs_data = reset(scenario, seed)
    obs = obs_data.get("observation", obs_data)

    assigned_this_episode: set = set()
    last_dev_idx = [0]   # mutable counter for round-robin dev rotation
    cumulative = 0.0
    step_num = 0
    use_llm_this_step = True

    print(f"\n[START] task={scenario}", flush=True)

    while True:
        step_num += 1
        day     = obs.get("current_day", step_num)
        sprint  = obs.get("current_sprint", 1)
        done    = obs.get("done", False)

        # Decide whether to call LLM this step
        use_llm_this_step = (step_num % LLM_CALL_EVERY == 1) and (client is not None)

        action = None

        if use_llm_this_step:
            action = call_llm(client, obs)

            # Validate action — reject if task already assigned/in-progress/missed
            if action and action.get("action_type") == "assign":
                tid = action.get("task_id")
                tasks = obs.get("tasks", [])
                task_statuses = {t["id"]: t.get("status") for t in tasks}
                tstat = task_statuses.get(tid, "unknown")
                if tstat in ("in_progress", "done", "missed") or tid in assigned_this_episode:
                    print(f"  [OVERRIDE] LLM picked {tid} (status={tstat}) → fallback", flush=True)
                    action = None

        if action is None:
            action = smart_fallback(obs, assigned_this_episode, last_dev_idx)

        # Execute step FIRST — then confirm success before tracking
        result = step(action)
        reward  = result.get("reward", 0.0)
        obs     = result.get("observation", result)
        done    = result.get("done", obs.get("done", False))
        cumulative += reward
        inst_score = obs.get("instruction_following_score", 0.0)
        debt_raw = obs.get("tech_debt", 0)

        # BUG 2 FIX: only add to set after server confirms task moved to in_progress
        # Checking post-step obs prevents blocking tasks whose assignment was rejected
        if action.get("action_type") == "assign" and action.get("task_id"):
            tid_sent = action["task_id"]
            post_statuses = {t["id"]: t.get("status") for t in obs.get("tasks", [])}
            if post_statuses.get(tid_sent) == "in_progress":
                assigned_this_episode.add(tid_sent)

        atype = action.get("action_type", "?")
        tid   = action.get("task_id", "None")
        dev   = action.get("dev_id", "None")
        # Display debt count (not raw list) in step log
        debt_display = len(debt_raw) if isinstance(debt_raw, list) else int(debt_raw or 0)
        print(
            f"[STEP] task={scenario} step={step_num} day={day} sprint={sprint} "
            f"action={atype} task_id={tid} dev={dev} "
            f"reward={reward:.4f} cumulative={cumulative:.4f} "
            f"inst_score={inst_score:.3f} debt={debt_display} done={done}",
            flush=True,
        )

        if done:
            break

    # Final score
    tasks = obs.get("tasks", [])
    completed = sum(1 for t in tasks if t.get("status") == "done")
    missed    = sum(1 for t in tasks if t.get("status") == "missed")
    inst_score = obs.get("instruction_following_score", 0.0)
    # BUG 1 FIX: tech_debt comes back as a list of task IDs, not an int
    debt_raw   = obs.get("tech_debt", 0)
    debt_count = len(debt_raw) if isinstance(debt_raw, list) else int(debt_raw or 0)

    # Normalised final score (mirrors project_grader formula)
    total = len(tasks) or 1
    delivery_rate = completed / total
    team_health   = max(0.01, 1.0 - debt_count * 0.02)
    raw_score = delivery_rate * 0.55 + inst_score * 0.30 + team_health * 0.15
    final_score = max(0.01, min(0.99, raw_score))

    print(
        f"[END] task={scenario} score={final_score:.4f} steps={step_num} "
        f"completed={completed} missed={missed} "
        f"inst_score={inst_score:.3f} debt={debt_count}",
        flush=True,
    )
    return {
        "scenario":    scenario,
        "score":       final_score,
        "completed":   completed,
        "missed":      missed,
        "inst_score":  inst_score,
        "debt":        debt_count,
        "steps":       step_num,
    }


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    scenarios = ["project_easy", "project_medium", "project_hard"]

    llm_enabled = bool(HF_TOKEN)
    client = make_client() if llm_enabled else None

    print(f"[INFO] model={MODEL_NAME}", flush=True)
    print(f"[INFO] server={ENV_BASE_URL}", flush=True)
    print(f"[INFO] llm_enabled={llm_enabled} call_every={LLM_CALL_EVERY} retries={MAX_RETRIES}", flush=True)

    try:
        h = health()
        print(f"[INFO] health={h}", flush=True)
    except Exception as e:
        print(f"[WARN] health check failed: {e}", flush=True)

    results = {}
    t0 = time.time()

    for scenario in scenarios:
        try:
            r = run_episode(scenario, client)
            results[scenario] = r
        except Exception as e:
            print(f"[ERROR] {scenario}: {e}", flush=True)
            results[scenario] = {"score": 0.01, "error": str(e)}

    elapsed = time.time() - t0
    scores = [results[s].get("score", 0) for s in scenarios if s in results]
    avg = sum(scores) / len(scores) if scores else 0

    print("\n" + "=" * 62, flush=True)
    print(" ROUND 2 — SCORES", flush=True)
    print("=" * 62, flush=True)
    for s in scenarios:
        sc = results.get(s, {}).get("score", 0)
        bar = "█" * int(sc * 20)
        print(f"  {s:<22} {sc:.4f}  {bar}", flush=True)
    print(f"\n  AVERAGE                {avg:.4f}", flush=True)
    print(f"\n  Runtime: {elapsed:.1f}s", flush=True)
    print("=" * 62, flush=True)


if __name__ == "__main__":
    main()