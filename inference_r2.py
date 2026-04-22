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
MODEL_NAME   = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-1.5B-Instruct")
ENV_BASE_URL = os.getenv("ENV_BASE_URL", "https://sejal-k-ai-sprint-manager.hf.space")

MAX_STEPS   = 60        # full 60-day project
TEMPERATURE = 0.2
MAX_TOKENS  = 350       # slightly more than R1 — instructions need space

TASKS = ["project_easy", "project_medium", "project_hard"]

client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

# ── System prompt ─────────────────────────────────────────────────────────────
# R2 additions over R1:
#   - "ACTIVE INSTRUCTIONS" section — agent must follow these
#   - "TECH DEBT" section — missed tasks drag productivity
#   - sprint context (current sprint, days remaining)
#   - sprint_plan action type for batch planning

R2_SYSTEM_PROMPT = """You are an expert Engineering Manager running a 6-sprint software project (60 days).

Your goals:
1. Complete tasks on time, highest priority first
2. Follow EVERY active instruction — these come from stakeholders and must not be ignored
3. Keep tech debt low — each missed task permanently reduces team productivity
4. Balance developer workload to avoid burnout

Each step output a JSON action with this exact schema:
{
  "action_type": "<assign|reassign|reprioritize|unblock|skip>",
  "task_id": "<task id or null>",
  "dev_id": "<developer id or null>",
  "new_priority": <1-5 or null>
}

Action rules:
- assign       : assign a BACKLOG task to an available developer — ALWAYS prefer skill match
- reassign     : move an IN_PROGRESS task to a different developer
- reprioritize : change task priority (1=highest, 5=lowest)
- unblock      : use ONLY when a task status is "blocked" — do NOT use on backlog tasks
- skip         : do nothing (costs -0.05 reward — avoid unless truly nothing to do)

Critical rules:
- ALWAYS act on ACTIVE INSTRUCTIONS first — assign their referenced tasks immediately
- Only assign tasks whose depends_on tasks are already "done"
- Match developer skill to task required_skill when possible
- Never assign to an unavailable developer (is_available=false)
- Do NOT repeat the same action if it failed last step — try a different task or dev

Output ONLY the JSON object. No explanation, no markdown."""


# ── Prompt builder ─────────────────────────────────────────────────────────────

def build_user_prompt(obs: dict) -> str:
    """
    Build the per-step user prompt including R2 fields:
    sprint context, active instructions, tech debt, tasks, developers.
    """
    current_day    = obs["current_day"]
    current_sprint = obs.get("current_sprint", 1)
    sprint_end     = current_sprint * 10
    days_left      = max(0, sprint_end - current_day + 1)
    total_days     = obs.get("sprint_length", 60)

    # ── Active instructions (R2 key feature) ─────────────────────────────────
    active_insts = [
        i for i in obs.get("instruction_queue", [])
        if not i.get("followed", False)
    ]
    if active_insts:
        inst_lines = "\n".join(
            f"  [{i['id']}] (Sprint {i.get('target_sprint','?')}) {i['text']}"
            for i in active_insts
        )
        inst_section = f"⚠️  ACTIVE INSTRUCTIONS (follow these NOW):\n{inst_lines}"
    else:
        inst_section = "✅ No active instructions pending."

    # ── Tech debt ─────────────────────────────────────────────────────────────
    tech_debt = obs.get("tech_debt", [])
    debt_section = (
        f"🔴 TECH DEBT ({len(tech_debt)} tasks): {', '.join(tech_debt)}"
        if tech_debt else
        "✅ No tech debt."
    )

    # ── Sprint rewards so far ─────────────────────────────────────────────────
    sprint_rewards = obs.get("sprint_rewards", [])
    rewards_str = (
        "  ".join(f"S{i+1}:{r:.2f}" for i, r in enumerate(sprint_rewards))
        or "none yet"
    )

    # ── Tasks (grouped by status for clarity) ─────────────────────────────────
    tasks = obs["tasks"]

    def fmt_task(t: dict) -> str:
        dep = t.get("metadata", {}).get("depends_on", [])
        dep_str = f" deps={dep}" if dep else ""
        return (
            f"  [{t['id']}] {t['name']} | {t['task_type']} | P{t['priority']} | "
            f"effort={t['effort']} | due=Day{t['deadline']} | "
            f"skill={t['required_skill']} | status={t['status']} | "
            f"sprint={t.get('metadata',{}).get('sprint','?')} | "
            f"dev={t['assigned_to']} | prog={t['progress']:.0%}{dep_str}"
        )

    backlog     = [t for t in tasks if t["status"] == "backlog"]
    in_progress = [t for t in tasks if t["status"] == "in_progress"]
    blocked     = [t for t in tasks if t["status"] == "blocked"]
    done        = [t for t in tasks if t["status"] == "done"]
    missed      = [t for t in tasks if t["status"] == "missed"]

    # Sort backlog by priority then deadline
    backlog_sorted = sorted(backlog, key=lambda t: (t["priority"], t["deadline"]))

    tasks_section = ""
    if backlog_sorted:
        tasks_section += "BACKLOG (assign these):\n" + "\n".join(fmt_task(t) for t in backlog_sorted[:8])
        if len(backlog_sorted) > 8:
            tasks_section += f"\n  ... and {len(backlog_sorted)-8} more backlog tasks"
    if in_progress:
        tasks_section += "\nIN PROGRESS:\n" + "\n".join(fmt_task(t) for t in in_progress)
    if blocked:
        tasks_section += "\nBLOCKED:\n" + "\n".join(fmt_task(t) for t in blocked)
    if missed:
        tasks_section += f"\nMISSED: {', '.join(t['id'] for t in missed)}"
    if done:
        tasks_section += f"\nDONE: {', '.join(t['id'] for t in done)}"

    # ── Developers ────────────────────────────────────────────────────────────
    devs_section = "\n".join(
        f"  [{d['id']}] {d['name']} | skill={d['skill']} | "
        f"load={d['current_load']}/{d['capacity']} | "
        f"avail={'YES' if d['is_available'] else 'NO (absent)'}"
        for d in obs["developers"]
    )

    events_str = "\n  ".join(obs.get("events", [])[-5:]) or "None"

    return f"""═══ PROJECT STATUS ═══════════════════════════════════════
Day {current_day}/{total_days} | Sprint {current_sprint}/6 | {days_left} days left in sprint
Completed:{obs['tasks_completed']}  Missed:{obs['tasks_missed']}  In-Progress:{obs['tasks_in_progress']}  Backlog:{obs['tasks_backlog']}
Inst-Following: {obs.get('instruction_following_score', 1.0):.2f} | Cumulative Reward: {obs['cumulative_reward']:.2f}
Sprint rewards so far: {rewards_str}

{inst_section}

{debt_section}

TASKS:
{tasks_section}

DEVELOPERS:
{devs_section}

Recent events:
  {events_str}
══════════════════════════════════════════════════════════
Output your JSON action:"""


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

def parse_action(text: str) -> dict:
    """
    Parse LLM output into an action dict.
    Strips markdown fences, extracts first JSON object found.
    Falls back to skip on any parse error.
    """
    text = text.strip()

    # Strip markdown code fences
    if "```" in text:
        lines = [l for l in text.split("\n") if not l.strip().startswith("```")]
        text = "\n".join(lines).strip()

    # Try direct parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Extract first {...} block
    start = text.find("{")
    end   = text.rfind("}") + 1
    if start >= 0 and end > start:
        try:
            return json.loads(text[start:end])
        except json.JSONDecodeError:
            pass

    # Fallback
    return {
        "action_type": "skip",
        "task_id": None, "dev_id": None,
        "new_priority": None, "task_ids": None, "notes": None,
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
            completion = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": R2_SYSTEM_PROMPT},
                    {"role": "user",   "content": build_user_prompt(obs)},
                ],
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
            )
            response_text = completion.choices[0].message.content or ""
        except Exception as e:
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