"""
Inference Script — AI Sprint Manager OpenEnv
============================================================
MANDATORY:
  API_BASE_URL  : LLM endpoint
  MODEL_NAME    : Model identifier
  HF_TOKEN      : Hugging Face / API key
"""
from __future__ import annotations
import os
import json
import time
import sys
import requests
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
API_KEY      = os.getenv("HF_TOKEN") or os.getenv("API_KEY", "dummy")
MODEL_NAME   = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-1.5B-Instruct")
ENV_BASE_URL = os.getenv("ENV_BASE_URL", "https://sejal-k-ai-sprint-manager.hf.space")

MAX_STEPS   = 12
TEMPERATURE = 0.2
MAX_TOKENS  = 300
TASKS       = ["easy_sprint", "medium_sprint", "hard_sprint"]

client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

SYSTEM_PROMPT = """You are an expert Tech Lead managing an agile sprint.
Your goal: maximize task completion, balance developer workload, and meet deadlines.

Each step output a JSON action with this exact schema:
{
  "action_type": "<assign|reassign|reprioritize|unblock|skip>",
  "task_id": "<task id or null>",
  "dev_id": "<developer id or null>",
  "new_priority": <1-5 or null>
}

Rules:
- assign: put a backlog task onto an available developer
- reassign: move an in-progress task to a different developer
- reprioritize: change a task priority (1=highest)
- unblock: unblock a blocked task
- skip: do nothing

Output ONLY the JSON object. No explanation."""


def build_user_prompt(obs: dict) -> str:
    tasks_summary = "\n".join(
        f"  [{t['id']}] {t['name']} | {t['task_type']} | P{t['priority']} | "
        f"effort={t['effort']} | due=Day{t['deadline']} | status={t['status']} | "
        f"dev={t['assigned_to']} | progress={t['progress']:.0%}"
        for t in obs["tasks"]
    )
    devs_summary = "\n".join(
        f"  [{d['id']}] {d['name']} | skill={d['skill']} | "
        f"load={d['current_load']}/{d['capacity']} | available={d['is_available']}"
        for d in obs["developers"]
    )
    events_str = "\n  ".join(obs.get("events", [])) or "None"
    return f"""Day: {obs['current_day']}/{obs['sprint_length']}
Done:{obs['tasks_completed']} Missed:{obs['tasks_missed']} InProgress:{obs['tasks_in_progress']} Backlog:{obs['tasks_backlog']}
Cumulative Reward: {obs['cumulative_reward']:.2f}

Events: {events_str}

TASKS:
{tasks_summary}

DEVELOPERS:
{devs_summary}

Output your JSON action:"""


def call_env(endpoint: str, payload: dict = None, method: str = "POST") -> dict:
    url = f"{ENV_BASE_URL}/{endpoint}"
    if method == "GET":
        resp = requests.get(url, timeout=30)
    else:
        resp = requests.post(url, json=payload or {}, timeout=30)
    resp.raise_for_status()
    return resp.json()


def get_rule_based_action(obs: dict) -> str:
    """Fallback rule-based action when LLM unavailable."""
    tasks = obs.get("tasks", [])
    devs = obs.get("developers", [])
    backlog = sorted(
        [t for t in tasks if t["status"] == "backlog"],
        key=lambda t: (t["priority"], t["deadline"])
    )
    if not backlog:
        return '{"action_type": "skip"}'
    task = backlog[0]
    available = [d for d in devs if d["is_available"] and d["current_load"] < d["capacity"]]
    skill_match = [d for d in available if d["skill"] == task["required_skill"] or d["skill"] == "fullstack"]
    dev = skill_match[0] if skill_match else (available[0] if available else None)
    if not dev:
        return '{"action_type": "skip"}'
    return json.dumps({"action_type": "assign", "task_id": task["id"], "dev_id": dev["id"], "new_priority": None})


def parse_action(text: str) -> dict:
    text = text.strip()
    if "```" in text:
        lines = [l for l in text.split("\n") if not l.strip().startswith("```")]
        text = "\n".join(lines)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}") + 1
        if start >= 0 and end > start:
            try:
                return json.loads(text[start:end])
            except Exception:
                pass
    return {"action_type": "skip", "task_id": None, "dev_id": None, "new_priority": None}


def run_episode(task_name: str) -> float:
    """Run one complete episode and return final score."""

    # ── [START] block ─────────────────────────────────────────────────────────
    print(f"[START] task={task_name}", flush=True)

    obs = call_env("reset", {"task_name": task_name, "seed": 42})
    final_score = 0.0
    step_num = 0

    for step_num in range(1, MAX_STEPS + 1):
        if obs.get("done", False):
            break

        try:
            completion = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": build_user_prompt(obs)},
                ],
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
            )
            response_text = completion.choices[0].message.content or ""
        except Exception as e:
            response_text = get_rule_based_action(obs)

        action = parse_action(response_text)
        result = call_env("step", {"action": action})
        obs = result["observation"]
        reward = result["reward"]
        done = result["done"]
        info = result.get("info", {})

        # ── [STEP] block ──────────────────────────────────────────────────────
        print(
            f"[STEP] task={task_name} step={step_num} "
            f"action={action.get('action_type')} reward={reward:.4f} "
            f"cumulative={obs.get('cumulative_reward', 0):.4f} done={done}",
            flush=True
        )

        if done:
            final_score = max(0.01, min(0.99, info.get("final_score", 0.01)))
            break

    # ── [END] block ───────────────────────────────────────────────────────────
    print(
        f"[END] task={task_name} score={final_score:.4f} steps={step_num}",
        flush=True
    )
    return final_score


def main():
    print(f"[INFO] model={MODEL_NAME} server={ENV_BASE_URL}", flush=True)

    try:
        health = call_env("health", method="GET")
        print(f"[INFO] health={health}", flush=True)
    except Exception as e:
        print(f"[ERROR] Cannot reach env server: {e}", flush=True)
        sys.exit(1)

    scores = {}
    start_time = time.time()

    for task in TASKS:
        try:
            score = run_episode(task)
            scores[task] = score
        except Exception as e:
            print(f"[ERROR] task={task} error={e}", flush=True)
            scores[task] = 0.0

    elapsed = time.time() - start_time

    # Human-readable summary
    print("\n" + "="*60, flush=True)
    print("  BASELINE SCORES", flush=True)
    print("="*60, flush=True)
    for task, score in scores.items():
        bar = "█" * int(score * 20)
        print(f"  {task:<20} {score:.4f}  {bar}", flush=True)
    avg = sum(scores.values()) / len(scores) if scores else 0.0
    print(f"  {'AVERAGE':<20} {avg:.4f}", flush=True)
    print(f"\n  Runtime: {elapsed:.1f}s", flush=True)
    print("="*60, flush=True)


if __name__ == "__main__":
    main()