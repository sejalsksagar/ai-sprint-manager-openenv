"""
inference.py — Round 1 LLM agent for AI Sprint Manager
=======================================================
Runs the trained Qwen/Qwen2.5-1.5B-Instruct model (or any OpenAI-compatible
endpoint) against the three R1 sprint tasks.

Environment variables:
  HF_TOKEN      : Required. HuggingFace / API key.
  MODEL_NAME    : Model to use (default: trained model on HF Hub).
  API_BASE_URL  : LLM endpoint (default: HF router).
  ENV_BASE_URL  : Sprint env server (default: HF Space).
"""
from __future__ import annotations

import json
import os
import sys
import time

import requests
from dotenv import load_dotenv
from openai import OpenAI, APIStatusError, APIConnectionError

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
API_KEY      = os.getenv("HF_TOKEN") or os.getenv("API_KEY", "dummy")
# Default: the GRPO-trained model pushed to HF Hub after training.
# Switch to "meta-llama/Llama-3.1-8B-Instruct" to reproduce the baseline. #sejal-k/ai-sprint-manager-trained
MODEL_NAME   = os.getenv("MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct")
ENV_BASE_URL = os.getenv("ENV_BASE_URL", "https://sejal-k-ai-sprint-manager.hf.space")

MAX_STEPS   = 12
TEMPERATURE = 0.1
MAX_TOKENS  = 80
TASKS       = ["easy_sprint", "medium_sprint", "hard_sprint"]

client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

SYSTEM_PROMPT = (
    "You are an expert Tech Lead managing an agile sprint. "
    "Your goal: maximise task completion, balance developer workload, meet deadlines. "
    "Each step output a JSON action with EXACTLY this schema:\n"
    '{"action_type":"<assign|reassign|reprioritize|unblock|skip>",'
    '"task_id":"<task id or null>","dev_id":"<developer id or null>",'
    '"new_priority":<1-5 or null>}\n'
    "Rules:\n"
    "  assign: put a BACKLOG task onto an AVAILABLE developer\n"
    "  reassign: move an in-progress task to a different developer\n"
    "  reprioritize: change a task priority (1=highest)\n"
    "  unblock: unblock a blocked task\n"
    "  skip: do nothing\n"
    "Output ONLY the JSON object. No markdown, no explanation."
)


def build_user_prompt(obs: dict) -> str:
    tasks_summary = "\n".join(
        f"  [{t['id']}] {t['name']} | {t.get('task_type','?')} | P{t['priority']} | "
        f"effort={t['effort']} | due=Day{t['deadline']} | status={t['status']} | "
        f"dev={t.get('assigned_to','none')} | progress={t.get('progress', 0):.0%}"
        for t in obs.get("tasks", [])
    )
    devs_summary = "\n".join(
        f"  [{d['id']}] {d['name']} | skill={d['skill']} | "
        f"load={d['current_load']}/{d['capacity']} | available={d['is_available']}"
        for d in obs.get("developers", [])
    )
    events_str = "\n  ".join(obs.get("events", [])) or "None"
    return (
        f"Day: {obs.get('current_day',0)}/{obs.get('sprint_length',10)}\n"
        f"Done:{obs.get('tasks_completed',0)} Missed:{obs.get('tasks_missed',0)} "
        f"InProgress:{obs.get('tasks_in_progress',0)} Backlog:{obs.get('tasks_backlog',0)}\n"
        f"Cumulative Reward: {obs.get('cumulative_reward',0):.2f}\n\n"
        f"Events:\n  {events_str}\n\n"
        f"TASKS:\n{tasks_summary}\n\n"
        f"DEVELOPERS:\n{devs_summary}\n\n"
        "Output your JSON action:"
    )


def call_env(endpoint: str, payload: dict = None, method: str = "POST") -> dict:
    url = f"{ENV_BASE_URL}/{endpoint}"
    if method == "GET":
        resp = requests.get(url, timeout=30)
    else:
        resp = requests.post(url, json=payload or {}, timeout=30)
    resp.raise_for_status()
    return resp.json()


def parse_action(text: str) -> dict:
    """Extract JSON action from LLM output, handling markdown fences."""
    text = text.strip()
    # Strip markdown fences
    import re
    text = re.sub(r"^```[a-z]*\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    # Try direct parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Fallback: find outermost {...}
    start, end = text.find("{"), text.rfind("}") + 1
    if start >= 0 and end > start:
        try:
            return json.loads(text[start:end])
        except Exception:
            pass

    return {"action_type": "skip", "task_id": None, "dev_id": None, "new_priority": None}


def rule_based_fallback(obs: dict) -> str:
    """Best-effort rule-based action when LLM is unavailable."""
    tasks = obs.get("tasks", [])
    devs  = obs.get("developers", [])
    backlog = sorted(
        [t for t in tasks if t["status"] == "backlog"],
        key=lambda t: (t["priority"], t["deadline"])
    )
    if not backlog:
        return '{"action_type":"skip"}'
    task    = backlog[0]
    avail   = [d for d in devs if d["is_available"] and d["current_load"] < d["capacity"]]
    matched = [d for d in avail if d["skill"] == task.get("required_skill") or d["skill"] == "fullstack"]
    dev     = matched[0] if matched else (avail[0] if avail else None)
    if not dev:
        return '{"action_type":"skip"}'
    return json.dumps({
        "action_type": "assign",
        "task_id": task["id"],
        "dev_id": dev["id"],
        "new_priority": None,
    })


def compute_final_score(obs: dict, info: dict) -> float:
    """
    Compute final score.
    Tries info["final_score"] first, then computes from obs.
    Guarantees the result is in [0.01, 0.99].
    """
    # Server may return a pre-computed score in info
    if "final_score" in info:
        return max(0.01, min(0.99, float(info["final_score"])))

    # Compute from terminal obs
    tasks  = obs.get("tasks", [])
    total  = len(tasks) or 1
    done   = sum(1 for t in tasks if t["status"] == "done")
    missed = sum(1 for t in tasks if t["status"] == "missed")
    raw    = done / total - missed / total * 0.3
    return round(max(0.01, min(0.99, raw)), 4)


def run_episode(task_name: str) -> float:
    """Run one complete episode and return final score."""
    print(f"\n[START] task={task_name}", flush=True)

    obs = call_env("reset", {"task_name": task_name, "seed": 42})
    final_score = 0.01
    step_num    = 0

    for step_num in range(1, MAX_STEPS + 1):
        if obs.get("done", False):
            break

        # Call LLM
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
        except (APIStatusError, APIConnectionError, Exception) as e:
            print(f"  [WARN] LLM error: {e} — using rule-based fallback", flush=True)
            response_text = rule_based_fallback(obs)

        action = parse_action(response_text)
        result = call_env("step", {"action": action})
        obs    = result["observation"]
        reward = result["reward"]
        done   = result["done"]
        info   = result.get("info", {})

        print(
            f"[STEP] task={task_name} step={step_num} "
            f"action={action.get('action_type')} "
            f"task_id={action.get('task_id')} "
            f"reward={reward:.4f} cumulative={obs.get('cumulative_reward',0):.4f} "
            f"done={done}",
            flush=True,
        )

        if done:
            final_score = compute_final_score(obs, info)
            break

    # Guard: if loop exhausted without done, compute from last obs
    if step_num == MAX_STEPS and final_score == 0.01:
        final_score = compute_final_score(obs, {})

    tasks_done   = sum(1 for t in obs.get("tasks", []) if t["status"] == "done")
    tasks_missed = sum(1 for t in obs.get("tasks", []) if t["status"] == "missed")
    print(
        f"[END] task={task_name} score={final_score:.4f} steps={step_num} "
        f"completed={tasks_done} missed={tasks_missed}",
        flush=True,
    )
    return final_score


def main():
    print(f"[INFO] model={MODEL_NAME}", flush=True)
    print(f"[INFO] server={ENV_BASE_URL}", flush=True)

    try:
        health = call_env("health", method="GET")
        print(f"[INFO] health={health}", flush=True)
    except Exception as e:
        print(f"[ERROR] Cannot reach env server: {e}", flush=True)
        sys.exit(1)

    scores     = {}
    start_time = time.time()

    for task in TASKS:
        try:
            score = run_episode(task)
            scores[task] = score
        except Exception as e:
            print(f"[ERROR] task={task} error={e}", flush=True)
            scores[task] = 0.01

    elapsed = time.time() - start_time

    print("\n" + "=" * 60, flush=True)
    print("  R1 SCORES", flush=True)
    print("=" * 60, flush=True)
    for task, score in scores.items():
        bar = "█" * int(score * 20)
        print(f"  {task:<20} {score:.4f}  {bar}", flush=True)
    avg = sum(scores.values()) / len(scores) if scores else 0.0
    print(f"  {'AVERAGE':<20} {avg:.4f}", flush=True)
    print(f"\n  Runtime: {elapsed:.1f}s", flush=True)
    print("=" * 60, flush=True)


if __name__ == "__main__":
    main()