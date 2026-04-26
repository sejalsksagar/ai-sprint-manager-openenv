"""
train_llm.py — AI Sprint Manager R1+R2 Training
================================================
TRAINING APPROACH: SFT warm-up → GRPO fine-tuning (curriculum)

FIXES IN THIS VERSION vs previous:
  [FIX-T1]  reward_fn resets env BEFORE calling /step
  [FIX-T2]  Observation extracted from reset response
  [FIX-T3]  SFT warmup phase added
  [FIX-T4]  Dataset samples from MIDDLE of episodes (steps 3-8)
  [FIX-T5]  Tokenizer pad_token fix
  [FIX-T6]  GRPOConfig: removed unsupported fields
  [FIX-T7]  Push uses model.merge_and_unload() before push
  [FIX-T8]  Neutral fallback reward changed from 0.3 to 0.5
  [FIX-T9]  build_grpo_dataset wraps each episode in try/except
  [FIX-T10] make_reward_fn: extra -0.2 penalty for unnecessary skips
             (skip when backlog+avail_devs are non-empty → normalized
             reward was 0.59, above the 0.5 neutral, teaching the model
             that skipping is "safe". Now: skip with valid assignments
             available is explicitly punished below neutral.)
  [FIX-T11] _build_r2_prompt: adds DO_NOT_ASSIGN line listing in_progress
             task IDs so the model stops trying to re-assign them during
             training (matches the EPISODE_MEMORY hint in inference).
  [FIX-T12] make_reward_fn: extra -0.1 penalty when assign/reassign
             returns step_r <= -0.1 (server rejected the assignment).
             Gives a stronger gradient signal against invalid assigns.
  [FIX-T13] make_format_reward_fn: skip gets 0.1 (was 0.3 = same as a
             valid assign). Format reward now discourages skip-spamming.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Optional

ENV_BASE_URL = os.getenv("ENV_BASE_URL", "https://sejal-k-ai-sprint-manager.hf.space")
HF_TOKEN     = os.getenv("HF_TOKEN", "")
MODEL_NAME   = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-1.5B-Instruct")
HF_REPO_ID   = os.getenv("HF_REPO_ID", "")

GRPO_CONFIG = {
    "learning_rate":               5e-6,
    "num_train_epochs":            1,
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 8,
    "max_prompt_length":           1024,
    "max_completion_length":       96,
    "num_generations":             4,
    "temperature":                 1.0,
    "beta":                        0.04,
    "logging_steps":               5,
    "save_steps":                  50,
    "warmup_steps":                11,
    "seed":                        42,
}

SFT_CONFIG = {
    "num_train_epochs":            2,
    "per_device_train_batch_size": 2,
    "gradient_accumulation_steps": 4,
    "learning_rate":               2e-5,
    "warmup_steps":                5,
    "logging_steps":               5,
    "save_steps":                  100,
}

R1_TASKS = ["easy_sprint", "medium_sprint", "hard_sprint"]
R2_TASKS = ["project_easy", "project_medium", "project_hard"]

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)


R1_SYSTEM_PROMPT = """You are an expert Tech Lead managing an agile sprint.
Each step output a JSON action with this exact schema:
{"action_type":"<assign|reassign|reprioritize|unblock|skip>","task_id":"<id or null>","dev_id":"<id or null>","new_priority":<1-5 or null>}
Rules:
- assign: backlog task onto available developer (prefer skill match)
- reassign: move in-progress task to different developer
- reprioritize: change priority (1=highest)
- unblock: only for BLOCKED tasks (not backlog)
- skip: do nothing
Output ONLY the JSON. No explanation."""

R2_SYSTEM_PROMPT = """You are an Engineering Manager running a 60-day software project.
Each step you MUST output exactly ONE JSON object and nothing else.

Schema (use null for unused fields):
{"action_type":"<assign|reassign|reprioritize|unblock|skip>","task_id":"<id or null>","dev_id":"<id or null>","new_priority":<1-5 or null>}

Rules (follow in order):
1. If ACTIVE INSTRUCTIONS exist, assign THEIR tasks first.
2. Only assign tasks with status=backlog (never in_progress or done).
3. Only assign if all dependency markers show ✓.
4. Only assign to an AVAILABLE developer with matching or fullstack skill.
5. Use unblock ONLY for explicitly blocked tasks whose deps are ✓.
6. skip is last resort.

Output ONLY the JSON. No explanation."""


def smart_fallback_r1(obs: dict) -> dict:
    tasks  = obs.get("tasks", [])
    devs   = obs.get("developers", [])
    avail  = [d for d in devs
              if d.get("is_available", False) and d.get("current_load", 0) < d.get("capacity", 5)]
    backlog = sorted(
        [t for t in tasks if t.get("status") == "backlog"],
        key=lambda t: (t.get("priority", 9), t.get("deadline", 99))
    )
    for task in backlog:
        skill = task.get("required_skill", "")
        match = [d for d in avail if d.get("skill") == skill or d.get("skill") == "fullstack"]
        dev   = match[0] if match else (avail[0] if avail else None)
        if dev:
            return {"action_type": "assign", "task_id": task["id"],
                    "dev_id": dev["id"], "new_priority": None}
    return {"action_type": "skip", "task_id": None, "dev_id": None, "new_priority": None}


def smart_fallback_r2(obs: dict, assigned_set: Optional[set] = None) -> dict:
    if assigned_set is None:
        assigned_set = set()

    tasks     = obs.get("tasks", [])
    devs      = obs.get("developers", [])
    done_ids  = {t["id"] for t in tasks if t.get("status") == "done"}
    available = [d for d in devs
                 if d.get("is_available", False)
                 and d.get("current_load", 0) < d.get("capacity", 5) * 2]

    def best_dev(task: dict) -> Optional[dict]:
        skill = task.get("required_skill", "")
        match = [d for d in available if d.get("skill") == skill or d.get("skill") == "fullstack"]
        return match[0] if match else (available[0] if available else None)

    def can_assign(task: dict) -> bool:
        if task.get("status") != "backlog":
            return False
        if task["id"] in assigned_set:
            return False
        deps = task.get("metadata", {}).get("depends_on", [])
        return all(d in done_ids for d in deps)

    skip = {"action_type": "skip", "task_id": None, "dev_id": None, "new_priority": None}

    active = sorted(
        [i for i in obs.get("instruction_queue", []) if not i.get("followed", False)],
        key=lambda i: i.get("target_sprint", 99)
    )
    for inst in active:
        for tid in inst.get("affects_tasks", []):
            task = next((t for t in tasks if t["id"] == tid), None)
            if task and can_assign(task):
                dev = best_dev(task)
                if dev:
                    return {"action_type": "assign", "task_id": task["id"],
                            "dev_id": dev["id"], "new_priority": None}

    backlog = sorted(
        [t for t in tasks if t.get("status") == "backlog"],
        key=lambda t: (t.get("priority", 9), t.get("deadline", 99))
    )
    for task in backlog:
        if can_assign(task):
            dev = best_dev(task)
            if dev:
                return {"action_type": "assign", "task_id": task["id"],
                        "dev_id": dev["id"], "new_priority": None}

    for task in tasks:
        if task.get("status") == "blocked":
            deps = task.get("metadata", {}).get("depends_on", [])
            if all(d in done_ids for d in deps):
                return {"action_type": "unblock", "task_id": task["id"],
                        "dev_id": None, "new_priority": None}

    return skip


_VALID_ACTIONS = {"assign", "reassign", "reprioritize", "skip", "unblock"}
_NULL_STRINGS  = {"null", "none", "None", "Null", "", "undefined", "N/A", "nil"}


def _parse_action(text) -> dict:
    if isinstance(text, list):
        text = " ".join(
            m.get("content", "") for m in text if m.get("role") == "assistant"
        )
    text = text.strip()
    if "```" in text:
        text = "\n".join(l for l in text.split("\n") if not l.strip().startswith("```"))

    d          = None
    depth      = 0
    obj_start  = -1
    last_start = -1
    last_end   = -1
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                obj_start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and obj_start >= 0:
                last_start = obj_start
                last_end   = i + 1

    if last_start >= 0:
        try:
            d = json.loads(text[last_start:last_end])
        except json.JSONDecodeError:
            pass
    if d is None:
        try:
            d = json.loads(text)
        except Exception:
            pass
    if d is None:
        return {"action_type": "skip", "task_id": None, "dev_id": None, "new_priority": None}

    raw = str(d.get("action_type", "skip")).lower().strip()
    d["action_type"] = raw if raw in _VALID_ACTIONS else "skip"

    for key in ("task_id", "dev_id", "new_priority"):
        val = d.get(key)
        if val is not None and str(val).strip() in _NULL_STRINGS:
            d[key] = None

    if d.get("new_priority") is not None:
        try:
            p = int(d["new_priority"])
            d["new_priority"] = p if 1 <= p <= 5 else None
        except (ValueError, TypeError):
            d["new_priority"] = None

    atype = d["action_type"]
    if atype in ("assign", "reassign") and (not d.get("task_id") or not d.get("dev_id")):
        d["action_type"] = "skip"
    if atype == "reprioritize" and (not d.get("task_id") or d.get("new_priority") is None):
        d["action_type"] = "skip"
    if atype == "unblock" and not d.get("task_id"):
        d["action_type"] = "skip"

    return {"action_type": d["action_type"], "task_id": d.get("task_id"),
            "dev_id": d.get("dev_id"), "new_priority": d.get("new_priority")}


def _build_r1_prompt(obs: dict) -> str:
    tasks_summary = "\n".join(
        f"  [{t['id']}] {t.get('name','?')} | P{t.get('priority','?')} | effort={t.get('effort','?')} "
        f"| due=Day{t.get('deadline','?')} | status={t.get('status','?')} | dev={t.get('assigned_to','none')}"
        for t in obs.get("tasks", [])
    )
    devs_summary = "\n".join(
        f"  [{d['id']}] {d.get('name','?')} | skill={d.get('skill','?')} "
        f"| load={d.get('current_load',0)}/{d.get('capacity',5)} | avail={d.get('is_available',False)}"
        for d in obs.get("developers", [])
    )
    return (
        f"Day: {obs.get('current_day',1)}/{obs.get('sprint_length',10)}\n"
        f"Done:{obs.get('tasks_completed',0)} Missed:{obs.get('tasks_missed',0)} "
        f"InProgress:{obs.get('tasks_in_progress',0)} Backlog:{obs.get('tasks_backlog',0)}\n"
        f"Cumulative Reward: {obs.get('cumulative_reward',0):.2f}\n\n"
        f"TASKS:\n{tasks_summary}\n\nDEVELOPERS:\n{devs_summary}\n\n"
        f"Output your JSON action:"
    )


def _build_r2_prompt(obs: dict) -> str:
    """
    [FIX-T11] Added DO_NOT_ASSIGN line listing in_progress task IDs.
    Previously the prompt showed IN_PROG tasks but didn't explicitly forbid
    re-assigning them. The model would still attempt to assign them, getting
    -0.15 penalties (server rejects assign of in_progress tasks). Now we
    add a hard explicit prohibition so the model stops wasting steps on them.
    This matches the EPISODE_MEMORY/NO_REASSIGN_UNTIL_BACKLOG hint in
    inference_r2.py, keeping training and inference distributions aligned.
    """
    current_sprint = obs.get("current_sprint", 1)
    current_day    = obs.get("current_day", 1)
    days_left      = max(0, current_sprint * 10 - current_day + 1)
    tasks     = obs.get("tasks", [])
    done_ids  = {t["id"] for t in tasks if t.get("status") == "done"}

    active_insts = [i for i in obs.get("instruction_queue", []) if not i.get("followed", False)]
    inst_section = (
        "⚡FOLLOW: " + " | ".join(f"[{i['id']}] {i['text'][:50]}" for i in active_insts[:2])
    ) if active_insts else "No instructions."

    debt_count = len(obs.get("tech_debt", []))
    backlog = sorted([t for t in tasks if t.get("status") == "backlog"],
                     key=lambda t: (t.get("priority", 9), t.get("deadline", 99)))
    in_prog = [t for t in tasks if t.get("status") == "in_progress"]

    def fmt(t: dict) -> str:
        deps = t.get("metadata", {}).get("depends_on", [])
        dep_ok = "✓" if all(d in done_ids for d in deps) else "✗"
        return f"[{t['id']}]P{t.get('priority','?')} {str(t.get('required_skill','?'))[:4]} {dep_ok} D{t.get('deadline','?')}"

    backlog_str = " ".join(fmt(t) for t in backlog[:6])
    if len(backlog) > 6:
        backlog_str += f" +{len(backlog)-6}"
    inprog_str = " ".join(f"[{t['id']}]→{t.get('assigned_to','?')}" for t in in_prog) or "none"
    avail_devs = [d for d in obs.get("developers", []) if d.get("is_available", False)]
    devs_str   = " ".join(
        f"[{d['id']}]{str(d.get('name','?'))[:4]}({str(d.get('skill','?'))[:3]}) "
        f"{d.get('current_load',0)}/{d.get('capacity',5)}"
        for d in avail_devs
    )

    # [FIX-T11] Explicit prohibition on in_progress tasks (aligns with inference memory block)
    no_assign_str = ""
    if in_prog:
        no_assign_str = (
            "DO_NOT_ASSIGN: " + " ".join(f"[{t['id']}]" for t in in_prog)
            + " — already in_progress, assigning again is INVALID.\n"
        )

    return (
        f"D{current_day}/60 S{current_sprint}/6 {days_left}d "
        f"done={obs.get('tasks_completed',0)} miss={obs.get('tasks_missed',0)} "
        f"inst={obs.get('instruction_following_score',0):.2f} debt={debt_count}\n"
        f"{inst_section}\n"
        f"BACKLOG(✓=deps_ok): {backlog_str}\n"
        f"IN_PROG: {inprog_str}\n"
        f"DEVS(avail): {devs_str}\n"
        f"{no_assign_str}"
        f"JSON:"
    )


def make_reward_fn(env_base_url: str, phase: str):
    """
    [FIX-T10] Added extra penalty for unnecessary skips:
      If the model skips when there ARE backlog tasks AND available devs,
      we subtract 0.2 from the normalized reward. Previously skip got
      normalized reward 0.59 (above 0.5 neutral), effectively teaching the
      model that skipping is "safe". Now unnecessary skip lands at ~0.39,
      clearly below neutral.

    [FIX-T12] Added extra -0.1 penalty when assign/reassign is explicitly
      rejected by the server (step_r <= -0.1). Gives a stronger gradient
      signal against invalid assigns beyond what the raw env reward provides.
    """
    import requests

    episode_counter = [0]

    def _post(url: str, payload: dict) -> dict:
        resp = requests.post(url, json=payload, timeout=60)
        resp.raise_for_status()
        return resp.json()

    def reward_fn(prompts, completions, **kwargs) -> list[float]:
        rewards = []
        for prompt, completion in zip(prompts, completions):
            episode_counter[0] += 1
            n = episode_counter[0]

            if phase == "r1":
                use_r2 = False
            elif phase == "r2":
                use_r2 = True
            else:
                use_r2 = (n % 4 >= 2)

            action = _parse_action(completion)
            r = 0.5   # true-neutral fallback [FIX-T8]

            try:
                if not use_r2:
                    task = R1_TASKS[n % len(R1_TASKS)]
                    _post(f"{env_base_url}/reset",
                          {"task_name": task, "seed": n % 100})
                    result = _post(f"{env_base_url}/step", {"action": action})

                    step_r = float(result.get("reward", 0.0))
                    r = max(0.0, min(1.0, (step_r + 3.0) / 5.0))

                    # [FIX-T12] Extra penalty for server-rejected assign/reassign
                    if action["action_type"] in ("assign", "reassign") and step_r <= -0.1:
                        r = max(0.0, r - 0.1)

                else:
                    task = R2_TASKS[n % len(R2_TASKS)]
                    _post(f"{env_base_url}/project/reset",
                          {"task_name": task, "seed": n % 100})
                    result = _post(f"{env_base_url}/project/step",
                                   {"action": action})

                    step_r     = float(result.get("reward", 0.0))
                    obs2       = result.get("observation", {})
                    inst_score = float(obs2.get("instruction_following_score", 0.5))
                    step_norm  = max(0.0, min(1.0, (step_r + 3.0) / 5.0))

                    r = step_norm * 0.6 + inst_score * 0.4

                    # [FIX-T12] Extra penalty for server-rejected assign/reassign
                    if action["action_type"] in ("assign", "reassign") and step_r <= -0.1:
                        r = max(0.0, r - 0.1)

                    # [FIX-T10] Extra penalty for unnecessary skip.
                    # If the model skipped when there were assignable tasks and
                    # available devs, penalize below neutral (was 0.59 → now ≤0.39).
                    if action["action_type"] == "skip":
                        backlog_ct = sum(
                            1 for t in obs2.get("tasks", []) if t.get("status") == "backlog"
                        )
                        avail_ct = sum(
                            1 for d in obs2.get("developers", []) if d.get("is_available")
                        )
                        if backlog_ct > 0 and avail_ct > 0:
                            r = max(0.0, r - 0.20)

            except Exception as e:
                print(f"[WARN] reward_fn env call failed: {e}", flush=True)
                r = 0.5  # [FIX-T8]

            rewards.append(float(r))
        return rewards

    return reward_fn


def make_format_reward_fn():
    """
    [FIX-T13] Reduced skip format reward from 0.3 → 0.1.
    Previously skip received 0.3 (same as a fully-valid assign/reassign action
    with task_id + dev_id). This meant the format signal did nothing to
    discourage skip-spamming. Now:
      0.3 = valid assign/reassign/reprioritize/unblock with required fields
      0.1 = valid JSON but action_type is skip OR missing required fields
      0.0 = not valid JSON
    This way the format reward actively discourages skip relative to a valid
    action with all required fields.
    """
    import re
    _VALID_ACTIONS_SET = {"assign", "reassign", "reprioritize", "unblock", "skip"}
    _TASK_ID_RE = re.compile(r"^T\d+$")

    def format_reward_fn(completions, **kwargs) -> list[float]:
        rewards = []
        for completion in completions:
            if isinstance(completion, list):
                text = " ".join(
                    m.get("content", "") for m in completion if m.get("role") == "assistant"
                )
            else:
                text = str(completion)

            text = text.strip()
            text = re.sub(r"^```[a-z]*\s*", "", text)
            text = re.sub(r"\s*```$", "", text)

            score = 0.0
            try:
                depth = 0; obj_start = -1; last_start = -1; last_end = -1
                for i, ch in enumerate(text):
                    if ch == "{":
                        if depth == 0: obj_start = i
                        depth += 1
                    elif ch == "}":
                        depth -= 1
                        if depth == 0 and obj_start >= 0:
                            last_start, last_end = obj_start, i + 1
                if last_start >= 0:
                    obj = json.loads(text[last_start:last_end])
                else:
                    obj = json.loads(text)

                action_type = obj.get("action_type", "")
                if action_type in _VALID_ACTIONS_SET:
                    # [FIX-T13] skip gets 0.1, not 0.3 — discourage skip-spamming
                    if action_type == "skip":
                        score = 0.1
                    else:
                        score = 0.1  # valid JSON + valid action_type but incomplete fields
                        if action_type in ("assign", "reassign"):
                            tid = obj.get("task_id")
                            did = obj.get("dev_id")
                            if (tid and _TASK_ID_RE.match(str(tid)) and did):
                                score = 0.3
                        elif action_type in ("reprioritize", "unblock"):
                            tid = obj.get("task_id")
                            if tid and _TASK_ID_RE.match(str(tid)):
                                score = 0.3
            except (json.JSONDecodeError, Exception):
                score = 0.0

            rewards.append(float(score))
        return rewards

    return format_reward_fn


def build_grpo_dataset(n_examples: int = 200, phase: str = "both"):
    try:
        from datasets import Dataset
    except ImportError:
        print("[ERROR] datasets not installed. Run: pip install datasets", flush=True)
        sys.exit(1)

    import requests

    def post(url: str, payload: dict) -> dict:
        r = requests.post(url, json=payload, timeout=60)
        r.raise_for_status()
        return r.json()

    examples = []
    tasks_r1 = R1_TASKS if phase in ("r1", "both") else []
    tasks_r2 = R2_TASKS if phase in ("r2", "both") else []
    per_task = max(1, n_examples // max(1, len(tasks_r1) + len(tasks_r2)))

    SKIP_STEPS_R1 = 1
    SKIP_STEPS_R2 = 2
    SAMPLE_PER_EP = 6

    for task_name in tasks_r1:
        print(f"  [DATASET] R1 {task_name} × {per_task} episodes...", flush=True)
        for ep in range(per_task):
            try:
                obs = post(f"{ENV_BASE_URL}/reset", {"task_name": task_name, "seed": ep})
                for _ in range(SKIP_STEPS_R1):
                    if obs.get("done", False):
                        break
                    action = smart_fallback_r1(obs)
                    result = post(f"{ENV_BASE_URL}/step", {"action": action})
                    obs    = result.get("observation", obs)
                    if result.get("done", False):
                        break

                for step in range(SAMPLE_PER_EP):
                    if obs.get("done", False):
                        break
                    prompt = _build_r1_prompt(obs)
                    examples.append({
                        "prompt": [
                            {"role": "system", "content": R1_SYSTEM_PROMPT},
                            {"role": "user",   "content": prompt},
                        ],
                    })
                    action = smart_fallback_r1(obs)
                    result = post(f"{ENV_BASE_URL}/step", {"action": action})
                    obs    = result.get("observation", obs)
                    if result.get("done", False):
                        break
            except Exception as e:
                print(f"  [WARN] R1 ep{ep} failed: {e}", flush=True)

    for task_name in tasks_r2:
        print(f"  [DATASET] R2 {task_name} × {per_task} episodes...", flush=True)
        for ep in range(per_task):
            try:
                obs = post(f"{ENV_BASE_URL}/project/reset",
                           {"task_name": task_name, "seed": ep})
                assigned_set: set[str] = set()

                for _ in range(SKIP_STEPS_R2):
                    if obs.get("done", False):
                        break
                    action = smart_fallback_r2(obs, assigned_set)
                    if action["action_type"] == "assign" and action.get("task_id"):
                        assigned_set.add(action["task_id"])
                    result = post(f"{ENV_BASE_URL}/project/step", {"action": action})
                    obs    = result.get("observation", obs)
                    if result.get("done", False):
                        break

                for step in range(SAMPLE_PER_EP):
                    if obs.get("done", False):
                        break
                    prompt = _build_r2_prompt(obs)
                    examples.append({
                        "prompt": [
                            {"role": "system", "content": R2_SYSTEM_PROMPT},
                            {"role": "user",   "content": prompt},
                        ],
                    })
                    action = smart_fallback_r2(obs, assigned_set)
                    if action["action_type"] == "assign" and action.get("task_id"):
                        assigned_set.add(action["task_id"])
                    result = post(f"{ENV_BASE_URL}/project/step", {"action": action})
                    obs    = result.get("observation", obs)
                    if result.get("done", False):
                        break
            except Exception as e:
                print(f"  [WARN] R2 ep{ep} failed: {e}", flush=True)

    print(f"  [DATASET] Total examples: {len(examples)}", flush=True)
    if not examples:
        print("[ERROR] Dataset is empty — check server connectivity", flush=True)
        sys.exit(1)
    return Dataset.from_list(examples)


def build_sft_dataset(n_examples: int = 100, phase: str = "both"):
    try:
        from datasets import Dataset
    except ImportError:
        sys.exit(1)

    import requests

    def post(url: str, payload: dict) -> dict:
        r = requests.post(url, json=payload, timeout=60)
        r.raise_for_status()
        return r.json()

    examples   = []
    tasks_r1   = R1_TASKS if phase in ("r1", "both") else []
    tasks_r2   = R2_TASKS if phase in ("r2", "both") else []
    per_task   = max(1, n_examples // max(1, len(tasks_r1) + len(tasks_r2)))
    SAMPLE_PER = 4

    for task_name in tasks_r1:
        for ep in range(per_task):
            try:
                obs = post(f"{ENV_BASE_URL}/reset", {"task_name": task_name, "seed": ep + 1000})
                for _ in range(SAMPLE_PER):
                    if obs.get("done", False):
                        break
                    action  = smart_fallback_r1(obs)
                    prompt  = _build_r1_prompt(obs)
                    completion = json.dumps(action)
                    examples.append({
                        "prompt": [
                            {"role": "system", "content": R1_SYSTEM_PROMPT},
                            {"role": "user",   "content": prompt},
                        ],
                        "completion": completion,
                    })
                    result = post(f"{ENV_BASE_URL}/step", {"action": action})
                    obs = result.get("observation", obs)
                    if result.get("done", False):
                        break
            except Exception as e:
                print(f"  [WARN] SFT R1 ep{ep} failed: {e}", flush=True)

    for task_name in tasks_r2:
        for ep in range(per_task):
            try:
                obs = post(f"{ENV_BASE_URL}/project/reset",
                           {"task_name": task_name, "seed": ep + 1000})
                assigned: set[str] = set()
                for _ in range(SAMPLE_PER):
                    if obs.get("done", False):
                        break
                    action = smart_fallback_r2(obs, assigned)
                    if action["action_type"] == "assign" and action.get("task_id"):
                        assigned.add(action["task_id"])
                    prompt     = _build_r2_prompt(obs)
                    completion = json.dumps(action)
                    examples.append({
                        "prompt": [
                            {"role": "system", "content": R2_SYSTEM_PROMPT},
                            {"role": "user",   "content": prompt},
                        ],
                        "completion": completion,
                    })
                    result = post(f"{ENV_BASE_URL}/project/step", {"action": action})
                    obs = result.get("observation", obs)
                    if result.get("done", False):
                        break
            except Exception as e:
                print(f"  [WARN] SFT R2 ep{ep} failed: {e}", flush=True)

    print(f"  [SFT DATASET] Total examples: {len(examples)}", flush=True)
    return Dataset.from_list(examples)


def _format_to_string(example, tokenizer):
    prompt = example.get("prompt")
    completion = example.get("completion")

    # Convert prompt -> string
    if isinstance(prompt, list):
        if hasattr(tokenizer, "apply_chat_template"):
            prompt = tokenizer.apply_chat_template(
                prompt,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            prompt = "\n".join(
                f"<|{m.get('role','user')}|>\n{m.get('content','')}"
                for m in prompt
            ) + "\n<|assistant|>\n"

    # Final safety cast (fixes list/object type crashes in tokenization)
    prompt = str(prompt) if prompt is not None else ""
    completion = str(completion) if completion is not None else ""

    return {
        "prompt": prompt,
        "completion": completion,
    }


def _force_str(example):
    return {
        "prompt": str(example["prompt"]),
        "completion": str(example["completion"]),
    }


def load_model_and_tokenizer(model_name: str):
    try:
        from unsloth import FastLanguageModel
        print(f"[INFO] Loading {model_name} with Unsloth 4-bit QLoRA...", flush=True)
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=2048,
            dtype=None,
            load_in_4bit=True,
            token=HF_TOKEN or None,
        )
        model = FastLanguageModel.get_peft_model(
            model,
            r=16,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
            lora_alpha=32,
            lora_dropout=0.0,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=42,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"[INFO] Unsloth loaded. Trainable params: {n_trainable:,}", flush=True)
        return model, tokenizer, "unsloth"
    except ImportError:
        print("[WARN] Unsloth not available. Falling back to HF + PEFT.", flush=True)
        return _load_hf_model(model_name)


def _load_hf_model(model_name: str):
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from peft import get_peft_model, LoraConfig, TaskType
        import torch

        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name, token=HF_TOKEN or None)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            model_name, quantization_config=bnb, device_map="auto",
            token=HF_TOKEN or None,
        )
        lora_cfg = LoraConfig(
            r=16, lora_alpha=32,
            lora_dropout=0.0,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, lora_cfg)
        print("[INFO] HF+PEFT model loaded.", flush=True)
        return model, tokenizer, "hf"
    except Exception as e:
        print(f"[ERROR] Cannot load model: {e}", flush=True)
        sys.exit(1)


def run_sft(model, tokenizer, phase: str, n_examples: int, output_dir: str):
    print(f"\n[SFT] Warm-up phase ({n_examples} examples)...", flush=True)
    try:
        from trl import SFTTrainer, SFTConfig
    except ImportError:
        print("[WARN] SFTTrainer not in this trl version — skipping SFT warm-up", flush=True)
        return model

    sft_data = build_sft_dataset(n_examples=n_examples, phase=phase)
    sft_data = sft_data.map(
        lambda x: _format_to_string(x, tokenizer),
        num_proc=1,  # keep 1 to avoid multiprocessing serialization issues
    )
    sft_data = sft_data.map(_force_str, num_proc=1)

    def format_fn(example):
        return {"text": f"{example['prompt']}{example['completion']}"}

    sft_data = sft_data.map(format_fn)

    sft_dir  = str(Path(output_dir) / "sft_warmup")
    sft_conf = SFTConfig(
        output_dir=sft_dir,
        dataset_text_field="text",
        max_seq_length=1024,
        report_to="none",
        **SFT_CONFIG,
    )
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=sft_data,
        args=sft_conf,
    )
    trainer.train()
    print("[SFT] Warm-up complete.", flush=True)
    return trainer.model


def train(
    phase: str         = "both",
    n_dataset_examples: int = 200,
    output_dir: str    = "results/trained_model",
    push_to_hub: bool  = False,
    sft_epochs: int    = 0,
    gpu_tier: str      = "t4",
):
    print(f"\n{'='*60}", flush=True)
    print(f" GRPO TRAINING — Phase: {phase.upper()} | GPU: {gpu_tier.upper()}", flush=True)
    print(f" Model:  {MODEL_NAME}", flush=True)
    print(f" Server: {ENV_BASE_URL}", flush=True)
    print(f" SFT warm-up epochs: {sft_epochs}", flush=True)
    print(f"{'='*60}\n", flush=True)

    cfg = dict(GRPO_CONFIG)
    if gpu_tier == "a10g":
        cfg["per_device_train_batch_size"] = 2
        cfg["num_generations"]             = 4
    elif gpu_tier == "a100":
        cfg["per_device_train_batch_size"] = 4
        cfg["num_generations"]             = 4
        cfg["gradient_accumulation_steps"] = 4

    model, tokenizer, backend = load_model_and_tokenizer(MODEL_NAME)

    if sft_epochs > 0:
        sft_n = max(50, n_dataset_examples // 3)
        SFT_CONFIG["num_train_epochs"] = sft_epochs
        model = run_sft(model, tokenizer, phase, sft_n, output_dir)

    print("[INFO] Building GRPO training dataset...", flush=True)
    dataset = build_grpo_dataset(n_examples=n_dataset_examples, phase=phase)

    reward_fn = make_reward_fn(ENV_BASE_URL, phase)

    try:
        from trl import GRPOConfig, GRPOTrainer
    except ImportError:
        print("[ERROR] trl not installed. Run: pip install trl>=0.9.0", flush=True)
        sys.exit(1)

    import trl as _trl
    _trl_version = tuple(int(x) for x in _trl.__version__.split(".")[:2])
    if _trl_version < (0, 9):
        print(f"[WARN] trl {_trl.__version__} detected — recommend trl>=0.9.0", flush=True)
        for key in ("warmup_ratio",):
            cfg.pop(key, None)

    format_reward_fn = make_format_reward_fn()

    grpo_config = GRPOConfig(
        output_dir=output_dir,
        report_to="none",
        loss_type="dapo",
        mask_truncated_completions=True,
        scale_rewards="group",
        disable_dropout=True,
        log_completions=True,
        reward_weights=[1.0, 0.2],
        **cfg,
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[reward_fn, format_reward_fn],
        args=grpo_config,
        train_dataset=dataset,
    )

    print("[INFO] Starting GRPO training...", flush=True)
    t0 = time.time()
    trainer.train()
    elapsed = time.time() - t0
    print(f"\n[INFO] Training complete in {elapsed/60:.1f} min", flush=True)

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"[INFO] Model saved to {output_dir}", flush=True)

    if push_to_hub and HF_REPO_ID:
        print(f"[INFO] Pushing to HF Hub: {HF_REPO_ID}", flush=True)
        if backend == "unsloth":
            try:
                from unsloth import FastLanguageModel
                merged = model.merge_and_unload()
                merged.push_to_hub(HF_REPO_ID, token=HF_TOKEN)
                tokenizer.push_to_hub(HF_REPO_ID, token=HF_TOKEN)
                print(f"[INFO] Merged model pushed to https://huggingface.co/{HF_REPO_ID}", flush=True)
            except Exception as e:
                print(f"[WARN] Merge failed ({e}), pushing LoRA adapter only", flush=True)
                model.push_to_hub(HF_REPO_ID, token=HF_TOKEN)
                tokenizer.push_to_hub(HF_REPO_ID, token=HF_TOKEN)
        else:
            model.push_to_hub(HF_REPO_ID, token=HF_TOKEN)
            tokenizer.push_to_hub(HF_REPO_ID, token=HF_TOKEN)

    return output_dir


def smoke_test():
    print("\n=== SMOKE TEST (rule-based, no GPU) ===\n", flush=True)
    import requests as _req

    try:
        r1h = _req.get(f"{ENV_BASE_URL}/health",         timeout=10).json()
        r2h = _req.get(f"{ENV_BASE_URL}/project/health", timeout=10).json()
        print(f"[OK] R1 health: {r1h}", flush=True)
        print(f"[OK] R2 health: {r2h}", flush=True)
    except Exception as e:
        print(f"[ERROR] Server not reachable: {e}", flush=True)
        sys.exit(1)

    results = {}

    task = "easy_sprint"
    print(f"\n[R1] {task} (10 steps)...", flush=True)
    try:
        obs = _req.post(f"{ENV_BASE_URL}/reset",
                        json={"task_name": task, "seed": 42}, timeout=30).json()
        total_r = 0.0
        for i in range(10):
            if obs.get("done", False):
                break
            action = smart_fallback_r1(obs)
            result = _req.post(f"{ENV_BASE_URL}/step",
                                json={"action": action}, timeout=30).json()
            obs     = result.get("observation", obs)
            total_r += result.get("reward", 0.0)
            print(f"  step {i+1}: {action['action_type']} "
                  f"day={obs.get('current_day','?')} "
                  f"done={obs.get('tasks_completed',0)} "
                  f"reward={result.get('reward',0):.3f}", flush=True)
            if result.get("done", False):
                break
        results["r1/easy_sprint"] = round(total_r, 3)
        print(f"[OK] R1 cumulative reward: {total_r:.3f}", flush=True)
    except Exception as e:
        print(f"  [ERROR] {e}", flush=True)
        results["r1/easy_sprint"] = None

    task = "project_easy"
    print(f"\n[R2] {task} (8 steps)...", flush=True)
    try:
        obs      = _req.post(f"{ENV_BASE_URL}/project/reset",
                             json={"task_name": task, "seed": 42}, timeout=30).json()
        assigned: set[str] = set()
        for i in range(8):
            if obs.get("done", False):
                break
            action = smart_fallback_r2(obs, assigned)
            if action["action_type"] == "assign" and action.get("task_id"):
                assigned.add(action["task_id"])
            result = _req.post(f"{ENV_BASE_URL}/project/step",
                                json={"action": action}, timeout=30).json()
            obs = result.get("observation", obs)
            print(f"  step {i+1}: {action['action_type']} "
                  f"task={action.get('task_id')} "
                  f"day={obs.get('current_day','?')} "
                  f"sprint={obs.get('current_sprint','?')} "
                  f"reward={result.get('reward',0):.3f} "
                  f"inst={obs.get('instruction_following_score',0):.2f}", flush=True)
            if result.get("done", False):
                break
        results["r2/project_easy"] = round(obs.get("cumulative_reward", 0), 3)
        print(f"[OK] R2 cumulative reward: {obs.get('cumulative_reward',0):.3f}", flush=True)
    except Exception as e:
        print(f"  [ERROR] {e}", flush=True)
        results["r2/project_easy"] = None

    print(f"\n[DATASET] Testing GRPO dataset (12 examples)...", flush=True)
    try:
        ds = build_grpo_dataset(n_examples=12, phase="both")
        print(f"  [OK] GRPO dataset size: {len(ds)}", flush=True)
        print(f"  [OK] Keys: {list(ds[0].keys())}", flush=True)
    except Exception as e:
        print(f"  [WARN] GRPO dataset failed: {e}", flush=True)

    print(f"\n[DATASET] Testing SFT dataset (12 examples)...", flush=True)
    try:
        sft_ds = build_sft_dataset(n_examples=12, phase="both")
        print(f"  [OK] SFT dataset size: {len(sft_ds)}", flush=True)
        print(f"  [OK] Keys: {list(sft_ds[0].keys())}", flush=True)
    except Exception as e:
        print(f"  [WARN] SFT dataset failed: {e}", flush=True)

    print(f"\n=== SMOKE TEST RESULTS ===", flush=True)
    for k, v in results.items():
        status = "[OK]" if v is not None else "[FAIL]"
        print(f"  {status} {k}: {v}", flush=True)
    print(f"\n✅ Smoke test complete. Server is ready for GPU training.", flush=True)
    print(f"\n Recommended training command (A100):", flush=True)
    print(f"   python train_llm.py --phase both --episodes 300 "
          f"--sft-epochs 2 --gpu-tier a100 --output results/trained_model --push", flush=True)


def main():
    parser = argparse.ArgumentParser(description="SFT+GRPO training for AI Sprint Manager")
    parser.add_argument("--smoke-test",  action="store_true")
    parser.add_argument("--phase",       choices=["r1", "r2", "both", "sft"], default="both")
    parser.add_argument("--episodes",    type=int, default=200)
    parser.add_argument("--sft-epochs",  type=int, default=0)
    parser.add_argument("--gpu-tier",    choices=["t4", "a10g", "a100"], default="t4")
    parser.add_argument("--output",      type=str, default="results/trained_model")
    parser.add_argument("--push",        action="store_true")
    args = parser.parse_args()

    if args.smoke_test:
        smoke_test()
        return

    if args.phase == "sft":
        model, tokenizer, _ = load_model_and_tokenizer(MODEL_NAME)
        run_sft(model, tokenizer, "both", 200, args.output)
        tokenizer.save_pretrained(args.output)
        return

    train(
        phase=args.phase,
        n_dataset_examples=args.episodes,
        output_dir=args.output,
        push_to_hub=args.push,
        sft_epochs=args.sft_epochs,
        gpu_tier=args.gpu_tier,
    )


if __name__ == "__main__":
    main()
