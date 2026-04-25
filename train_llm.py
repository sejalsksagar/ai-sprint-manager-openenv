"""
train_llm.py — Round 2 LLM Training with GRPO
===============================================
Trains Llama-3.1-8B (or Qwen2.5-1.5B) using GRPO on the sprint management
environment. Uses curriculum learning:
  Phase r1  : Single-sprint R1 tasks (10 steps each)    — warm up
  Phase r2  : Multi-sprint R2 tasks (60 steps each)     — full project horizon
  Phase both: Curriculum — alternates R1 and R2, 2:2 ratio (default)

Training method: GRPO (Group Relative Policy Optimisation) — NOT SFT.
  - We do NOT use supervised fine-tuning (SFT) at all.
  - GRPO directly optimises the policy using environment reward signals.
  - The model generates action candidates, the environment scores them,
    GRPO updates weights to prefer higher-reward completions.
  - This is the same family as PPO but simpler: no value network needed.
  - Reference: DeepSeek-R1 uses GRPO for RL training, which is why the
    competition framing calls it "R1 inference".

Reward design — see REWARD DESIGN section below.

Environment:
  ENV_BASE_URL must point to a running server with BOTH R1 and R2 endpoints.

Run locally to verify (no GPU):
    python train_llm.py --smoke-test

Run on GPU (full training):
    python train_llm.py --phase both --episodes 300 --output results/trained_model --push

Required env vars:
    HF_TOKEN       : HuggingFace token (read + write for push)
    ENV_BASE_URL   : Running environment server
    MODEL_NAME     : Base model (default: meta-llama/Llama-3.1-8B-Instruct)
    HF_REPO_ID     : (optional) push trained model here after training

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 REWARD DESIGN
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
We use TWO complementary reward signals, both returned from the environment:

1. STEP REWARD (dense, every step)
   Returned directly from /step → result["reward"].
   Defined in project_grader.py. Values typically in [-3, +2] range.
   Examples:
     +1.7  : assign a high-priority task correctly on first try
     +1.1  : complete a task (task moves to done)
     -0.05 : skip (opportunity cost)
     -0.1  : redundant assign (task already in_progress)
     -2.1  : end of sprint with task still incomplete (sprint miss penalty)
     -2.45 : sprint boundary with tech debt accumulation

   For GRPO we normalise this to [0, 1]:
     normalised = clip((step_reward + 3.0) / 5.0, 0, 1)
   The +3 shift centres the zero-reward case at 0.6 so GRPO doesn't
   always see near-zero rewards (which causes training collapse).

2. EPISODE REWARD (sparse, final step only)
   Computed from final observation when done=True.
   Formula (matches project_grader.py medium weights):
     delivery_rate = tasks_completed / total_tasks
     team_health   = max(0.01, 1.0 - tech_debt_count * 0.02)
     final_score   = delivery_rate * 0.55
                   + instruction_following_score * 0.30
                   + team_health * 0.15
   Clamped to [0.01, 0.99].

3. INSTRUCTION FOLLOWING BONUS (injected into R2 reward)
   obs["instruction_following_score"] is a running average of how often
   the agent acted on active instructions when they became available.
   We add this as an auxiliary reward component:
     combined_r2 = step_norm * 0.6 + inst_score * 0.4
   This gives the model a denser signal on instruction compliance
   rather than waiting for the sparse end-of-sprint penalty.

Why GRPO, not SFT?
  - SFT would require labelled (observation, optimal_action) pairs.
    We don't have those — there's no single "correct" action per step.
  - GRPO generates multiple action candidates per prompt (num_generations=4),
    evaluates each with the environment, and trains toward higher-reward ones.
  - This is fundamentally RL, not imitation learning.

Why NOT call the LLM during the reward function?
  - The HF router rate-limits at ~10k tokens/minute on the free tier.
  - Training with 4 generations × 200 examples = 800 completions × ~300 tokens
    each = 240k tokens. This takes ~24 minutes at the free-tier rate.
  - We use the environment's own reward signal — no external LLM calls needed.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Optional

# ── Env config ────────────────────────────────────────────────────────────────

ENV_BASE_URL = os.getenv("ENV_BASE_URL", "https://sejal-k-ai-sprint-manager.hf.space")
HF_TOKEN     = os.getenv("HF_TOKEN", "")
MODEL_NAME   = os.getenv("MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct")
HF_REPO_ID   = os.getenv("HF_REPO_ID", "")

# ── GRPO hyperparameters ───────────────────────────────────────────────────────
# Tuned for T4 (16GB). For A100, double batch sizes and num_generations.
GRPO_CONFIG = {
    "learning_rate":               5e-6,
    "num_train_epochs":            1,
    "per_device_train_batch_size": 2,   # → 1 if OOM on T4
    "gradient_accumulation_steps": 4,   # effective batch = 8
    "max_prompt_length":           1024,
    "max_completion_length":       128,
    "num_generations":             2,   # → 2 if OOM; must be ≥ 2
    "temperature":                 0.8,
    "beta":                        0.04,  # KL penalty (lower = more exploration)
    "logging_steps":               5,
    "save_steps":                  50,
    "warmup_ratio":                0.08,
    "seed":                        42,
}

R1_TASKS = ["easy_sprint", "medium_sprint", "hard_sprint"]
R2_TASKS = ["project_easy", "project_medium", "project_hard"]

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)


# ── System prompts ────────────────────────────────────────────────────────────
# Must match inference_r2.py exactly so training and inference use same format.

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

R2_SYSTEM_PROMPT = """You are an Engineering Manager. Output ONLY a JSON action each step.
Schema: {"action_type":"<assign|reassign|reprioritize|unblock|skip>","task_id":"<id or null>","dev_id":"<id or null>","new_priority":<1-5 or null>}
Rules:
- Follow ACTIVE INSTRUCTIONS first — assign their tasks immediately
- Only assign BACKLOG tasks (not in_progress or done)
- Only assign if deps marked ✓ and developer is available
- unblock: only for blocked tasks
- skip: last resort
Output ONLY the JSON. No explanation."""


# ── Smart rule-based fallback ─────────────────────────────────────────────────

def smart_fallback_r1(obs: dict) -> dict:
    """R1 rule-based policy: assign highest-priority backlog task with skill match."""
    tasks  = obs.get("tasks", [])
    devs   = obs.get("developers", [])
    avail  = [d for d in devs if d["is_available"] and d["current_load"] < d["capacity"]]
    backlog = sorted(
        [t for t in tasks if t["status"] == "backlog"],
        key=lambda t: (t["priority"], t.get("deadline", 99))
    )
    for task in backlog:
        skill = task.get("required_skill", "")
        match = [d for d in avail if d["skill"] == skill or d["skill"] == "fullstack"]
        dev   = match[0] if match else (avail[0] if avail else None)
        if dev:
            return {"action_type": "assign", "task_id": task["id"],
                    "dev_id": dev["id"], "new_priority": None}
    return {"action_type": "skip", "task_id": None, "dev_id": None, "new_priority": None}


def smart_fallback_r2(obs: dict, assigned_set: Optional[set] = None) -> dict:
    """
    R2 rule-based policy. Respects:
    - Active instructions (assign their tasks first, by target_sprint order)
    - Dependency chains (only assign when deps are done)
    - Skill matching
    - assigned_set: don't re-assign tasks already assigned this episode
    """
    if assigned_set is None:
        assigned_set = set()

    tasks     = obs.get("tasks", [])
    devs      = obs.get("developers", [])
    done_ids  = {t["id"] for t in tasks if t["status"] == "done"}
    available = [d for d in devs if d["is_available"] and d["current_load"] < d["capacity"] * 2]

    def best_dev(task: dict):
        skill = task.get("required_skill", "")
        match = [d for d in available if d["skill"] == skill or d["skill"] == "fullstack"]
        return match[0] if match else (available[0] if available else None)

    def can_assign(task: dict) -> bool:
        if task["status"] != "backlog":
            return False
        if task["id"] in assigned_set:
            return False
        deps = task.get("metadata", {}).get("depends_on", [])
        return all(d in done_ids for d in deps)

    skip = {"action_type": "skip", "task_id": None, "dev_id": None, "new_priority": None}

    # 1. Follow active instructions
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

    # 2. Highest-priority backlog with deps met
    backlog = sorted(
        [t for t in tasks if t["status"] == "backlog"],
        key=lambda t: (t["priority"], t.get("deadline", 99))
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


# ── Action parser ─────────────────────────────────────────────────────────────

_VALID_ACTIONS = {"assign", "reassign", "reprioritize", "skip", "unblock"}
_NULL_STRINGS  = {"null", "none", "None", "Null", "", "undefined", "N/A"}


def _parse_action(text: str) -> dict:
    text = text.strip()
    if "```" in text:
        text = "\n".join(l for l in text.split("\n") if not l.strip().startswith("```"))
    d = None
    try:
        d = json.loads(text)
    except Exception:
        s, e = text.find("{"), text.rfind("}") + 1
        if s >= 0 and e > s:
            try:
                d = json.loads(text[s:e])
            except Exception:
                pass
    if d is None:
        return {"action_type": "skip", "task_id": None, "dev_id": None, "new_priority": None}

    raw = str(d.get("action_type", "skip")).lower().strip()
    d["action_type"] = raw if raw in _VALID_ACTIONS else "skip"
    for key in ("task_id", "dev_id", "new_priority"):
        if str(d.get(key, "")).strip() in _NULL_STRINGS:
            d[key] = None
    if d.get("new_priority") is not None:
        try:
            d["new_priority"] = int(d["new_priority"])
            if d["new_priority"] not in range(1, 6):
                d["new_priority"] = None
        except (ValueError, TypeError):
            d["new_priority"] = None

    atype = d["action_type"]
    if atype in ("assign", "reassign") and (not d.get("task_id") or not d.get("dev_id")):
        d["action_type"] = "skip"
    if atype == "reprioritize" and (not d.get("task_id") or not d.get("new_priority")):
        d["action_type"] = "skip"
    if atype == "unblock" and not d.get("task_id"):
        d["action_type"] = "skip"

    return {"action_type": d["action_type"], "task_id": d.get("task_id"),
            "dev_id": d.get("dev_id"), "new_priority": d.get("new_priority")}


# ── Prompt builders ───────────────────────────────────────────────────────────

def _build_r1_prompt(obs: dict) -> str:
    tasks_summary = "\n".join(
        f"  [{t['id']}] {t['name']} | P{t['priority']} | effort={t['effort']} "
        f"| due=Day{t['deadline']} | status={t['status']} | dev={t['assigned_to']}"
        for t in obs["tasks"]
    )
    devs_summary = "\n".join(
        f"  [{d['id']}] {d['name']} | skill={d['skill']} "
        f"| load={d['current_load']}/{d['capacity']} | avail={d['is_available']}"
        for d in obs["developers"]
    )
    return (
        f"Day: {obs['current_day']}/{obs['sprint_length']}\n"
        f"Done:{obs['tasks_completed']} Missed:{obs['tasks_missed']} "
        f"InProgress:{obs['tasks_in_progress']} Backlog:{obs['tasks_backlog']}\n"
        f"Cumulative Reward: {obs['cumulative_reward']:.2f}\n\n"
        f"TASKS:\n{tasks_summary}\n\nDEVELOPERS:\n{devs_summary}\n\n"
        f"Output your JSON action:"
    )


def _build_r2_prompt(obs: dict) -> str:
    """Compact R2 prompt matching inference_r2.py format."""
    current_sprint = obs.get("current_sprint", 1)
    current_day    = obs.get("current_day", 1)
    days_left      = max(0, current_sprint * 10 - current_day + 1)
    tasks     = obs.get("tasks", [])
    done_ids  = {t["id"] for t in tasks if t["status"] == "done"}

    active_insts = [i for i in obs.get("instruction_queue", []) if not i.get("followed", False)]
    inst_section = (
        "⚡FOLLOW: " + " | ".join(f"[{i['id']}] {i['text'][:45]}" for i in active_insts[:2])
    ) if active_insts else "No instructions."

    debt_count = len(obs.get("tech_debt", []))
    backlog = sorted([t for t in tasks if t["status"] == "backlog"],
                     key=lambda t: (t["priority"], t.get("deadline", 99)))
    in_prog = [t for t in tasks if t["status"] == "in_progress"]

    def fmt(t):
        deps = t.get("metadata", {}).get("depends_on", [])
        dep_ok = "✓" if all(d in done_ids for d in deps) else "✗"
        return f"[{t['id']}]P{t['priority']} {t['required_skill'][:4]} {dep_ok} D{t.get('deadline','?')}"

    backlog_str = " ".join(fmt(t) for t in backlog[:6])
    if len(backlog) > 6:
        backlog_str += f" +{len(backlog)-6}"
    inprog_str  = " ".join(f"[{t['id']}]→{t['assigned_to']}" for t in in_prog) or "none"
    avail_devs  = [d for d in obs["developers"] if d["is_available"]]
    devs_str    = " ".join(
        f"[{d['id']}]{d['name'][:4]}({d['skill'][:3]}) {d['current_load']}/{d['capacity']}"
        for d in avail_devs
    )

    return (
        f"D{current_day}/60 S{current_sprint}/6 {days_left}d "
        f"done={obs['tasks_completed']} miss={obs['tasks_missed']} "
        f"inst={obs.get('instruction_following_score',0):.2f} debt={debt_count}\n"
        f"{inst_section}\n"
        f"BACKLOG(✓=deps_ok): {backlog_str}\n"
        f"IN_PROG: {inprog_str}\n"
        f"DEVS(avail): {devs_str}\n"
        f"JSON:"
    )


# ── GRPO reward functions ─────────────────────────────────────────────────────
#
# IMPORTANT: We do NOT call any LLM inside these reward functions.
# We use the environment's own reward signal exclusively.
# This avoids the HF router rate-limit problem entirely.
#
# How GRPO works:
#   1. Trainer samples `num_generations` completions from the model for each prompt.
#   2. Each completion is an action JSON string.
#   3. We parse the action, call /step on the environment, get a reward.
#   4. GRPO normalises rewards within the group and computes a policy gradient loss.
#   5. Model weights update to prefer completions with above-average reward.
#
# The reward functions below are called by GRPOTrainer for each (prompt, completion) pair.

def make_reward_fn(env_base_url: str, phase: str):
    """
    Returns a GRPO reward function that evaluates completions against the live env.

    Reward normalisation:
      R1: normalised = clip((step_reward + 3.0) / 5.0, 0, 1)
          step_reward range is roughly [-3, +2]. Shift by +3, divide by 5.
          This maps: -3→0, 0→0.6, 2→1.0

      R2: combined = step_norm * 0.6 + inst_score * 0.4
          Blends step reward with instruction-following signal.
          This makes instruction compliance a first-class training objective.
    """
    import requests
    episode_counter = [0]

    def call(endpoint: str, payload: dict, base: str = env_base_url) -> dict:
        resp = requests.post(f"{base}/{endpoint}", json=payload, timeout=30)
        resp.raise_for_status()
        return resp.json()

    def reward_fn(prompts, completions, **kwargs) -> list[float]:
        rewards = []
        for prompt, completion in zip(prompts, completions):
            episode_counter[0] += 1
            n = episode_counter[0]

            # Curriculum: R1 for first 2 of every 4 steps, R2 for last 2
            if phase == "r1":
                use_r2 = False
            elif phase == "r2":
                use_r2 = True
            else:  # both
                use_r2 = (n % 4 >= 2)

            action = _parse_action(completion)

            try:
                if not use_r2:
                    task = R1_TASKS[n % len(R1_TASKS)]
                    obs  = call("reset", {"task_name": task, "seed": n})
                    result = requests.post(
                        f"{env_base_url}/step",
                        json={"action": action}, timeout=30
                    ).json()
                    step_r = float(result.get("reward", 0.0))
                    # Normalise R1 reward to [0,1]
                    r = max(0.0, min(1.0, (step_r + 3.0) / 5.0))
                else:
                    task = R2_TASKS[n % len(R2_TASKS)]
                    obs  = call("project/reset", {"task_name": task, "seed": n})
                    result = requests.post(
                        f"{env_base_url}/project/step",
                        json={"action": action}, timeout=30
                    ).json()
                    step_r     = float(result.get("reward", 0.0))
                    obs2       = result.get("observation", {})
                    inst_score = float(obs2.get("instruction_following_score", 0.5))
                    step_norm  = max(0.0, min(1.0, (step_r + 3.0) / 5.0))
                    # Blend: step quality + instruction compliance
                    r = step_norm * 0.6 + inst_score * 0.4
            except Exception as e:
                # Env call failed — give a neutral reward, don't crash training
                print(f"[WARN] reward_fn env call failed: {e}", flush=True)
                r = 0.3  # slightly below average so bad actions don't get free pass

            rewards.append(float(r))
        return rewards

    return reward_fn


# ── Dataset builder ───────────────────────────────────────────────────────────
#
# We collect observation snapshots using our smart rule-based policy.
# This gives us diverse, realistic game states for GRPO to train on.
# The LLM is NOT called here — only the rule-based policy advances the game.
#
# Why not call LLM here?
#   - Each LLM call costs tokens against the HF rate limit.
#   - Dataset collection for 300 examples × 6 steps = 1800 rule-based steps.
#   - Rule-based is deterministic, fast, and free of rate limits.

def build_grpo_dataset(n_examples: int = 200, phase: str = "both"):
    """
    Build a HuggingFace Dataset of (system, user) chat prompt pairs.
    Each example is an observation snapshot from a rule-based episode.
    """
    try:
        from datasets import Dataset
    except ImportError:
        print("[ERROR] datasets not installed. Run: pip install datasets", flush=True)
        sys.exit(1)

    import requests

    def call_r1(endpoint: str, payload: dict) -> dict:
        r = requests.post(f"{ENV_BASE_URL}/{endpoint}", json=payload, timeout=30)
        r.raise_for_status()
        return r.json()

    def call_r2(endpoint: str, payload: dict) -> dict:
        r = requests.post(f"{ENV_BASE_URL}/project/{endpoint}", json=payload, timeout=30)
        r.raise_for_status()
        return r.json()

    examples = []
    tasks_r1 = R1_TASKS if phase in ("r1", "both") else []
    tasks_r2 = R2_TASKS if phase in ("r2", "both") else []
    per_task = max(1, n_examples // max(1, len(tasks_r1) + len(tasks_r2)))

    # ── Collect R1 snapshots ──────────────────────────────────────────────────
    for task_name in tasks_r1:
        print(f"  [DATASET] Collecting R1 {task_name} × {per_task} episodes...", flush=True)
        for ep in range(per_task):
            try:
                obs = call_r1("reset", {"task_name": task_name, "seed": ep})
                for step in range(6):   # sample 6 states per episode
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
                    result = requests.post(
                        f"{ENV_BASE_URL}/step", json={"action": action}, timeout=30
                    ).json()
                    obs = result["observation"]
                    if result.get("done", False):
                        break
            except Exception as e:
                print(f"  [WARN] R1 episode {ep} failed: {e}", flush=True)

    # ── Collect R2 snapshots ──────────────────────────────────────────────────
    for task_name in tasks_r2:
        print(f"  [DATASET] Collecting R2 {task_name} × {per_task} episodes...", flush=True)
        assigned_set: set[str] = set()
        for ep in range(per_task):
            try:
                obs = call_r2("reset", {"task_name": task_name, "seed": ep})
                assigned_set.clear()
                for step in range(8):   # 8 states: covers first instruction release (usually day 3-5)
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
                    result = requests.post(
                        f"{ENV_BASE_URL}/project/step",
                        json={"action": action}, timeout=30
                    ).json()
                    obs = result["observation"]
                    if result.get("done", False):
                        break
            except Exception as e:
                print(f"  [WARN] R2 episode {ep} failed: {e}", flush=True)

    print(f"  [DATASET] Total examples: {len(examples)}", flush=True)
    return Dataset.from_list(examples)


# ── Model loader ───────────────────────────────────────────────────────────────

def load_model_and_tokenizer(model_name: str):
    """
    Load model with Unsloth 4-bit QLoRA. Falls back to HF+PEFT if Unsloth unavailable.
    LoRA targets all attention + FFN projection layers for maximum coverage.
    """
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
            lora_dropout=0.05,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=42,
        )
        n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"[INFO] Unsloth model loaded. Trainable params: {n_trainable:,}", flush=True)
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
        model = AutoModelForCausalLM.from_pretrained(
            model_name, quantization_config=bnb, device_map="auto",
            token=HF_TOKEN or None,
        )
        lora_cfg = LoraConfig(
            r=16, lora_alpha=32, lora_dropout=0.05,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, lora_cfg)
        print("[INFO] HF+PEFT model loaded.", flush=True)
        return model, tokenizer, "hf"
    except Exception as e:
        print(f"[ERROR] Cannot load model: {e}", flush=True)
        sys.exit(1)


# ── GRPO trainer ───────────────────────────────────────────────────────────────

def train(
    phase: str = "both",
    n_dataset_examples: int = 200,
    output_dir: str = "results/trained_model",
    push_to_hub: bool = False,
):
    print(f"\n{'='*60}", flush=True)
    print(f" GRPO TRAINING — Phase: {phase.upper()}", flush=True)
    print(f" Model:  {MODEL_NAME}", flush=True)
    print(f" Server: {ENV_BASE_URL}", flush=True)
    print(f" Method: GRPO (NOT SFT) — reward-driven RL", flush=True)
    print(f"{'='*60}\n", flush=True)

    # 1. Load model
    model, tokenizer, backend = load_model_and_tokenizer(MODEL_NAME)

    # 2. Build dataset
    print("[INFO] Building training dataset (rule-based collection, no LLM calls)...", flush=True)
    dataset = build_grpo_dataset(n_examples=n_dataset_examples, phase=phase)

    # 3. Build reward function
    reward_fn = make_reward_fn(ENV_BASE_URL, phase)

    # 4. Configure GRPOTrainer
    try:
        from trl import GRPOConfig, GRPOTrainer
    except ImportError:
        print("[ERROR] trl not installed. Run: pip install trl>=0.9.0", flush=True)
        sys.exit(1)

    grpo_config = GRPOConfig(
        output_dir=output_dir,
        report_to="none",
        **GRPO_CONFIG,
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=reward_fn,
        args=grpo_config,
        train_dataset=dataset,
    )

    # 5. Train
    print("[INFO] Starting GRPO training...", flush=True)
    t0 = time.time()
    trainer.train()
    elapsed = time.time() - t0
    print(f"\n[INFO] Training complete in {elapsed/60:.1f} min", flush=True)

    # 6. Save
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"[INFO] Model saved to {output_dir}", flush=True)

    # 7. Push to Hub
    if push_to_hub and HF_REPO_ID:
        print(f"[INFO] Pushing to HF Hub: {HF_REPO_ID}", flush=True)
        model.push_to_hub(HF_REPO_ID, token=HF_TOKEN)
        tokenizer.push_to_hub(HF_REPO_ID, token=HF_TOKEN)
        print(f"[INFO] Pushed to https://huggingface.co/{HF_REPO_ID}", flush=True)

    return output_dir


# ── Smoke test ─────────────────────────────────────────────────────────────────

def smoke_test():
    """
    5-episode smoke test — no GPU, no model loading.
    Verifies server connectivity, rule-based fallback, and dataset pipeline.
    Run this locally before the GPU session.
    """
    print("\n=== SMOKE TEST (rule-based, no GPU) ===\n", flush=True)

    import requests as _req

    # Health checks
    try:
        r1h = _req.get(f"{ENV_BASE_URL}/health", timeout=10).json()
        r2h = _req.get(f"{ENV_BASE_URL}/project/health", timeout=10).json()
        print(f"[OK] R1 health: {r1h}", flush=True)
        print(f"[OK] R2 health: {r2h}", flush=True)
    except Exception as e:
        print(f"[ERROR] Server not reachable: {e}", flush=True)
        print(f"        Start your server: python ui.py", flush=True)
        sys.exit(1)

    results = {}

    # ── R1 test ──────────────────────────────────────────────────────────────
    for task in ["easy_sprint"]:
        print(f"\n[R1] {task}...", flush=True)
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
                obs = result["observation"]
                total_r += result["reward"]
                print(
                    f"  step {i+1}: action={action['action_type']} "
                    f"day={obs['current_day']} "
                    f"done_tasks={obs['tasks_completed']} reward={result['reward']:.3f}",
                    flush=True,
                )
                if result.get("done", False):
                    break
            results[f"r1/{task}"] = round(total_r, 3)
        except Exception as e:
            print(f"  [ERROR] {e}", flush=True)
            results[f"r1/{task}"] = None

    # ── R2 test ──────────────────────────────────────────────────────────────
    for task in ["project_easy"]:
        print(f"\n[R2] {task} (first 8 steps)...", flush=True)
        try:
            obs = _req.post(f"{ENV_BASE_URL}/project/reset",
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
                obs = result["observation"]
                print(
                    f"  step {i+1}: action={action['action_type']} "
                    f"task={action.get('task_id')} "
                    f"day={obs['current_day']} sprint={obs['current_sprint']} "
                    f"reward={result['reward']:.3f} inst={obs['instruction_following_score']:.2f}",
                    flush=True,
                )
                if result.get("done", False):
                    break
            results["r2/project_easy"] = round(obs["cumulative_reward"], 3)
        except Exception as e:
            print(f"  [ERROR] {e}", flush=True)
            results["r2/project_easy"] = None

    # ── Dataset pipeline test ─────────────────────────────────────────────────
    print(f"\n[DATASET] Testing dataset collection (12 examples)...", flush=True)
    try:
        dataset = build_grpo_dataset(n_examples=12, phase="both")
        print(f"  [OK] Dataset size: {len(dataset)}", flush=True)
        print(f"  [OK] Keys: {list(dataset[0].keys())}", flush=True)
    except Exception as e:
        print(f"  [WARN] Dataset collection failed: {e}", flush=True)

    print(f"\n=== SMOKE TEST RESULTS ===", flush=True)
    for k, v in results.items():
        print(f"  {k}: {v}", flush=True)
    print(f"\n✅ Smoke test complete. Server is ready for GPU training.", flush=True)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="GRPO training for AI Sprint Manager R2")
    parser.add_argument("--smoke-test", action="store_true",
                        help="Run smoke test (no GPU). Do this locally first.")
    parser.add_argument("--phase", choices=["r1", "r2", "both"], default="both")
    parser.add_argument("--episodes", type=int, default=200,
                        help="Number of dataset examples to collect (default: 200)")
    parser.add_argument("--output", type=str, default="results/trained_model")
    parser.add_argument("--push", action="store_true",
                        help="Push trained model to HF Hub (requires HF_REPO_ID)")
    args = parser.parse_args()

    if args.smoke_test:
        smoke_test()
        return

    train(phase=args.phase, n_dataset_examples=args.episodes,
          output_dir=args.output, push_to_hub=args.push)


if __name__ == "__main__":
    main()
