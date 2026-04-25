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

SFT (supervised fine-tuning) — collect rule-based (prompt → JSON action) pairs:
    python inference_r2.py --sft-collect --sft-out data/r2_sft.jsonl --sft-n 200

Optional in-process SFT warm-up (GPU + trl + unsloth/transformers required):
    python inference_r2.py --sft-train --sft-out-dir results/sft_warmup --sft-n 200

Smoke test (no GPU / no LLM; needs live ENV_BASE_URL):
    python inference_r2.py --smoke-test
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Optional

import requests
from dotenv import load_dotenv

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

# Episode-level score shaping (client-side): strongly discourage skip / overload patterns
SKIP_SCORE_PENALTY_PER_STEP   = 0.022   # subtract from blended raw per skip
SKIP_SCORE_PENALTY_CAP        = 0.38
OVERLOAD_SCORE_PENALTY_ASSIGN = 0.028  # per assign to a dev already ≥ OVERLOAD_LOAD_RATIO of capacity
OVERLOAD_SCORE_PENALTY_CAP    = 0.32
OVERLOAD_LOAD_RATIO           = 0.78   # load/capacity at or above this counts as overloaded partner

TASKS = ["project_easy", "project_medium", "project_hard"]

# SFT warm-up defaults (same spirit as train_llm.py SFT_CONFIG; LR tuned for small JSON completions)
SFT_TRAIN_CONFIG: dict[str, Any] = {
    "num_train_epochs": 2,
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 4,
    "learning_rate": 2e-5,
    "warmup_ratio": 0.05,
    "logging_steps": 5,
    "save_steps": 100,
}

_openai_client = None


def _get_openai_client():
    """Lazy import so --sft-collect / --help work without openai installed."""
    global _openai_client
    if _openai_client is None:
        from openai import OpenAI

        _openai_client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    return _openai_client

# ── System prompt ─────────────────────────────────────────────────────────────
# Kept deliberately short. Every token in the system prompt burns quota.
# Rules that matter: follow instructions, check deps, don't re-assign in_progress.

R2_SYSTEM_PROMPT = """You are an Engineering Manager. Output ONLY a JSON action each step.

Schema: {"action_type":"<assign|reassign|reprioritize|unblock|skip>","task_id":"<id or null>","dev_id":"<id or null>","new_priority":<1-5 or null>}

Rules:
- Follow ACTIVE INSTRUCTIONS first — assign their tasks immediately
- Only assign BACKLOG tasks (not in_progress or done)
- Only assign if deps marked ✓ and developer is available (✓)
- NEVER overload one developer: pick whoever has the most headroom under capacity; do not assign if their load is already near capacity
- unblock: only for blocked tasks
- skip is catastrophic for delivery and scores — use ONLY when literally no assign/unblock/reprioritize is legal

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
        f"PENALTY_HINT: skip and overload(assign to nearly-full dev) tank score heavily.\n"
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


def unwrap_observation(data: dict) -> dict:
    """Reset/step payloads may nest state under 'observation'."""
    obs = data.get("observation")
    if isinstance(obs, dict):
        return obs
    return data


def dev_headroom(d: dict) -> int:
    """Integer slots left before hard capacity (uses remaining_capacity when present)."""
    cap = int(d.get("capacity", 5) or 5)
    load = int(d.get("current_load", 0) or 0)
    rem = d.get("remaining_capacity")
    if rem is not None:
        try:
            return max(0, int(rem))
        except (TypeError, ValueError):
            pass
    return max(0, cap - load)


def dev_is_overloaded(d: dict, ratio: float = OVERLOAD_LOAD_RATIO) -> bool:
    """True when current_load already consumes ≥ ratio of nominal capacity."""
    cap = int(d.get("capacity", 5) or 5)
    if cap <= 0:
        return False
    load = int(d.get("current_load", 0) or 0)
    return (load / cap) >= ratio


def pick_least_loaded_dev(candidates: list) -> Optional[dict]:
    """Prefer max headroom, then min current_load (spread work; avoid overload)."""
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda d: (dev_headroom(d), -int(d.get("current_load", 0) or 0)),
    )


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
    # Strict capacity: never treat 2× capacity as "available" (that caused overload).
    available = [
        d for d in devs
        if d.get("is_available", False) and dev_headroom(d) > 0
    ]

    def best_dev(task: dict):
        skill_match = [
            d for d in available
            if d["skill"] == task["required_skill"] or d["skill"] == "fullstack"
        ]
        pool = skill_match if skill_match else available
        return pick_least_loaded_dev(pool)

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


def rule_based_would_skip(obs: dict, assigned_this_episode: set,
                          last_failed_task: Optional[str]) -> bool:
    """True only when deterministic policy would also skip (no productive work)."""
    a = get_rule_based_action(obs, assigned_this_episode, last_failed_task)
    return a.get("action_type") == "skip"


def resolve_action_avoid_skip_and_overload(
    obs: dict,
    action: dict,
    assigned_this_episode: set,
    last_failed_task: Optional[str],
) -> dict:
    """
    If the model skips while work exists, or assigns onto an overloaded dev,
    replace with rule-based action (same policy as fallback).
    """
    devs = obs.get("developers", [])

    if action.get("action_type") == "skip":
        if not rule_based_would_skip(obs, assigned_this_episode, last_failed_task):
            return get_rule_based_action(obs, assigned_this_episode, last_failed_task)
        return action

    if action.get("action_type") in ("assign", "reassign") and action.get("dev_id"):
        dev = next((d for d in devs if d.get("id") == action.get("dev_id")), None)
        if dev is None or dev_headroom(dev) <= 0 or dev_is_overloaded(dev):
            return get_rule_based_action(obs, assigned_this_episode, last_failed_task)

    return action


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


# ── SFT: supervised dataset + optional TRL warm-up ───────────────────────────
# Pairs (R2_SYSTEM_PROMPT + build_user_prompt, rule-based JSON) teach format +
# reasonable policy before RL-style methods (see train_llm.py).


def action_to_sft_completion(action: dict) -> str:
    """Target string for SFT: one compact JSON object per line."""
    return json.dumps(
        {
            "action_type": action.get("action_type", "skip"),
            "task_id": action.get("task_id"),
            "dev_id": action.get("dev_id"),
            "new_priority": action.get("new_priority"),
        },
        separators=(",", ":"),
    )


def build_sft_chat_example(user_text: str, completion_json: str) -> dict:
    """TRL-compatible row: chat messages + assistant completion."""
    return {
        "prompt": [
            {"role": "system", "content": R2_SYSTEM_PROMPT},
            {"role": "user", "content": user_text},
        ],
        "completion": completion_json,
    }


def collect_r2_sft_examples(
    n_target: int = 200,
    *,
    min_record_step: int = 3,
    max_record_step: int = 12,
    max_steps_per_episode: int = 60,
    seed_base: int = 1000,
    scenarios: Optional[list[str]] = None,
) -> list[dict]:
    """
    Roll the live R2 env using only get_rule_based_action; record (prompt, completion).

    Only steps in [min_record_step, max_record_step] are written (per-episode window)
    so the set is not dominated by trivial early states (same motivation as train_llm
    sampling mid-episode).
    """
    scenarios = scenarios or list(TASKS)
    examples: list[dict] = []
    per_task = max(1, (n_target + len(scenarios) - 1) // len(scenarios))

    for task_name in scenarios:
        for ep in range(per_task):
            if len(examples) >= n_target:
                print(f"  [SFT] Collected {len(examples)} examples (target {n_target})", flush=True)
                return examples
            try:
                raw = call_env("reset", {"task_name": task_name, "seed": seed_base + ep})
                obs = unwrap_observation(raw)
                assigned: set[str] = set()
                last_failed: Optional[str] = None
                for step_i in range(1, max_steps_per_episode + 1):
                    if obs.get("done", False):
                        break
                    user_text    = build_user_prompt(obs, assigned)
                    action       = get_rule_based_action(obs, assigned, last_failed)
                    completion   = action_to_sft_completion(action)
                    if min_record_step <= step_i <= max_record_step:
                        examples.append(build_sft_chat_example(user_text, completion))
                        if len(examples) >= n_target:
                            print(
                                f"  [SFT] Collected {len(examples)} examples (target {n_target})",
                                flush=True,
                            )
                            return examples
                    if action["action_type"] == "assign" and action.get("task_id"):
                        assigned.add(action["task_id"])
                    result = call_env("step", {"action": action})
                    obs    = unwrap_observation(result)
                    reward = float(result.get("reward", 0.0))
                    if reward < -0.5 and action.get("task_id"):
                        last_failed = action["task_id"]
                    elif reward > 0:
                        last_failed = None
                    if result.get("done", False):
                        break
            except Exception as e:
                print(f"  [WARN] SFT collect task={task_name} ep={ep}: {e}", flush=True)

    print(f"  [SFT] Collected {len(examples)} examples (target {n_target})", flush=True)
    return examples


def save_sft_jsonl(path: str | Path, rows: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"  [SFT] Wrote {len(rows)} rows → {path}", flush=True)


def _sft_row_to_text(row: dict) -> dict:
    parts = []
    for msg in row["prompt"]:
        parts.append(f"<|{msg['role']}|>\n{msg['content']}")
    parts.append(f"<|assistant|>\n{row['completion']}")
    return {"text": "\n".join(parts)}


def _load_model_for_sft(model_name: str):
    """Prefer Unsloth 4-bit + LoRA; fall back to transformers + PEFT."""
    token = os.getenv("HF_TOKEN") or None
    try:
        from unsloth import FastLanguageModel

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=2048,
            dtype=None,
            load_in_4bit=True,
            token=token,
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
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        return model, tokenizer, "unsloth"
    except ImportError:
        pass

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig, TaskType, get_peft_model

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb,
        device_map="auto",
        token=token,
    )
    peft_cfg = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, peft_cfg)
    return model, tokenizer, "hf_peft"


def run_sft_trainer(
    output_dir: str | Path,
    n_examples: int = 200,
    *,
    min_step: int = 3,
    max_step: int = 12,
    model_name: Optional[str] = None,
) -> None:
    """
    Collect trajectories from ENV_BASE_URL, then run TRL SFTTrainer when installed.
    On missing GPU stack, still writes JSONL for offline training (e.g. train_llm.py).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = output_dir / "sft_dataset.jsonl"

    rows = collect_r2_sft_examples(
        n_target=n_examples,
        min_record_step=min_step,
        max_record_step=max_step,
    )
    if not rows:
        print("[SFT] No examples collected — aborting.", flush=True)
        return
    save_sft_jsonl(dataset_path, rows)

    try:
        from datasets import Dataset
        from trl import SFTConfig, SFTTrainer
    except ImportError as e:
        print(
            f"[SFT] Saved dataset only. For in-process training install datasets+trl: {e}",
            flush=True,
        )
        return

    mname = model_name or MODEL_NAME
    try:
        model, tokenizer, backend = _load_model_for_sft(mname)
        print(f"[SFT] Loaded {mname} via {backend}", flush=True)
    except Exception as e:
        print(f"[SFT] Model load failed ({e}). Dataset: {dataset_path}", flush=True)
        return

    ds     = Dataset.from_list(rows).map(_sft_row_to_text)
    sft_hf = str(output_dir / "sft_warmup_hf")
    cfg    = SFTConfig(
        output_dir=sft_hf,
        dataset_text_field="text",
        max_seq_length=1024,
        report_to="none",
        **SFT_TRAIN_CONFIG,
    )
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=ds,
        args=cfg,
    )
    trainer.train()
    print(f"[SFT] Warm-up complete → {sft_hf}", flush=True)


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
            completion = _get_openai_client().chat.completions.create(
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

    obs = unwrap_observation(call_env("reset", {"task_name": task_name, "seed": 42}))
    final_score = 0.01

    # ── Episode memory ────────────────────────────────────────────────────────
    assigned_this_episode: set[str] = set()   # tasks we've sent assign for
    last_failed_task: Optional[str] = None    # last task that got negative reward
    llm_fail_streak: int            = 0       # consecutive LLM failures
    LLM_ABANDON_AFTER               = 5       # give up on LLM after this many consecutive failures
    episode_skip_steps              = 0
    episode_overload_assign_steps   = 0

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
                action = resolve_action_avoid_skip_and_overload(
                    obs, action, assigned_this_episode, last_failed_task
                )
            else:
                llm_fail_streak += 1
                action = get_rule_based_action(obs, assigned_this_episode, last_failed_task)
                action = resolve_action_avoid_skip_and_overload(
                    obs, action, assigned_this_episode, last_failed_task
                )
        else:
            # Either LLM abandoned or between batch intervals
            action = get_rule_based_action(obs, assigned_this_episode, last_failed_task)
            action = resolve_action_avoid_skip_and_overload(
                obs, action, assigned_this_episode, last_failed_task
            )

        # Penalty accounting (pre-step state)
        if action.get("action_type") == "skip":
            episode_skip_steps += 1
        if action.get("action_type") in ("assign", "reassign") and action.get("dev_id"):
            _d = next(
                (d for d in obs.get("developers", []) if d.get("id") == action.get("dev_id")),
                None,
            )
            if _d is not None and dev_is_overloaded(_d):
                episode_overload_assign_steps += 1

        # Track assigned tasks to avoid re-assigning
        if action["action_type"] == "assign" and action.get("task_id"):
            assigned_this_episode.add(action["task_id"])

        # ── Call environment ──────────────────────────────────────────────────
        result = call_env("step", {"action": action})
        obs    = unwrap_observation(result)
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
            skip_penalty = min(
                SKIP_SCORE_PENALTY_CAP,
                episode_skip_steps * SKIP_SCORE_PENALTY_PER_STEP,
            )
            overload_penalty = min(
                OVERLOAD_SCORE_PENALTY_CAP,
                episode_overload_assign_steps * OVERLOAD_SCORE_PENALTY_ASSIGN,
            )
            raw -= skip_penalty + overload_penalty
            final_score = max(0.01, min(0.99, raw))
            break

    # ── [END] ─────────────────────────────────────────────────────────────────
    print(
        f"[END] task={task_name} score={final_score:.4f} steps={step_num} "
        f"completed={obs.get('tasks_completed', 0)} "
        f"missed={obs.get('tasks_missed', 0)} "
        f"inst_score={obs.get('instruction_following_score', 0):.3f} "
        f"debt={len(obs.get('tech_debt', []))} "
        f"skips={episode_skip_steps} overload_assigns={episode_overload_assign_steps}",
        flush=True,
    )

    return final_score


# ── Main ───────────────────────────────────────────────────────────────────────

def smoke_test() -> None:
    """
    Short connectivity + policy smoke — no GPU, no model loading.
    Verifies R2 server, this file's rule-based path, optional R1 + GRPO dataset (train_llm).
    Run before a GPU training session.
    """
    print("\n=== SMOKE TEST (no GPU, no LLM) ===\n", flush=True)

    # R2 health is required for this repo entrypoint
    try:
        r2h = requests.get(f"{ENV_BASE_URL}/project/health", timeout=10)
        r2h.raise_for_status()
        print(f"[OK] R2 health: {r2h.json()}", flush=True)
    except Exception as e:
        print(f"[ERROR] R2 server not reachable: {e}", flush=True)
        print(f"        Check ENV_BASE_URL (default: {ENV_BASE_URL!r})", flush=True)
        sys.exit(1)

    try:
        r1h = requests.get(f"{ENV_BASE_URL}/health", timeout=10).json()
        print(f"[OK] R1 health: {r1h}", flush=True)
    except Exception as e:
        print(f"[WARN] R1 health not available (optional): {e}", flush=True)

    results: dict[str, Optional[float]] = {}

    # ── R1 (optional — needs train_llm.smart_fallback_r1) ─────────────────────
    try:
        from train_llm import smart_fallback_r1  # type: ignore[import-untyped]
    except ImportError:
        smart_fallback_r1 = None  # type: ignore[misc, assignment]
        print("\n[R1] skipped (train_llm not importable)", flush=True)

    if smart_fallback_r1 is not None:
        for task in ["easy_sprint"]:
            print(f"\n[R1] {task}...", flush=True)
            try:
                r = requests.post(
                    f"{ENV_BASE_URL}/reset",
                    json={"task_name": task, "seed": 42},
                    timeout=30,
                )
                r.raise_for_status()
                obs = r.json()
                obs = obs.get("observation", obs)
                total_r = 0.0
                for i in range(10):
                    if obs.get("done", False):
                        break
                    action = smart_fallback_r1(obs)
                    result = requests.post(
                        f"{ENV_BASE_URL}/step",
                        json={"action": action},
                        timeout=30,
                    ).json()
                    obs = result.get("observation", obs)
                    total_r += float(result.get("reward", 0.0))
                    print(
                        f"  step {i + 1}: action={action['action_type']} "
                        f"day={obs.get('current_day', '?')} "
                        f"done_tasks={obs.get('tasks_completed', '?')} "
                        f"reward={float(result.get('reward', 0)):.3f}",
                        flush=True,
                    )
                    if result.get("done", False):
                        break
                results[f"r1/{task}"] = round(total_r, 3)
            except Exception as e:
                print(f"  [ERROR] {e}", flush=True)
                results[f"r1/{task}"] = None

    # ── R2 (this module's rule-based policy) ─────────────────────────────────
    for task in ["project_easy"]:
        print(f"\n[R2] {task} (first 8 steps)...", flush=True)
        try:
            obs = unwrap_observation(
                call_env("reset", {"task_name": task, "seed": 42})
            )
            assigned: set[str] = set()
            for i in range(8):
                if obs.get("done", False):
                    break
                action = get_rule_based_action(obs, assigned, None)
                if action["action_type"] == "assign" and action.get("task_id"):
                    assigned.add(action["task_id"])
                result = call_env("step", {"action": action})
                obs = unwrap_observation(result)
                print(
                    f"  step {i + 1}: action={action['action_type']} "
                    f"task={action.get('task_id')} "
                    f"day={obs.get('current_day', '?')} "
                    f"sprint={obs.get('current_sprint', '?')} "
                    f"reward={float(result.get('reward', 0)):.3f} "
                    f"inst={float(obs.get('instruction_following_score', 0)):.2f}",
                    flush=True,
                )
                if result.get("done", False):
                    break
            results["r2/project_easy"] = round(
                float(obs.get("cumulative_reward", 0.0)), 3
            )
        except Exception as e:
            print(f"  [ERROR] {e}", flush=True)
            results["r2/project_easy"] = None

    # ── Dataset pipeline ──────────────────────────────────────────────────────
    print("\n[DATASET] Testing collection (12 examples)...", flush=True)
    try:
        from train_llm import build_grpo_dataset  # type: ignore[import-untyped]

        dataset = build_grpo_dataset(n_examples=12, phase="both")
        print(f"  [OK] GRPO dataset size: {len(dataset)}", flush=True)
        print(f"  [OK] Keys: {list(dataset[0].keys())}", flush=True)
    except Exception as e:
        print(f"  [WARN] build_grpo_dataset failed: {e}", flush=True)
        try:
            rows = collect_r2_sft_examples(
                n_target=12,
                min_record_step=1,
                max_record_step=8,
            )
            print(f"  [OK] R2 SFT rows (fallback): {len(rows)}", flush=True)
            if rows:
                print(f"  [OK] Keys: {list(rows[0].keys())}", flush=True)
        except Exception as e2:
            print(f"  [WARN] collect_r2_sft_examples failed: {e2}", flush=True)

    print("\n=== SMOKE TEST RESULTS ===", flush=True)
    for k, v in results.items():
        print(f"  {k}: {v}", flush=True)
    print("\n[OK] Smoke test complete.", flush=True)


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


def _cli() -> None:
    parser = argparse.ArgumentParser(
        description="R2 OpenEnv inference, SFT JSONL export, or TRL SFT warm-up.",
    )
    parser.add_argument(
        "--sft-collect",
        action="store_true",
        help="Collect rule-based (prompt→JSON) pairs and write JSONL, then exit",
    )
    parser.add_argument(
        "--sft-train",
        action="store_true",
        help="Collect pairs and run TRL SFTTrainer (GPU + trl + transformers)",
    )
    parser.add_argument(
        "--sft-out",
        type=str,
        default="data/r2_sft.jsonl",
        help="Output path for --sft-collect",
    )
    parser.add_argument(
        "--sft-out-dir",
        type=str,
        default="results/sft_r2",
        help="Output directory for --sft-train (JSONL + HF checkpoints)",
    )
    parser.add_argument(
        "--sft-n",
        type=int,
        default=200,
        help="Target number of supervised rows",
    )
    parser.add_argument(
        "--sft-min-step",
        type=int,
        default=3,
        help="First timestep index (per episode) to record",
    )
    parser.add_argument(
        "--sft-max-step",
        type=int,
        default=12,
        help="Last timestep index (per episode) to record",
    )
    parser.add_argument(
        "--sft-model",
        type=str,
        default=None,
        help="HF model id for --sft-train (default: MODEL_NAME env / default)",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run R1/R2 + dataset connectivity smoke (no GPU, no LLM)",
    )
    args = parser.parse_args()

    if args.smoke_test:
        smoke_test()
        return

    if args.sft_collect or args.sft_train:
        try:
            health = call_env("health", method="GET")
            print(f"[INFO] health={health}", flush=True)
            if health.get("round") != 2:
                print("[WARN] health.round != 2 — continuing anyway", flush=True)
        except Exception as e:
            print(f"[ERROR] Cannot reach R2 env: {e}", flush=True)
            sys.exit(1)

    if args.sft_train:
        run_sft_trainer(
            args.sft_out_dir,
            n_examples=args.sft_n,
            min_step=args.sft_min_step,
            max_step=args.sft_max_step,
            model_name=args.sft_model,
        )
        return

    if args.sft_collect:
        rows = collect_r2_sft_examples(
            n_target=args.sft_n,
            min_record_step=args.sft_min_step,
            max_record_step=args.sft_max_step,
        )
        if not rows:
            print("[SFT] No rows collected.", flush=True)
            sys.exit(1)
        save_sft_jsonl(args.sft_out, rows)
        return

    main()


if __name__ == "__main__":
    _cli()
