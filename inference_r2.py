"""
inference_r2.py — Round 2 LLM agent for AI Sprint Manager
==========================================================

FIXES IN THIS VERSION (on top of i_r2_4.py):

  [FIX-R1] MODULO BUG (CRITICAL) — step_num % LLM_CALL_EVERY == 1 was ALWAYS
            False when LLM_CALL_EVERY=1 because any integer mod 1 == 0, never 1.
            The LLM was NEVER called — every step fell through to smart_fallback.
            Fix: changed to == 0 (always True when divisor is 1).

  [FIX-R2] SMART_FALLBACK TIER 2 (CRITICAL for inst_score=30% of score) —
            Tier 2 previously only *reprioritized* instruction tasks (wasting an
            entire step). Now it directly ASSIGNS them to the best available
            developer, immediately advancing instruction_following_score.

  [FIX-R3] SMART_FALLBACK TASK ITERATION — previously called assignable[0] and
            sent the action even if no skilled dev existed (env silently rejects,
            wastes the day). Now iterates all assignable tasks until it finds one
            where a skilled dev is available.

  [FIX-R4] SMART_FALLBACK SKILL MISMATCH — when no skilled dev exists for a task
            the fallback previously passed any random dev (skill mismatch → env
            reject). Now returns skip for that task and tries the next one instead.

  [FIX-R5] SMART_FALLBACK DEV SELECTION — now sorts candidates by remaining
            capacity (most available first) and prefers exact skill match over
            fullstack, keeping developer load balanced.

  [FIX-R6] SMART_FALLBACK ASSIGNED_THIS_EPISODE — smart_fallback now accepts and
            filters out already-started task IDs to avoid redundant re-attempts.

  [FIX-R7] INSTRUCTION TEXT PARSER — new _parse_instruction_assignment() extracts
            explicit "assign T3 to D2" type directives from instruction text using
            regex. These are tried first in smart_fallback Tier 0, maximising
            instruction_following_score.

  [FIX-R8] FALLBACK ACTION VALIDATION — smart_fallback output is now run through
            validate_llm_action() before being sent to the env so invalid actions
            are caught and retried rather than silently wasted.

  [FIX-R9] REPRIORITIZE ONLY WHEN ALREADY ASSIGNED — if the top instruction task
            is already in_progress, skip the reprioritize step (env would reject
            it anyway and waste the day).

  [FIX-R10] REDUCED COOLDOWN AGGRESSION — reduced LLM_COOLDOWN_STEPS from 15→5
            and MAX_LLM_SOFT_FAIL_STREAK from 3→5 so the LLM gets more attempts
            before being benched.
"""

from __future__ import annotations

import json
import os
import re
import time
import random
from typing import Optional, Tuple

import requests

# ─── Config ───────────────────────────────────────────────────────────────────
LOCAL_MODEL_PATH = os.getenv("LOCAL_MODEL_PATH", "priyaaaaaasharmaaaaa/trial1")
MODEL_NAME       = os.getenv("MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct")
ENV_BASE_URL     = os.getenv("ENV_BASE_URL", "https://sejal-k-ai-sprint-manager.hf.space")
API_BASE_URL     = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
HF_TOKEN         = os.getenv("HF_TOKEN", "")

_use_llm_raw = os.getenv("USE_LLM", "1").strip().lower()
USE_LLM      = _use_llm_raw not in ("0", "false", "no", "off", "")

LLM_COOLDOWN_STEPS       = int(os.getenv("LLM_COOLDOWN_STEPS",       "5"))   # [FIX-R10] was 15
MAX_LLM_SOFT_FAIL_STREAK = int(os.getenv("MAX_LLM_SOFT_FAIL_STREAK", "5"))   # [FIX-R10] was 3
MAX_LLM_SKIP_STREAK      = int(os.getenv("MAX_LLM_SKIP_STREAK",      "4"))
MAX_SAME_BAD_ASSIGN_STREAK = int(os.getenv("MAX_SAME_BAD_ASSIGN_STREAK", "2"))

MAX_TOKENS     = 96
MAX_RETRIES    = 2
LLM_CALL_EVERY = 1
TEMPERATURE    = 0.3

TASK_ID_RE      = re.compile(r"^T\d+$")
RETRYABLE_CODES = {429, 500, 502, 503, 504}


# ─── Prompts — EXACTLY matching train_llm.py ─────────────────────────────────

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


def build_user_prompt(
    obs: dict,
    *,
    assigned_this_episode: Optional[set] = None,
    assign_attempted_episode: Optional[set] = None,
) -> str:
    current_sprint = obs.get("current_sprint", 1)
    current_day    = obs.get("current_day", 1)
    days_left      = max(0, current_sprint * 10 - current_day + 1)
    tasks          = obs.get("tasks", [])
    done_ids       = {t["id"] for t in tasks if t.get("status") == "done"}

    active_insts = [i for i in obs.get("instruction_queue", []) if not i.get("followed", False)]
    inst_section = (
        "⚡FOLLOW: " + " | ".join(f"[{i['id']}] {i['text'][:50]}" for i in active_insts[:2])
    ) if active_insts else "No instructions."

    debt_raw   = obs.get("tech_debt", [])
    debt_count = len(debt_raw) if isinstance(debt_raw, list) else int(debt_raw or 0)

    backlog = sorted(
        [t for t in tasks if t.get("status") == "backlog"],
        key=lambda t: (t.get("priority", 9), t.get("deadline", 99))
    )
    in_prog = [t for t in tasks if t.get("status") == "in_progress"]

    def fmt(t: dict) -> str:
        meta   = t.get("metadata", {}) or {}
        deps   = t.get("depends_on", []) or meta.get("depends_on", [])
        dep_ok = "✓" if all(d in done_ids for d in deps) else "✗"
        return (f"[{t['id']}]P{t.get('priority','?')} "
                f"{str(t.get('required_skill','?'))[:4]} {dep_ok} "
                f"D{t.get('deadline', t.get('deadline_day','?'))}")

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

    memory_lines: list[str] = []
    in_prog_ids = [t["id"] for t in in_prog]
    if in_prog_ids:
        memory_lines.append(
            "NO_REASSIGN_UNTIL_BACKLOG: " + " ".join(in_prog_ids)
            + " — do NOT assign/reassign these while they stay in_progress."
        )
    if assigned_this_episode:
        memory_lines.append(
            "ASSIGNED_OK_THIS_EP: " + " ".join(sorted(assigned_this_episode))
            + " — already started; do not assign again."
        )
    if assign_attempted_episode:
        extra = assign_attempted_episode - set(assigned_this_episode or ())
        if extra:
            memory_lines.append(
                "ASSIGN_ALREADY_TRIED: " + " ".join(sorted(extra))
                + " — pick a different backlog task if still open."
            )
    memory_block = ("EPISODE_MEMORY:\n" + "\n".join(memory_lines) + "\n") if memory_lines else ""

    return (
        f"D{current_day}/60 S{current_sprint}/6 {days_left}d "
        f"done={obs.get('tasks_completed',0)} miss={obs.get('tasks_missed',0)} "
        f"inst={obs.get('instruction_following_score',0):.2f} debt={debt_count}\n"
        f"{inst_section}\n"
        f"BACKLOG(✓=deps_ok): {backlog_str}\n"
        f"IN_PROG: {inprog_str}\n"
        f"DEVS(avail): {devs_str}\n"
        f"{memory_block}"
        f"JSON:"
    )


def _completed_ids(obs: dict) -> set:
    return {t["id"] for t in obs.get("tasks", []) if t.get("status") == "done"}


def _deps_met_task(obs: dict, task: dict) -> bool:
    done_ids = _completed_ids(obs)
    meta = task.get("metadata", {}) or {}
    deps = task.get("depends_on", []) or meta.get("depends_on", [])
    return all(d in done_ids for d in deps)


def _dev_by_id(obs: dict, dev_id: object) -> Optional[dict]:
    sid = str(dev_id)
    for d in obs.get("developers", []):
        if str(d.get("id")) == sid:
            return d
    return None


def validate_llm_action(
    obs: dict,
    action: Optional[dict],
    assigned_this_episode: set,
) -> Tuple[bool, str]:
    """Hard gate: reject LLM/fallback output that violates env rules."""
    if not action:
        return False, "empty"
    at = action.get("action_type")
    if at not in {"assign", "reassign", "reprioritize", "unblock", "skip"}:
        return False, "bad_type"

    if at == "skip":
        return True, "ok"

    tid = action.get("task_id")
    if not tid:
        return False, "missing_task_id"

    by_id = {t["id"]: t for t in obs.get("tasks", [])}
    task = by_id.get(tid)
    if task is None:
        return False, "unknown_task"

    if at == "unblock":
        if task.get("status") != "blocked":
            return False, "unblock_not_blocked"
        if not _deps_met_task(obs, task):
            return False, "unblock_deps"
        return True, "ok"

    if at == "reprioritize":
        if task.get("status") != "backlog":
            return False, "reprioritize_not_backlog"
        if not _deps_met_task(obs, task):
            return False, "reprioritize_deps"
        np = action.get("new_priority")
        if np is None:
            return False, "reprioritize_no_priority"
        try:
            npi = int(np)
        except (TypeError, ValueError):
            return False, "reprioritize_bad_priority"
        if not (1 <= npi <= 5):
            return False, "reprioritize_range"
        return True, "ok"

    if at in ("assign", "reassign"):
        st = task.get("status")
        if st != "backlog":
            return False, f"assign_bad_status:{st}"
        if not _deps_met_task(obs, task):
            return False, "assign_deps"
        if tid in assigned_this_episode:
            return False, "assign_already_started_episode"
        did = action.get("dev_id")
        if not did:
            return False, "assign_no_dev"
        dev = _dev_by_id(obs, did)
        if dev is None:
            return False, "assign_unknown_dev"
        if not dev.get("is_available", False):
            return False, "assign_dev_unavailable"
        try:
            rem = int(dev.get("remaining_capacity", dev.get("capacity", 1)))
            if rem <= 0:
                return False, "assign_dev_no_capacity"
        except (TypeError, ValueError):
            pass
        skill = task.get("required_skill", "")
        dskill = dev.get("skill", "")
        if dskill not in (skill, "fullstack"):
            return False, "assign_skill_mismatch"
        return True, "ok"

    return False, "unhandled"


# ─── [FIX-I2] Local fine-tuned model loader ───────────────────────────────────

_local_model     = None
_local_tokenizer = None
_local_backend   = None


def _load_local_model(model_path: str) -> bool:
    global _local_model, _local_tokenizer, _local_backend
    if _local_model is not None:
        return True

    print(f"[INFO] Loading fine-tuned model: {model_path}", flush=True)

    unsloth_err: Optional[BaseException] = None
    try:
        from unsloth import FastLanguageModel
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_path, max_seq_length=2048, dtype=None,
            load_in_4bit=True, token=HF_TOKEN or None,
        )
        FastLanguageModel.for_inference(model)
        _local_model, _local_tokenizer, _local_backend = model, tokenizer, "unsloth"
        print("[INFO] Loaded via Unsloth (fast 4-bit inference).", flush=True)
        return True
    except Exception as e:
        unsloth_err = e
        print(f"[WARN] Unsloth failed: {e}", flush=True)

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from peft import PeftModel
        import huggingface_hub

        try:
            cfg_path = huggingface_hub.hf_hub_download(
                model_path, "adapter_config.json", token=HF_TOKEN or None
            )
        except Exception:
            cfg_path = os.path.join(model_path, "adapter_config.json")

        with open(cfg_path) as f:
            adapter_cfg = json.load(f)
        base_id = adapter_cfg.get("base_model_name_or_path", "Qwen/Qwen2.5-1.5B-Instruct")
        print(f"[INFO] Base model: {base_id}", flush=True)

        bnb  = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
        tok  = AutoTokenizer.from_pretrained(base_id, token=HF_TOKEN or None)
        base = AutoModelForCausalLM.from_pretrained(
            base_id, quantization_config=bnb, device_map="auto", token=HF_TOKEN or None
        )
        model = PeftModel.from_pretrained(base, model_path, token=HF_TOKEN or None)
        model.eval()
        _local_model, _local_tokenizer, _local_backend = model, tok, "peft"
        print("[INFO] Loaded via PEFT + bitsandbytes 4-bit.", flush=True)
        return True
    except Exception as e2:
        print(
            f"[ERROR] Cannot load local model.\n  Unsloth: {unsloth_err}\n  PEFT: {e2}",
            flush=True,
        )
        return False


def _call_local_model(user_prompt: str) -> Optional[dict]:
    """Run one inference step on the locally loaded fine-tuned model."""
    if _local_model is None:
        return None
    import torch

    messages = [
        {"role": "system", "content": R2_SYSTEM_PROMPT},
        {"role": "user",   "content": user_prompt},
    ]
    tok = _local_tokenizer
    try:
        if hasattr(tok, "apply_chat_template"):
            prompt_text = tok.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            prompt_text = "\n".join(
                f"<|{m['role']}|>\n{m['content']}" for m in messages
            ) + "\n<|assistant|>\n"

        inputs  = tok(prompt_text, return_tensors="pt").to(_local_model.device)
        inp_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            outputs = _local_model.generate(
                **inputs, max_new_tokens=MAX_TOKENS,
                temperature=TEMPERATURE, do_sample=True,
                pad_token_id=tok.pad_token_id or tok.eos_token_id,
            )

        completion = tok.decode(outputs[0][inp_len:], skip_special_tokens=True).strip()
        return parse_action(completion)

    except Exception as e:
        print(f"  [WARN] Local model inference error: {e}", flush=True)
        return None


# ─── HF Router client (fallback when no local model) ─────────────────────────

def _call_api_model(user_prompt: str) -> Optional[dict]:
    """Use HF Router — only for full (non-adapter) models."""
    try:
        from openai import OpenAI, APIStatusError
    except ImportError:
        return None
    if not HF_TOKEN:
        return None

    client   = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
    messages = [
        {"role": "system", "content": R2_SYSTEM_PROMPT},
        {"role": "user",   "content": user_prompt},
    ]
    for attempt in range(1, MAX_RETRIES + 2):
        try:
            resp = client.chat.completions.create(
                model=MODEL_NAME, messages=messages,
                max_tokens=MAX_TOKENS, temperature=TEMPERATURE,
            )
            return parse_action(resp.choices[0].message.content or "")
        except APIStatusError as e:
            if e.status_code not in RETRYABLE_CODES:
                return None
            if attempt <= MAX_RETRIES:
                time.sleep(2 ** attempt + random.uniform(0, 0.5))
            else:
                return None
        except Exception:
            if attempt <= MAX_RETRIES:
                time.sleep(2 ** attempt)
            else:
                return None
    return None


def call_llm_user_prompt(user_prompt: str) -> Optional[dict]:
    """Prefer local fine-tuned model; fall back to HF Router."""
    if _local_model is not None:
        return _call_local_model(user_prompt)
    return _call_api_model(user_prompt)


# ─── Action parser ────────────────────────────────────────────────────────────

def parse_action(raw: str) -> Optional[dict]:
    if not raw:
        return None
    raw = raw.strip()
    raw = re.sub(r"^```[a-z]*\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)

    # Find last balanced JSON object (handles CoT prefix the model might emit)
    depth = 0; obj_start = -1; last_start = -1; last_end = -1
    for i, ch in enumerate(raw):
        if ch == "{":
            if depth == 0: obj_start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and obj_start >= 0:
                last_start, last_end = obj_start, i + 1

    obj = None
    if last_start >= 0:
        try:
            obj = json.loads(raw[last_start:last_end])
        except json.JSONDecodeError:
            pass
    if obj is None:
        try:
            obj = json.loads(raw)
        except Exception:
            return None

    action_type = obj.get("action_type", "")
    if action_type not in {"assign", "reassign", "reprioritize", "unblock", "skip"}:
        return None

    null_vals = {"null", "none", "None", "Null", "", "undefined", "N/A", "nil"}
    for key in ("task_id", "dev_id", "new_priority"):
        v = obj.get(key)
        if v is not None and str(v).strip() in null_vals:
            obj[key] = None

    task_id = obj.get("task_id")
    if task_id is not None and not TASK_ID_RE.match(str(task_id)):
        print(f"  [INVALID] task_id={repr(task_id)} rejected", flush=True)
        return None

    if action_type in {"assign", "reassign"}:
        if not task_id or not obj.get("dev_id"):
            return None

    return {
        "action_type":  action_type,
        "task_id":      task_id,
        "dev_id":       obj.get("dev_id"),
        "new_priority": obj.get("new_priority"),
    }


# Aliases for evaluate_r2 / train naming
_build_r2_prompt = build_user_prompt
_parse_action = parse_action


# ─── [FIX-R7] Instruction text parser ────────────────────────────────────────

def _parse_instruction_assignment(inst_text: str, obs: dict) -> Optional[dict]:
    """
    [FIX-R7] Extract explicit task→dev assignment from instruction text.
    Handles patterns like:
      "Assign T3 to D2", "Use D4 for T5", "T2 → D1", "prioritize T6 assign to D3"
    Returns a validated assign action or None.
    """
    task_match = re.search(r'\b(T\d+)\b', inst_text)
    dev_match  = re.search(r'\b(D\d+)\b', inst_text)
    if not task_match or not dev_match:
        return None

    tid = task_match.group(1)
    did = dev_match.group(1)

    by_id = {t["id"]: t for t in obs.get("tasks", [])}
    task  = by_id.get(tid)
    if not task or task.get("status") != "backlog":
        return None
    if not _deps_met_task(obs, task):
        return None

    dev = _dev_by_id(obs, did)
    if not dev or not dev.get("is_available", False):
        # Try any available dev with the right skill as a fallback
        return None

    skill  = task.get("required_skill", "")
    dskill = dev.get("skill", "")
    if dskill not in (skill, "fullstack"):
        return None

    return {"action_type": "assign", "task_id": tid, "dev_id": did, "new_priority": None}


# ─── [FIX-R2/R3/R4/R5/R6] Smart fallback ─────────────────────────────────────

def _find_best_dev(
    task: dict,
    available_devs: list,
    assigned_this_episode: Optional[set] = None,
) -> Optional[dict]:
    """
    [FIX-R4/R5] Pick the best developer for a task.
    Priority: exact-skill match > fullstack.
    Within each group: sort by remaining capacity descending (most free first).
    Filters out devs with no remaining capacity.
    """
    skill = task.get("required_skill", "")

    capable = [
        d for d in available_devs
        if d.get("skill") in (skill, "fullstack")
        and d.get("is_available", False)
    ]
    if not capable:
        return None

    # Sort: exact skill first, then by remaining capacity descending
    def dev_sort_key(d: dict) -> tuple:
        exact   = 0 if d.get("skill") == skill else 1
        rem_cap = -int(d.get("remaining_capacity", d.get("capacity", 1)))
        return (exact, rem_cap)

    capable.sort(key=dev_sort_key)
    return capable[0]


def smart_fallback(
    obs: dict,
    assigned_this_episode: set,
    last_dev_idx: list,           # kept for API compat, no longer used for selection
) -> dict:
    """
    Tiered rule-based fallback. Fixed priority:

    Tier 0: Parse active instruction text for explicit T→D assignments [FIX-R7]
    Tier 1: Unblock blocked tasks whose deps are met
    Tier 2: ASSIGN (not just reprioritize) active instruction tasks [FIX-R2/R3]
    Tier 3: ASSIGN any backlog task in priority order [FIX-R3/R4/R6]
    Tier 4: Skip
    """
    tasks          = obs.get("tasks", [])
    devs           = obs.get("developers", [])
    instructions   = obs.get("instruction_queue", [])
    current_sprint = obs.get("current_sprint", 1)
    completed_ids  = {t["id"] for t in tasks if t.get("status") == "done"}

    skip = {"action_type": "skip", "task_id": None, "dev_id": None, "new_priority": None}

    def deps_met(task: dict) -> bool:
        meta = task.get("metadata", {}) or {}
        deps = task.get("depends_on", []) or meta.get("depends_on", [])
        return all(dep in completed_ids for dep in deps)

    # Available devs with capacity (strict), fall back to any available
    available_devs = [
        d for d in devs
        if d.get("is_available", False)
        and int(d.get("remaining_capacity", d.get("capacity", 1))) > 0
    ]
    if not available_devs:
        available_devs = [d for d in devs if d.get("is_available", False)]

    active_insts = [i for i in instructions if not i.get("followed", False)]

    # Instruction task IDs from all active instructions
    instruction_task_ids: set = set()
    for inst in active_insts:
        for tid in inst.get("affects_tasks", []):
            instruction_task_ids.add(tid)

    # ── Tier 0: parse instruction text for explicit T→D directives [FIX-R7] ──
    for inst in active_insts:
        action = _parse_instruction_assignment(inst.get("text", ""), obs)
        if action and action.get("task_id") not in assigned_this_episode:
            return action

    # ── Tier 1: unblock blocked tasks with met deps ────────────────────────────
    for task in tasks:
        if task.get("status") == "blocked" and deps_met(task):
            return {
                "action_type": "unblock",
                "task_id": task["id"],
                "dev_id": None,
                "new_priority": None,
            }

    # Helper: all backlog tasks eligible for assignment
    def assignable_tasks() -> list:
        return [
            t for t in tasks
            if t.get("status") == "backlog"
            and deps_met(t)
            and t["id"] not in assigned_this_episode   # [FIX-R6]
        ]

    # ── Tier 2: ASSIGN instruction tasks directly [FIX-R2] ────────────────────
    inst_backlog = sorted(
        [t for t in assignable_tasks() if t["id"] in instruction_task_ids],
        key=lambda t: t.get("priority", 9),
    )
    for task in inst_backlog:
        dev = _find_best_dev(task, available_devs)
        if dev:
            return {
                "action_type": "assign",
                "task_id": task["id"],
                "dev_id": dev["id"],
                "new_priority": None,
            }

    # ── Tier 3: assign any backlog task [FIX-R3/R4] ───────────────────────────
    def sort_key(t: dict) -> tuple:
        meta          = t.get("metadata", {}) or {}
        sprint_target = meta.get("sprint", current_sprint + 99)
        in_inst       = 0 if t["id"] in instruction_task_ids else 1
        return (in_inst, sprint_target, t.get("priority", 99), t.get("deadline", 99))

    all_assignable = sorted(assignable_tasks(), key=sort_key)
    for task in all_assignable:                      # [FIX-R3] iterate, don't just take [0]
        dev = _find_best_dev(task, available_devs)
        if dev:
            return {
                "action_type": "assign",
                "task_id": task["id"],
                "dev_id": dev["id"],
                "new_priority": None,
            }

    # ── Tier 4: skip ──────────────────────────────────────────────────────────
    return skip


# ─── Environment helpers ──────────────────────────────────────────────────────

def _post(url: str, payload: dict, timeout: int = 15) -> dict:
    resp = requests.post(url, json=payload, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def step_env(action: dict) -> dict:
    return _post(f"{ENV_BASE_URL}/project/step", {"action": action})


def reset_env(scenario: str, seed: int = 42) -> dict:
    return _post(f"{ENV_BASE_URL}/project/reset", {"task_name": scenario, "seed": seed})


def health() -> dict:
    resp = requests.get(f"{ENV_BASE_URL}/project/health", timeout=10)
    resp.raise_for_status()
    return resp.json()


# ─── Episode runner ───────────────────────────────────────────────────────────

def run_episode(scenario: str, seed: int = 42) -> dict:
    obs_data = reset_env(scenario, seed)
    obs      = obs_data.get("observation", obs_data)

    assigned_this_episode: set  = set()
    assign_attempted_episode: set = set()
    last_dev_idx = [0]
    cumulative   = 0.0
    step_num     = 0

    llm_skip_streak      = 0
    llm_soft_fail_streak = 0
    bad_assign_tid: Optional[str] = None
    bad_assign_streak    = 0
    llm_cooldown_until   = 0

    MAX_STEPS       = 200
    MAX_STEP_RETRIES = 3

    print(f"\n[START] task={scenario}", flush=True)

    while True:
        step_num += 1

        if step_num > MAX_STEPS:
            print("[WARN] Max steps reached — forcing termination", flush=True)
            break

        day    = obs.get("current_day", step_num)
        sprint = obs.get("current_sprint", 1)

        router_ok = _local_model is not None or bool(HF_TOKEN)
        allow_llm = (
            USE_LLM
            and (step_num % LLM_CALL_EVERY == 0)   # [FIX-R1] was == 1 (always False!)
            and router_ok
            and step_num > llm_cooldown_until
        )

        action = None

        if allow_llm:
            user_prompt = build_user_prompt(
                obs,
                assigned_this_episode=assigned_this_episode,
                assign_attempted_episode=assign_attempted_episode,
            )
            proposed = call_llm_user_prompt(user_prompt)

            if proposed is None:
                llm_soft_fail_streak += 1
                print("  [LLM] no parse / API fail → fallback", flush=True)
            else:
                ok, reason = validate_llm_action(obs, proposed, assigned_this_episode)
                if ok:
                    if proposed.get("action_type") == "skip":
                        llm_skip_streak += 1
                        if llm_skip_streak >= MAX_LLM_SKIP_STREAK:
                            action = None
                            llm_cooldown_until = max(
                                llm_cooldown_until, step_num + LLM_COOLDOWN_STEPS
                            )
                            llm_skip_streak = 0
                            llm_soft_fail_streak += 1
                            print(
                                "  [COOLDOWN] skip-spam — rule-based this step + "
                                f"{LLM_COOLDOWN_STEPS}-step LLM pause",
                                flush=True,
                            )
                        else:
                            action = proposed
                            llm_soft_fail_streak = 0
                            bad_assign_tid = None
                            bad_assign_streak = 0
                    else:
                        llm_skip_streak = 0
                        action = proposed
                        llm_soft_fail_streak = 0
                        bad_assign_tid = None
                        bad_assign_streak = 0
                else:
                    llm_soft_fail_streak += 1
                    print(
                        f"  [REJECT] LLM action invalid ({reason}) → fallback",
                        flush=True,
                    )
                    if proposed.get("action_type") in ("assign", "reassign"):
                        tidp    = proposed.get("task_id")
                        tid_key = str(tidp) if tidp is not None else None
                        if tid_key == bad_assign_tid:
                            bad_assign_streak += 1
                        else:
                            bad_assign_tid    = tid_key
                            bad_assign_streak = 1
                    else:
                        bad_assign_tid    = None
                        bad_assign_streak = 0

            if bad_assign_streak >= MAX_SAME_BAD_ASSIGN_STREAK:
                llm_cooldown_until = max(
                    llm_cooldown_until, step_num + LLM_COOLDOWN_STEPS
                )
                bad_assign_streak = 0
                bad_assign_tid    = None
                print(
                    f"  [COOLDOWN] repeated bad assign → rule-based for {LLM_COOLDOWN_STEPS} steps",
                    flush=True,
                )

            if llm_soft_fail_streak >= MAX_LLM_SOFT_FAIL_STREAK:
                llm_cooldown_until = max(
                    llm_cooldown_until, step_num + LLM_COOLDOWN_STEPS
                )
                llm_soft_fail_streak = 0
                print(
                    f"  [COOLDOWN] repeated LLM invalid → rule-based for {LLM_COOLDOWN_STEPS} steps",
                    flush=True,
                )

        # ── [FIX-R8] Validate fallback output before sending ──────────────────
        if action is None:
            fallback = smart_fallback(obs, assigned_this_episode, last_dev_idx)
            fb_ok, fb_reason = validate_llm_action(obs, fallback, assigned_this_episode)
            if fb_ok:
                action = fallback
            else:
                # Fallback produced an invalid action — log and force skip
                print(
                    f"  [FALLBACK_INVALID] {fb_reason} — forcing skip",
                    flush=True,
                )
                action = {"action_type": "skip", "task_id": None,
                          "dev_id": None, "new_priority": None}

        if action.get("action_type") in ("assign", "reassign") and action.get("task_id"):
            assign_attempted_episode.add(action["task_id"])

        # ── STEP WITH RETRY + STATE VALIDATION ───────────────────────────────
        success = False

        for attempt in range(MAX_STEP_RETRIES):
            result  = step_env(action)
            new_obs = result.get("observation", result)

            if (
                new_obs.get("current_day", 0) < obs.get("current_day", 0) or
                new_obs.get("current_sprint", 0) < obs.get("current_sprint", 0)
            ):
                print(
                    f"[ERROR] State regression detected (attempt {attempt+1}) — retrying",
                    flush=True,
                )
                time.sleep(0.5)
                continue

            success = True
            break

        if not success:
            print("[FATAL] Repeated environment corruption — aborting episode", flush=True)
            break

        reward     = result.get("reward", 0.0)
        obs        = new_obs
        done       = result.get("done", obs.get("done", False))
        cumulative += reward

        inst_score = obs.get("instruction_following_score", 0.0)
        debt_raw   = obs.get("tech_debt", 0)

        if action.get("action_type") in ("assign", "reassign") and action.get("task_id"):
            tid_sent = action["task_id"]
            post_s   = {t["id"]: t.get("status") for t in obs.get("tasks", [])}
            if post_s.get(tid_sent) == "in_progress":
                assigned_this_episode.add(tid_sent)

        debt_d = len(debt_raw) if isinstance(debt_raw, list) else int(debt_raw or 0)

        print(
            f"[STEP] task={scenario} step={step_num} day={day} sprint={sprint} "
            f"action={action.get('action_type','?')} "
            f"task_id={action.get('task_id','None')} "
            f"dev={action.get('dev_id','None')} "
            f"reward={reward:.4f} cumulative={cumulative:.4f} "
            f"inst_score={inst_score:.3f} debt={debt_d} done={done}",
            flush=True,
        )

        if done:
            break

    # ─── FINAL METRICS ───────────────────────────────────────────────────────
    tasks      = obs.get("tasks", [])
    completed  = sum(1 for t in tasks if t.get("status") == "done")
    missed     = sum(1 for t in tasks if t.get("status") == "missed")
    inst_score = obs.get("instruction_following_score", 0.0)

    debt_raw   = obs.get("tech_debt", 0)
    debt_count = len(debt_raw) if isinstance(debt_raw, list) else int(debt_raw or 0)

    total      = len(tasks) or 1

    final_score = max(0.01, min(0.99,
        (completed / total) * 0.55 +
        inst_score * 0.30 +
        max(0.01, 1.0 - debt_count * 0.02) * 0.15
    ))

    print(
        f"[END] task={scenario} score={final_score:.4f} steps={step_num} "
        f"completed={completed} missed={missed} "
        f"inst_score={inst_score:.3f} debt={debt_count}",
        flush=True,
    )

    return {
        "scenario":   scenario,
        "score":      final_score,
        "completed":  completed,
        "missed":     missed,
        "inst_score": inst_score,
        "debt":       debt_count,
        "steps":      step_num,
    }


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    scenarios = ["project_easy", "project_medium", "project_hard"]

    local_ok = False
    if LOCAL_MODEL_PATH:
        local_ok = _load_local_model(LOCAL_MODEL_PATH)
        if not local_ok:
            print("[WARN] Local model load failed — running rule-based fallback only.", flush=True)

    if not USE_LLM:
        mode = "rule-based-only (USE_LLM=0)"
    else:
        mode = ("local-finetuned" if local_ok
                else ("hf-router-base" if HF_TOKEN else "rule-based-only"))

    print(f"[INFO] model={LOCAL_MODEL_PATH if local_ok else MODEL_NAME}", flush=True)
    print(f"[INFO] inference mode={mode}", flush=True)
    print(f"[INFO] USE_LLM={USE_LLM} cooldown_steps={LLM_COOLDOWN_STEPS}", flush=True)
    print(f"[INFO] server={ENV_BASE_URL}", flush=True)

    try:
        print(f"[INFO] health={health()}", flush=True)
    except Exception as e:
        print(f"[WARN] health check failed: {e}", flush=True)

    results = {}
    t0 = time.time()
    for scenario in scenarios:
        try:
            results[scenario] = run_episode(scenario)
        except Exception as e:
            print(f"[ERROR] {scenario}: {e}", flush=True)
            results[scenario] = {"score": 0.01, "error": str(e)}

    scores = [results[s].get("score", 0) for s in scenarios if s in results]
    avg    = sum(scores) / len(scores) if scores else 0

    print("\n" + "=" * 62, flush=True)
    print(f" ROUND 2 — SCORES  [{mode}]", flush=True)
    print("=" * 62, flush=True)
    for s in scenarios:
        sc = results.get(s, {}).get("score", 0)
        print(f"  {s:<22} {sc:.4f}  {'█' * int(sc * 20)}", flush=True)
    print(f"\n  AVERAGE                {avg:.4f}", flush=True)
    print(f"\n  Runtime: {time.time()-t0:.1f}s", flush=True)
    print("=" * 62, flush=True)


if __name__ == "__main__":
    main()
