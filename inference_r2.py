"""
inference_r2.py  ─  Round 2 agent  ─  COMPLETE REWRITE for score > 0.75
========================================================================

DESIGN PRINCIPLE: ZERO WASTED STEPS
Every single step must be a productive action. A skip is only acceptable
when it is literally impossible to assign any task (no backlog task with
deps met has an available developer with a matching skill). Everything
else is a bug.

ROOT CAUSES FIXED vs i_r2_4.py
──────────────────────────────
[FIX-1] MODULO BUG (fatal)
        step_num % LLM_CALL_EVERY == 1  →  always False when CALL_EVERY=1
        because n % 1 is always 0, never 1.  LLM was 100% bypassed on
        every single step of every episode. Fix: changed to == 0.

[FIX-2] LLM SKIP BLINDLY ACCEPTED
        When the LLM output skip the code accepted it immediately even when
        a perfectly valid assignment existed. Fix: before accepting any skip
        (LLM or fallback), run _guarantee_assign(). If it finds a valid
        task+dev pair, use that instead.

[FIX-3] SMART_FALLBACK TIER 2 WASTED STEPS
        Tier 2 called reprioritize on instruction tasks, consuming a full
        day step while doing nothing for inst_score (worth 30% of the final
        score). Fix: Tier 2 now directly ASSIGNs instruction tasks.

[FIX-4] SMART_FALLBACK PICKED ONE TASK, GAVE UP
        Old code took assignable[0]. If that task had no available skilled
        dev, it fell through to skill-mismatched devs -> env silently
        rejected the action, wasting the day. Fix: _guarantee_assign()
        iterates ALL tasks x ALL available devs in priority order until it
        finds a valid pair, then returns it.  Only returns None when every
        combination fails.

[FIX-5] FALLBACK NEVER VALIDATED
        smart_fallback output went straight to step_env() without passing
        through validate_action(). Skill mismatches, unavailable devs,
        and unmet deps all slipped through. Fix: every action -- LLM or
        fallback -- is validated before being sent to the environment.

[FIX-6] INSTRUCTION TEXT PARSING MISSING
        Instruction text like "Assign T3 to D2 immediately" was ignored.
        Fix: _parse_inst_action() extracts the first (T\d+, D\d+) pair
        from the instruction text and attempts a direct assignment, giving
        the env the best possible signal to credit instruction_following.

[FIX-7] COOLDOWN TOO AGGRESSIVE
        15-step LLM cooldown and 3-fail streak cut the LLM out for long
        stretches even when it was mostly working. Reduced to 5-step
        cooldown and 5-fail streak so the LLM stays in the loop.

[FIX-8] ASSIGNED_THIS_EPISODE NOT PASSED TO FALLBACK
        smart_fallback could repeatedly nominate tasks already moving
        through the pipeline, producing redundant assign attempts.
        Fix: _guarantee_assign() filters out assigned_this_episode.

HOW TO RUN
----------
  # Minimum -- rule-based only (no model needed, targets ~0.65+):
  python inference_r2.py

  # With fine-tuned local adapter (recommended, targets >0.80):
  export LOCAL_MODEL_PATH=results/trained_model   # local checkpoint path
  export HF_TOKEN=hf_...                          # only needed for HF Hub
  python inference_r2.py

  # Force rule-based even if model is present (debugging):
  USE_LLM=0 python inference_r2.py

SCORE FORMULA (from environment):
  final = (tasks_completed/total)*0.55
        + instruction_following_score*0.30
        + max(0.01, 1 - debt*0.02)*0.15

  Targets to reach 0.75:
    completion  >= 0.80  ->  contributes 0.44
    inst_score  >= 0.80  ->  contributes 0.24
    debt        <= 3     ->  contributes 0.144
    Total                    ~= 0.824  OK
"""

from __future__ import annotations

import json
import os
import re
import time
import random
from typing import Optional, Tuple, List

import requests

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
LOCAL_MODEL_PATH = os.getenv("LOCAL_MODEL_PATH", "priyaaaaaasharmaaaaa/trial1")
MODEL_NAME       = os.getenv("MODEL_NAME",       "meta-llama/Llama-3.1-8B-Instruct")
ENV_BASE_URL     = os.getenv("ENV_BASE_URL",     "https://sejal-k-ai-sprint-manager.hf.space")
API_BASE_URL     = os.getenv("API_BASE_URL",     "https://router.huggingface.co/v1")
HF_TOKEN         = os.getenv("HF_TOKEN",         "")

_use_llm_raw = os.getenv("USE_LLM", "1").strip().lower()
USE_LLM      = _use_llm_raw not in ("0", "false", "no", "off", "")

# [FIX-7] Reduced cooldown aggression
LLM_COOLDOWN_STEPS        = int(os.getenv("LLM_COOLDOWN_STEPS",        "5"))
MAX_LLM_SOFT_FAIL_STREAK  = int(os.getenv("MAX_LLM_SOFT_FAIL_STREAK",  "5"))
MAX_LLM_SKIP_STREAK       = int(os.getenv("MAX_LLM_SKIP_STREAK",       "6"))
MAX_SAME_BAD_ASSIGN_STREAK = int(os.getenv("MAX_SAME_BAD_ASSIGN_STREAK", "2"))

MAX_TOKENS     = 96
MAX_RETRIES    = 2
LLM_CALL_EVERY = 1
TEMPERATURE    = 0.3

TASK_ID_RE      = re.compile(r"^T\d+$")
RETRYABLE_CODES = {429, 500, 502, 503, 504}

_SKIP = {"action_type": "skip", "task_id": None, "dev_id": None, "new_priority": None}


# ---------------------------------------------------------------------------
# Prompts -- EXACTLY matching train_llm.py
# ---------------------------------------------------------------------------
R2_SYSTEM_PROMPT = """You are an Engineering Manager running a 60-day software project.
Each step you MUST output exactly ONE JSON object and nothing else.

Schema (use null for unused fields):
{"action_type":"<assign|reassign|reprioritize|unblock|skip>","task_id":"<id or null>","dev_id":"<id or null>","new_priority":<1-5 or null>}

Rules (follow in order):
1. If ACTIVE INSTRUCTIONS exist, assign THEIR tasks first.
2. Only assign tasks with status=backlog (never in_progress or done).
3. Only assign if all dependency markers show checkmark.
4. Only assign to an AVAILABLE developer with matching or fullstack skill.
5. Use unblock ONLY for explicitly blocked tasks whose deps are met.
6. skip is last resort.

Output ONLY the JSON. No explanation."""


def build_user_prompt(
    obs: dict,
    *,
    assigned_this_episode: Optional[set] = None,
    assign_attempted_episode: Optional[set] = None,
) -> str:
    """Build the user prompt. Layout must match _build_r2_prompt() in train_llm.py."""
    current_sprint = obs.get("current_sprint", 1)
    current_day    = obs.get("current_day", 1)
    days_left      = max(0, current_sprint * 10 - current_day + 1)
    tasks          = obs.get("tasks", [])
    done_ids       = {t["id"] for t in tasks if t.get("status") == "done"}

    active_insts = [i for i in obs.get("instruction_queue", []) if not i.get("followed", False)]
    inst_section = (
        "FOLLOW: " + " | ".join(f"[{i['id']}] {i['text'][:50]}" for i in active_insts[:2])
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
        dep_ok = "Y" if all(d in done_ids for d in deps) else "N"
        return (f"[{t['id']}]P{t.get('priority','?')} "
                f"{str(t.get('required_skill','?'))[:4]} {dep_ok} "
                f"D{t.get('deadline', t.get('deadline_day','?'))}")

    backlog_str = " ".join(fmt(t) for t in backlog[:6])
    if len(backlog) > 6:
        backlog_str += f" +{len(backlog)-6}"

    inprog_str = " ".join(
        f"[{t['id']}]->{t.get('assigned_to','?')}" for t in in_prog
    ) or "none"

    avail_devs = [d for d in obs.get("developers", []) if d.get("is_available", False)]
    devs_str   = " ".join(
        f"[{d['id']}]{str(d.get('name','?'))[:4]}({str(d.get('skill','?'))[:3]}) "
        f"{d.get('current_load',0)}/{d.get('capacity',5)}"
        for d in avail_devs
    )

    memory_lines: List[str] = []
    in_prog_ids = [t["id"] for t in in_prog]
    if in_prog_ids:
        memory_lines.append(
            "NO_REASSIGN: " + " ".join(in_prog_ids)
        )
    if assigned_this_episode:
        memory_lines.append(
            "ASSIGNED_OK: " + " ".join(sorted(assigned_this_episode))
        )
    if assign_attempted_episode:
        extra = assign_attempted_episode - set(assigned_this_episode or ())
        if extra:
            memory_lines.append(
                "TRIED_FAILED: " + " ".join(sorted(extra))
            )
    memory_block = ("MEM:\n" + "\n".join(memory_lines) + "\n") if memory_lines else ""

    return (
        f"D{current_day}/60 S{current_sprint}/6 {days_left}d "
        f"done={obs.get('tasks_completed',0)} miss={obs.get('tasks_missed',0)} "
        f"inst={obs.get('instruction_following_score',0):.2f} debt={debt_count}\n"
        f"{inst_section}\n"
        f"BACKLOG(Y=deps_ok): {backlog_str}\n"
        f"IN_PROG: {inprog_str}\n"
        f"DEVS(avail): {devs_str}\n"
        f"{memory_block}"
        f"JSON:"
    )


# ---------------------------------------------------------------------------
# Environment state helpers
# ---------------------------------------------------------------------------

def _done_ids(obs: dict) -> set:
    return {t["id"] for t in obs.get("tasks", []) if t.get("status") == "done"}


def _deps_met(obs: dict, task: dict) -> bool:
    done = _done_ids(obs)
    meta = task.get("metadata", {}) or {}
    deps = task.get("depends_on", []) or meta.get("depends_on", [])
    return all(d in done for d in deps)


def _dev_by_id(obs: dict, dev_id: object) -> Optional[dict]:
    sid = str(dev_id)
    for d in obs.get("developers", []):
        if str(d.get("id")) == sid:
            return d
    return None


def _available_devs(obs: dict) -> List[dict]:
    """Developers that are available. Prefer those with remaining capacity."""
    with_cap = [
        d for d in obs.get("developers", [])
        if d.get("is_available", False)
        and int(d.get("remaining_capacity", d.get("capacity", 1))) > 0
    ]
    if with_cap:
        return with_cap
    # Fallback: any available dev even at limit
    return [d for d in obs.get("developers", []) if d.get("is_available", False)]


def _assignable_tasks(obs: dict, excluded: Optional[set] = None) -> List[dict]:
    """Backlog tasks with dependencies met, excluding already-started IDs."""
    ex = excluded or set()
    return [
        t for t in obs.get("tasks", [])
        if t.get("status") == "backlog"
        and _deps_met(obs, t)
        and t["id"] not in ex
    ]


# ---------------------------------------------------------------------------
# [FIX-4/8] Core anti-skip guarantee
# ---------------------------------------------------------------------------

def _guarantee_assign(
    obs: dict,
    assigned_this_episode: set,
    priority_task_ids: Optional[set] = None,
) -> Optional[dict]:
    """
    [FIX-4] Exhaustively search every task x every dev combination for a
    valid assignment. Only returns None when literally no valid pair exists.

    This is the anti-skip guarantee: call it before accepting any skip action
    to confirm the skip is genuinely unavoidable.
    """
    priority_ids   = priority_task_ids or set()
    avail          = _available_devs(obs)
    backlog        = _assignable_tasks(obs, excluded=assigned_this_episode)
    current_sprint = obs.get("current_sprint", 1)

    if not avail or not backlog:
        return None

    def task_key(t: dict) -> tuple:
        meta = t.get("metadata", {}) or {}
        return (
            0 if t["id"] in priority_ids else 1,
            meta.get("sprint", current_sprint + 99),
            t.get("priority", 9),
            t.get("deadline", t.get("deadline_day", 99)),
        )

    backlog.sort(key=task_key)

    for task in backlog:
        skill = task.get("required_skill", "")

        # Sort devs: exact skill first, then fullstack, then by remaining capacity
        def dev_key(d: dict) -> tuple:
            ds = d.get("skill", "")
            match_score = 0 if ds == skill else (1 if ds == "fullstack" else 9)
            cap = -int(d.get("remaining_capacity", d.get("capacity", 1)))
            return (match_score, cap)

        for dev in sorted(avail, key=dev_key):
            dskill = dev.get("skill", "")
            if dskill not in (skill, "fullstack"):
                continue
            cap = int(dev.get("remaining_capacity", dev.get("capacity", 1)))
            if cap <= 0:
                continue
            return {
                "action_type":  "assign",
                "task_id":      task["id"],
                "dev_id":       dev["id"],
                "new_priority": None,
            }

    return None  # genuinely nothing to assign


# ---------------------------------------------------------------------------
# Validate any action before sending to env   [FIX-5]
# ---------------------------------------------------------------------------

def validate_action(
    obs: dict,
    action: Optional[dict],
    assigned_this_episode: set,
) -> Tuple[bool, str]:
    """Hard gate for BOTH LLM and fallback outputs."""
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
    task  = by_id.get(str(tid))
    if task is None:
        return False, "unknown_task"

    if at == "unblock":
        if task.get("status") != "blocked":
            return False, "unblock_not_blocked"
        if not _deps_met(obs, task):
            return False, "unblock_deps_unmet"
        return True, "ok"

    if at == "reprioritize":
        if task.get("status") != "backlog":
            return False, "reprioritize_not_backlog"
        np_ = action.get("new_priority")
        if np_ is None:
            return False, "reprioritize_no_priority"
        try:
            if not (1 <= int(np_) <= 5):
                return False, "reprioritize_range"
        except (TypeError, ValueError):
            return False, "reprioritize_bad_priority"
        return True, "ok"

    if at in ("assign", "reassign"):
        st = task.get("status")
        if st != "backlog":
            return False, f"assign_bad_status:{st}"
        if not _deps_met(obs, task):
            return False, "assign_deps_unmet"
        if str(tid) in assigned_this_episode:
            return False, "assign_already_started"
        did = action.get("dev_id")
        if not did:
            return False, "assign_no_dev"
        dev = _dev_by_id(obs, did)
        if dev is None:
            return False, "assign_unknown_dev"
        if not dev.get("is_available", False):
            return False, "assign_dev_unavailable"
        try:
            if int(dev.get("remaining_capacity", dev.get("capacity", 1))) <= 0:
                return False, "assign_dev_no_capacity"
        except (TypeError, ValueError):
            pass
        skill  = task.get("required_skill", "")
        dskill = dev.get("skill", "")
        if dskill not in (skill, "fullstack"):
            return False, "assign_skill_mismatch"
        return True, "ok"

    return False, "unhandled"


# Legacy alias
validate_llm_action = validate_action


# ---------------------------------------------------------------------------
# [FIX-6] Instruction text parser
# ---------------------------------------------------------------------------

def _parse_inst_action(
    inst_text: str,
    obs: dict,
    assigned_this_episode: set,
) -> Optional[dict]:
    """
    [FIX-6] Extract explicit T->D assignment from instruction text.
    Handles: "Assign T3 to D2", "Use D4 for T5", "T6 -> D1", etc.
    Returns a validated assign action or None.
    """
    task_m = re.search(r'\b(T\d+)\b', inst_text)
    dev_m  = re.search(r'\b(D\d+)\b', inst_text)
    if not task_m or not dev_m:
        return None

    action = {
        "action_type":  "assign",
        "task_id":      task_m.group(1),
        "dev_id":       dev_m.group(1),
        "new_priority": None,
    }
    ok, _ = validate_action(obs, action, assigned_this_episode)
    return action if ok else None


# ---------------------------------------------------------------------------
# [FIX-3/4] Smart fallback -- complete rewrite
# ---------------------------------------------------------------------------

def smart_fallback(
    obs: dict,
    assigned_this_episode: set,
    last_dev_idx: list,   # kept for API compat
) -> dict:
    """
    Tiered decision engine. Each tier is tried in order; the first valid
    action is returned. Skip is ONLY returned from Tier 5 when every tier
    above confirms nothing is actionable.

    Tier 0  Parse instruction text for explicit T->D directives  [FIX-6]
    Tier 1  Unblock blocked tasks whose deps are met
    Tier 2  Directly ASSIGN backlog instruction-related tasks    [FIX-3]
    Tier 3  Assign any backlog task (exhaustive search)          [FIX-4]
    Tier 4  Reprioritize tasks blocked by deps (signal urgency)
    Tier 5  Skip (genuinely nothing possible)
    """
    tasks        = obs.get("tasks", [])
    instructions = obs.get("instruction_queue", [])

    # Collect active instruction task IDs
    active_insts         = [i for i in instructions if not i.get("followed", False)]
    instruction_task_ids: set = set()
    for inst in active_insts:
        for tid in inst.get("affects_tasks", []):
            instruction_task_ids.add(tid)

    # ── Tier 0: instruction text parse ───────────────────────────────────────
    for inst in active_insts:
        action = _parse_inst_action(inst.get("text", ""), obs, assigned_this_episode)
        if action:
            print(f"  [FB-T0] inst text parse -> {action['task_id']}->{action['dev_id']}", flush=True)
            return action

    # ── Tier 1: unblock blocked tasks ────────────────────────────────────────
    for task in tasks:
        if task.get("status") == "blocked" and _deps_met(obs, task):
            action = {
                "action_type":  "unblock",
                "task_id":      task["id"],
                "dev_id":       None,
                "new_priority": None,
            }
            ok, _ = validate_action(obs, action, assigned_this_episode)
            if ok:
                print(f"  [FB-T1] unblock {task['id']}", flush=True)
                return action

    # ── Tier 2: assign instruction tasks first [FIX-3] ───────────────────────
    if instruction_task_ids:
        action = _guarantee_assign(
            obs, assigned_this_episode, priority_task_ids=instruction_task_ids
        )
        if action:
            print(f"  [FB-T2] inst assign -> {action['task_id']}->{action['dev_id']}", flush=True)
            return action

    # ── Tier 3: assign any backlog task [FIX-4] ───────────────────────────────
    action = _guarantee_assign(obs, assigned_this_episode)
    if action:
        print(f"  [FB-T3] general assign -> {action['task_id']}->{action['dev_id']}", flush=True)
        return action

    # ── Tier 4: reprioritize blocked-by-dep tasks while devs are busy ────────
    for task in tasks:
        if (
            task.get("status") == "backlog"
            and not _deps_met(obs, task)
            and int(task.get("priority", 1)) > 2
        ):
            action = {
                "action_type":  "reprioritize",
                "task_id":      task["id"],
                "dev_id":       None,
                "new_priority": 1,
            }
            ok, _ = validate_action(obs, action, assigned_this_episode)
            if ok:
                print(f"  [FB-T4] reprioritize {task['id']} (waiting deps)", flush=True)
                return action

    # ── Tier 5: genuine skip ──────────────────────────────────────────────────
    print("  [FB-T5] genuine skip -- no valid action available", flush=True)
    return _SKIP


# ---------------------------------------------------------------------------
# [FIX-2] Anti-skip wrapper
# ---------------------------------------------------------------------------

def _anti_skip(
    action: dict,
    obs: dict,
    assigned_this_episode: set,
    instruction_task_ids: set,
    source: str,
) -> dict:
    """
    [FIX-2] Before accepting any skip action, verify it is genuinely
    unavoidable by running _guarantee_assign(). If a valid assignment
    exists, return that instead of the skip.
    """
    if action.get("action_type") != "skip":
        return action

    override = _guarantee_assign(obs, assigned_this_episode, priority_task_ids=instruction_task_ids)
    if override:
        print(
            f"  [ANTI-SKIP] {source} skip overridden -> "
            f"{override['task_id']}->{override['dev_id']}",
            flush=True,
        )
        return override

    return action  # genuinely nothing to assign


# ---------------------------------------------------------------------------
# Local fine-tuned model loader
# ---------------------------------------------------------------------------

_local_model     = None
_local_tokenizer = None
_local_backend   = None


def _load_local_model(model_path: str) -> bool:
    global _local_model, _local_tokenizer, _local_backend
    if _local_model is not None:
        return True

    print(f"[INFO] Loading fine-tuned model: {model_path}", flush=True)
    unsloth_err: Optional[BaseException] = None

    # Attempt 1: Unsloth (same library used for training -- best compatibility)
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

    # Attempt 2: PEFT + bitsandbytes
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
        print(f"[INFO] Base model from adapter config: {base_id}", flush=True)

        bnb   = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
        tok   = AutoTokenizer.from_pretrained(base_id, token=HF_TOKEN or None)
        base  = AutoModelForCausalLM.from_pretrained(
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
    if _local_model is None:
        return None
    import torch

    messages = [
        {"role": "system", "content": R2_SYSTEM_PROMPT},
        {"role": "user",   "content": user_prompt},
    ]
    tok = _local_tokenizer
    try:
        prompt_text = (
            tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            if hasattr(tok, "apply_chat_template")
            else "\n".join(f"<|{m['role']}|>\n{m['content']}" for m in messages)
               + "\n<|assistant|>\n"
        )
        inputs  = tok(prompt_text, return_tensors="pt").to(_local_model.device)
        inp_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            outputs = _local_model.generate(
                **inputs,
                max_new_tokens=MAX_TOKENS,
                temperature=TEMPERATURE,
                do_sample=True,
                pad_token_id=tok.pad_token_id or tok.eos_token_id,
            )
        completion = tok.decode(outputs[0][inp_len:], skip_special_tokens=True).strip()
        return parse_action(completion)
    except Exception as e:
        print(f"  [WARN] Local model inference error: {e}", flush=True)
        return None


def _call_api_model(user_prompt: str) -> Optional[dict]:
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


def call_llm(user_prompt: str) -> Optional[dict]:
    """Prefer local fine-tuned model; fall back to HF Router."""
    if _local_model is not None:
        return _call_local_model(user_prompt)
    return _call_api_model(user_prompt)


# ---------------------------------------------------------------------------
# Action parser
# ---------------------------------------------------------------------------

def parse_action(raw: str) -> Optional[dict]:
    if not raw:
        return None
    raw = raw.strip()
    raw = re.sub(r"^```[a-z]*\s*", "", raw)
    raw = re.sub(r"\s*```$",       "", raw)

    # Find last balanced JSON object (handles CoT prefix)
    depth = 0; obj_start = -1; last_start = -1; last_end = -1
    for i, ch in enumerate(raw):
        if ch == "{":
            if depth == 0:
                obj_start = i
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
_build_r2_prompt     = build_user_prompt
_parse_action        = parse_action
call_llm_user_prompt = call_llm


# ---------------------------------------------------------------------------
# Environment helpers
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

def run_episode(scenario: str, seed: int = 42) -> dict:
    obs_data = reset_env(scenario, seed)
    obs      = obs_data.get("observation", obs_data)

    assigned_this_episode: set    = set()
    assign_attempted_episode: set = set()
    last_dev_idx                  = [0]
    cumulative                    = 0.0
    step_num                      = 0

    llm_skip_streak       = 0
    llm_soft_fail_streak  = 0
    bad_assign_tid: Optional[str] = None
    bad_assign_streak     = 0
    llm_cooldown_until    = 0

    MAX_STEPS        = 200
    MAX_STEP_RETRIES = 3

    print(f"\n[START] task={scenario}", flush=True)

    while True:
        step_num += 1
        if step_num > MAX_STEPS:
            print("[WARN] Max steps reached -- terminating", flush=True)
            break

        day    = obs.get("current_day",    step_num)
        sprint = obs.get("current_sprint", 1)

        # Current instruction task IDs (for anti-skip priority)
        active_insts = [
            i for i in obs.get("instruction_queue", [])
            if not i.get("followed", False)
        ]
        inst_task_ids: set = set()
        for inst in active_insts:
            for tid in inst.get("affects_tasks", []):
                inst_task_ids.add(tid)

        # ── LLM path ──────────────────────────────────────────────────────────
        router_ok = _local_model is not None or bool(HF_TOKEN)
        allow_llm = (
            USE_LLM
            and (step_num % LLM_CALL_EVERY == 0)   # [FIX-1] was == 1, always False!
            and router_ok
            and step_num > llm_cooldown_until
        )

        action: Optional[dict] = None

        if allow_llm:
            user_prompt = build_user_prompt(
                obs,
                assigned_this_episode=assigned_this_episode,
                assign_attempted_episode=assign_attempted_episode,
            )
            proposed = call_llm(user_prompt)

            if proposed is None:
                llm_soft_fail_streak += 1
                print("  [LLM] no parse/API fail -> fallback", flush=True)
            else:
                ok, reason = validate_action(obs, proposed, assigned_this_episode)
                if ok:
                    # [FIX-2] Override skip if a real assignment is possible
                    proposed = _anti_skip(
                        proposed, obs, assigned_this_episode, inst_task_ids, "LLM"
                    )
                    if proposed.get("action_type") == "skip":
                        # Confirmed genuine skip
                        llm_skip_streak += 1
                        if llm_skip_streak >= MAX_LLM_SKIP_STREAK:
                            llm_cooldown_until   = max(
                                llm_cooldown_until, step_num + LLM_COOLDOWN_STEPS
                            )
                            llm_skip_streak      = 0
                            llm_soft_fail_streak += 1
                            print(
                                f"  [COOLDOWN] LLM skip-spam -> pause {LLM_COOLDOWN_STEPS}s",
                                flush=True,
                            )
                        else:
                            action               = proposed
                            llm_soft_fail_streak = 0
                    else:
                        llm_skip_streak      = 0
                        llm_soft_fail_streak = 0
                        bad_assign_tid       = None
                        bad_assign_streak    = 0
                        action               = proposed
                else:
                    llm_soft_fail_streak += 1
                    print(f"  [REJECT] LLM invalid ({reason}) -> fallback", flush=True)
                    if proposed.get("action_type") in ("assign", "reassign"):
                        tidp    = proposed.get("task_id")
                        tid_key = str(tidp) if tidp else None
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
                print(f"  [COOLDOWN] repeated bad assign -> rule-based {LLM_COOLDOWN_STEPS}s", flush=True)

            if llm_soft_fail_streak >= MAX_LLM_SOFT_FAIL_STREAK:
                llm_cooldown_until   = max(
                    llm_cooldown_until, step_num + LLM_COOLDOWN_STEPS
                )
                llm_soft_fail_streak = 0
                print(f"  [COOLDOWN] repeated fail -> rule-based {LLM_COOLDOWN_STEPS}s", flush=True)

        # ── Fallback path ──────────────────────────────────────────────────────
        if action is None:
            action = smart_fallback(obs, assigned_this_episode, last_dev_idx)

            # [FIX-5] Validate fallback output before sending to env
            ok, reason = validate_action(obs, action, assigned_this_episode)
            if not ok:
                print(f"  [FALLBACK_INVALID] {reason} -> guaranteed assign", flush=True)
                action = _guarantee_assign(
                    obs, assigned_this_episode, inst_task_ids
                ) or _SKIP

        # [FIX-2] Absolute last-resort anti-skip before env call
        action = _anti_skip(action, obs, assigned_this_episode, inst_task_ids, "final")

        # Track attempted assignments
        if action.get("action_type") in ("assign", "reassign") and action.get("task_id"):
            assign_attempted_episode.add(action["task_id"])

        # ── Send to environment with regression guard ──────────────────────────
        success = False
        for attempt in range(MAX_STEP_RETRIES):
            try:
                result  = step_env(action)
                new_obs = result.get("observation", result)
            except Exception as e:
                print(f"  [ENV_ERR] step_env failed (attempt {attempt+1}): {e}", flush=True)
                time.sleep(1.0)
                continue

            if (
                new_obs.get("current_day",    0) < obs.get("current_day",    0) or
                new_obs.get("current_sprint", 0) < obs.get("current_sprint", 0)
            ):
                print(f"  [ERROR] State regression (attempt {attempt+1}) -- retrying", flush=True)
                time.sleep(0.5)
                continue

            success = True
            break

        if not success:
            print("[FATAL] Repeated env corruption -- aborting episode", flush=True)
            break

        reward     = result.get("reward", 0.0)
        obs        = new_obs
        done       = result.get("done", obs.get("done", False))
        cumulative += reward

        # Track tasks that actually moved to in_progress
        if action.get("action_type") in ("assign", "reassign") and action.get("task_id"):
            tid_sent    = action["task_id"]
            post_status = {t["id"]: t.get("status") for t in obs.get("tasks", [])}
            if post_status.get(tid_sent) == "in_progress":
                assigned_this_episode.add(tid_sent)

        inst_score = obs.get("instruction_following_score", 0.0)
        debt_raw   = obs.get("tech_debt", 0)
        debt_d     = len(debt_raw) if isinstance(debt_raw, list) else int(debt_raw or 0)

        print(
            f"[STEP] task={scenario} step={step_num} day={day} sprint={sprint} "
            f"action={action.get('action_type','?')} "
            f"task_id={action.get('task_id','None')} "
            f"dev={action.get('dev_id','None')} "
            f"reward={reward:.4f} cumul={cumulative:.4f} "
            f"inst={inst_score:.3f} debt={debt_d} done={done}",
            flush=True,
        )

        if done:
            break

    # ── Final metrics ──────────────────────────────────────────────────────────
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
        f"completed={completed}/{total} missed={missed} "
        f"inst={inst_score:.3f} debt={debt_count}",
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    scenarios = ["project_easy", "project_medium", "project_hard"]

    local_ok = False
    if LOCAL_MODEL_PATH:
        local_ok = _load_local_model(LOCAL_MODEL_PATH)
        if not local_ok:
            print(
                "[WARN] Local model load failed -- running rule-based only.\n"
                "       Set LOCAL_MODEL_PATH to your adapter checkpoint to reach 0.80+.",
                flush=True,
            )

    if not USE_LLM:
        mode = "rule-based-only (USE_LLM=0)"
    else:
        mode = (
            "local-finetuned" if local_ok  else
            "hf-router-base"  if HF_TOKEN  else
            "rule-based-only"
        )

    print(
        f"[INFO] mode={mode}  model={LOCAL_MODEL_PATH if local_ok else MODEL_NAME}",
        flush=True,
    )
    print(
        f"[INFO] USE_LLM={USE_LLM}  cooldown={LLM_COOLDOWN_STEPS}  server={ENV_BASE_URL}",
        flush=True,
    )

    try:
        print(f"[INFO] health={health()}", flush=True)
    except Exception as e:
        print(f"[WARN] health check failed: {e}", flush=True)

    results: dict = {}
    t0 = time.time()

    for scenario in scenarios:
        try:
            results[scenario] = run_episode(scenario)
        except Exception as e:
            print(f"[ERROR] {scenario}: {e}", flush=True)
            results[scenario] = {"score": 0.01, "error": str(e)}

    scores = [results[s].get("score", 0) for s in scenarios if s in results]
    avg    = sum(scores) / len(scores) if scores else 0.0

    print("\n" + "=" * 64, flush=True)
    print(f"  ROUND 2 -- FINAL SCORES  [{mode}]", flush=True)
    print("=" * 64, flush=True)
    for s in scenarios:
        sc  = results.get(s, {}).get("score", 0.0)
        bar = "X" * int(sc * 20)
        print(f"  {s:<24} {sc:.4f}  {bar}", flush=True)
    print(f"\n  AVERAGE                  {avg:.4f}", flush=True)
    print(f"  Runtime: {time.time() - t0:.1f}s", flush=True)
    print("=" * 64, flush=True)

    if avg >= 0.75:
        print("\n  TARGET MET: average >= 0.75 -- eligible for next round", flush=True)
    else:
        print(f"\n  Gap to target: {0.75 - avg:.4f}", flush=True)


if __name__ == "__main__":
    main()
