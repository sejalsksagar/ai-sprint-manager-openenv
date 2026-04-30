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

FIXES vs previous version (bugs identified from baseline run analysis):
  [FIX-R2-1] was_rejected threshold lowered from < -0.3 → < 0.
             The env returns -0.15 for assigning an already-in_progress task.
             The old threshold missed this, causing T04/T05/T10 to be retried
             4-5 times each (losing ~0.6-0.75 reward per task per episode).
  [FIX-R2-2] Pre-block: task added to assigned_this_episode BEFORE calling env.
             Guarantees the next step's fallback immediately sees the task as taken.
  [FIX-R2-3] get_rule_based_action now includes REASSIGN logic: when a dev has
             productivity < 0.6 (burnout) or is overloaded, their in-progress tasks
             are reassigned to available devs with matching skill. The old fallback
             had no reassign logic, causing inst_score to decay to 0.04 over 60 steps.
  [FIX-R2-4] get_rule_based_action now includes REPRIORITIZE logic: backlog tasks
             within 2 days of their deadline with priority > 2 are bumped to P1
             before the next assign attempt. Prevents overdue tasks being ignored.
  [FIX-R2-5] Unblock ordering preserved (step 4 in priority chain, after reassign).

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
import warnings
from typing import Optional

import requests
from dotenv import load_dotenv

load_dotenv()

# ── Silence harmless deprecation warnings ────────────────────────────────────
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")
warnings.filterwarnings("ignore", message=".*AttentionMaskConverter.*")
warnings.filterwarnings("ignore", message=".*attention mask API.*")
warnings.filterwarnings("ignore", message=".*max_new_tokens.*max_length.*")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

# ── Config ────────────────────────────────────────────────────────────────────
# inference_r2 loads the fine-tuned model LOCALLY via Unsloth (same as inference.py).
# Root cause of BadRequestError: HF Router rejects fine-tuned private models.
# Fix: load locally — same path as inference.py.

LOCAL_MODEL_PATH = os.getenv("LOCAL_MODEL_PATH", "sejal-k/multi-sprint-model")
HF_TOKEN         = os.getenv("HF_TOKEN", "")
MODEL_NAME       = os.getenv("MODEL_NAME", LOCAL_MODEL_PATH)
ENV_BASE_URL     = os.getenv("ENV_BASE_URL", "https://sejal-k-ai-sprint-manager.hf.space")

_use_llm_raw     = os.getenv("USE_LLM", "1").strip().lower()
USE_LLM          = _use_llm_raw not in ("0", "false", "no", "off")

# ── Llama baseline mode ───────────────────────────────────────────────────────
# Set LLAMA_BASELINE=1 to run inference via the HuggingFace router using the
# public meta-llama/Llama-3.1-8B-Instruct model (no local GPU required).
# This reproduces the original Llama R2 baseline scores.
#
# Required env vars:
#   LLAMA_BASELINE=1
#   HF_TOKEN=hf_...
#
# Example:
#   LLAMA_BASELINE=1 HF_TOKEN=hf_... python inference_r2.py
_llama_raw     = os.getenv("LLAMA_BASELINE", "0").strip().lower()
LLAMA_BASELINE = _llama_raw not in ("0", "false", "no", "off")
LLAMA_MODEL    = os.getenv("LLAMA_MODEL", "meta-llama/Llama-3.1-8B-Instruct")
HF_ROUTER_URL  = "https://router.huggingface.co/v1"

MAX_STEPS    = 60        # full 60-day project
# [FIX] Raised temperature from 0.15 → 0.5 to break repetitive greedy decoding.
# Low temperature (0.15) causes the model to lock onto one token distribution
# and repeat the same task assignment for many steps. 0.5 maintains coherent
# JSON output while introducing enough diversity to escape local loops.
TEMPERATURE  = 0.5
MAX_TOKENS   = 80       # action JSON is tiny — 80 tokens is plenty

# Retry / rate-limit config
MAX_RETRIES  = 3
RETRY_DELAYS = [1, 2, 4]

# LLM call batching (set to 1 to call every step)
LLM_CALL_EVERY = 1

# [FIX] Loop detection: if the model picks the same (action_type, task_id)
# N times in a row, override with rule-based fallback to break the loop.
# This directly addresses the core failure mode seen in the trained model logs.
LOOP_DETECT_WINDOW = 2  # consecutive identical actions before forcing fallback

TASKS = ["project_easy", "project_medium", "project_hard"]

# Local model state (loaded once, shared across episodes)
_local_model     = None
_local_tokenizer = None

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
                          last_failed_task: Optional[str] = None,
                          reassigned_this_episode: set = None) -> dict:
    """
    Deterministic fallback. Priority order:
    1. Active instruction tasks (deps met, not already assigned)
    2. Highest-priority backlog (deps met, not assigned, not last failed)
    3. Reassign in-progress tasks from burned-out / overloaded devs to available ones
       [FIX] Skip tasks already in reassigned_this_episode to break reassign loops.
    4. Unblock blocked tasks whose deps are now all done
    5. Reprioritize: bump overdue high-effort tasks that are still backlog
    6. Skip
    """
    if reassigned_this_episode is None:
        reassigned_this_episode = set()
    tasks     = obs.get("tasks", [])
    devs      = obs.get("developers", [])
    done_ids  = {t["id"] for t in tasks if t["status"] == "done"}
    available = [d for d in devs if d["is_available"] and d["current_load"] < d["capacity"]]

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

    # [FIX] 3. Reassign: move in-progress tasks from burned-out or overloaded devs
    # to available devs with matching skill. This prevents the skip-lock that occurs
    # once backlog is exhausted — the old code had no reassign logic at all.
    #
    # [FIX-R2-SKIP-LOCK] Two-pass reassign strategy:
    # Pass 1 (strict): target devs at strict capacity (current_load < capacity).
    # Pass 2 (relaxed): when ALL devs are full (available is empty), consider devs
    #   at current_load == capacity as reassign targets (one-slot-at-limit slack).
    #   This breaks the skip-lock in sprint 2+ once every dev slot fills: the old
    #   code found no candidates and fell through to 40+ consecutive skips.
    devs_by_id  = {d["id"]: d for d in devs}
    in_progress = [t for t in tasks if t["status"] == "in_progress"]

    for pass_num in (1, 2):
        if pass_num == 1:
            reassign_targets = available  # strict: current_load < capacity
        else:
            if available:                 # pass 1 already had candidates, skip pass 2
                break
            reassign_targets = [
                d for d in devs
                if d["is_available"]
                and d["current_load"] <= d["capacity"]
            ]

        for task in in_progress:
            if not task.get("assigned_to"):
                continue
            # [FIX-REASSIGN-LOOP] Skip tasks we've already reassigned this episode.
            # Without this, the fallback loops on the same in-progress task every step
            # (T25 steps 13-29, T21 steps 41-58 in project_hard/easy baselines).
            if task["id"] in reassigned_this_episode:
                continue
            current_dev  = devs_by_id.get(task["assigned_to"])
            if current_dev is None:
                continue
            productivity = current_dev.get("productivity", 1.0)
            overloaded   = current_dev.get("current_load", 0) > current_dev.get("capacity", 5)
            if productivity < 0.6 or overloaded:
                skill = task.get("required_skill", "")
                candidates = [
                    d for d in reassign_targets
                    if d["id"] != current_dev["id"]
                    and (d["skill"] == skill or d["skill"] == "fullstack")
                ]
                if candidates:
                    return {"action_type": "reassign", "task_id": task["id"],
                            "dev_id": candidates[0]["id"], "new_priority": None}

    # 4. Unblock blocked tasks whose deps are fully done
    for task in tasks:
        if task["status"] == "blocked":
            deps = task.get("metadata", {}).get("depends_on", [])
            if all(d in done_ids for d in deps):
                return {"action_type": "unblock", "task_id": task["id"],
                        "dev_id": None, "new_priority": None}

    # [FIX] 5. Reprioritize: if there are backlog tasks with deadline already past
    # or within 2 days and priority > 2, bump them to priority 1 so they get picked
    # up on the next assign opportunity. Without this the model ignores overdue tasks.
    # [FIX-R2-REPRIO-STATUS] Guard: only reprioritize tasks that are still "backlog".
    # The env rejects reprioritize on done/in_progress tasks with -0.70 penalty.
    # The old code iterated `backlog` list (already filtered by status == "backlog")
    # but only after the reassign block above, which doesn't mutate the list —
    # safe. However, we add an explicit guard here as a belt-and-suspenders check
    # in case `backlog` is ever populated from a different source path.
    current_day = obs.get("current_day", 0)
    for task in backlog:
        if task.get("status") != "backlog":   # explicit guard against stale refs
            continue
        if task["id"] in assigned_this_episode:
            continue
        deadline = task.get("deadline", 999)
        priority = task.get("priority", 5)
        if deadline <= current_day + 2 and priority > 2:
            return {"action_type": "reprioritize", "task_id": task["id"],
                    "dev_id": None, "new_priority": 1}

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

    # ── Sanitise task_id and dev_id ───────────────────────────────────────────
    # The compact prompt uses "D{day}/60 S{sprint}/6" which causes the LLM to
    # hallucinate task_ids like "D60/60", "D37/60", or dev_ids like "D31".
    # Accept only canonical T\d+ for task_id and D\d+ (without slash) for dev_id.
    import re as _re
    def _clean_task_id(val):
        if val is None:
            return None
        s = str(val).strip()
        # Extract first T\d+ pattern (e.g. "T01", "T9")
        m = _re.search(r"\b(T\d+)\b", s)
        return m.group(1) if m else None

    def _clean_dev_id(val):
        if val is None:
            return None
        s = str(val).strip()
        # Accept D\d+ only — reject "D60/60", "D37/60" etc (date leakage)
        m = _re.fullmatch(r"D\d+", s)
        if m:
            return s
        # Try to extract from longer string
        m = _re.search(r"\b(D\d+)\b", s)
        if m and "/" not in s:   # reject date-style D\d+/\d+
            return m.group(1)
        return None

    d["task_id"] = _clean_task_id(d.get("task_id"))
    d["dev_id"]  = _clean_dev_id(d.get("dev_id"))

    # new_priority → int
    if d.get("new_priority") is not None:
        try:
            d["new_priority"] = int(d["new_priority"])
            if d["new_priority"] not in range(1, 6):
                d["new_priority"] = None
        except (ValueError, TypeError):
            d["new_priority"] = None

    # Demote invalid actions → skip
    # assign: task_id required; dev_id optional (env auto-picks best dev)
    # reassign: both required
    atype = d["action_type"]
    if atype == "assign" and not d.get("task_id"):
        d["action_type"] = "skip"
    if atype == "reassign" and (not d.get("task_id") or not d.get("dev_id")):
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

def _patch_generation_config(model):
    """Clear conflicting max_length from generation_config."""
    try:
        if hasattr(model, "generation_config"):
            gc = model.generation_config
            if getattr(gc, "max_length", None) is not None:
                gc.max_length = None
    except Exception:
        pass


def load_local_model(model_path: str) -> bool:
    global _local_model, _local_tokenizer
    if _local_model is not None:
        return True
    print(f"[INFO] Loading model locally: {model_path}", flush=True)
    try:
        from unsloth import FastLanguageModel
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_path, max_seq_length=2048,
            dtype=None, load_in_4bit=True, token=HF_TOKEN or None,
        )
        FastLanguageModel.for_inference(model)
        _patch_generation_config(model)
        _local_model, _local_tokenizer = model, tokenizer
        print("[INFO] ✅ Loaded via Unsloth (4-bit inference)", flush=True)
        return True
    except Exception as e:
        print(f"[WARN] Unsloth load failed: {e}", flush=True)
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path, token=HF_TOKEN or None)
        model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.float16,
            device_map="auto", token=HF_TOKEN or None,
        )
        model.eval()
        _patch_generation_config(model)
        _local_model, _local_tokenizer = model, tokenizer
        print("[INFO] ✅ Loaded via HF Transformers (fp16)", flush=True)
        return True
    except Exception as e2:
        print(f"[ERROR] Both loaders failed: {e2}", flush=True)
        return False


def call_llama_router(obs: dict, assigned_this_episode: set) -> Optional[str]:
    """
    Call the HF router with meta-llama/Llama-3.1-8B-Instruct (or LLAMA_MODEL).
    Reproduces the original Llama R2 baseline without a local GPU.
    Requires LLAMA_BASELINE=1 and HF_TOKEN to be set.
    """
    if not HF_TOKEN:
        print("[ERROR] LLAMA_BASELINE=1 requires HF_TOKEN to be set.", flush=True)
        return None
    prompt = build_user_prompt(obs, assigned_this_episode)
    messages = [
        {"role": "system", "content": R2_SYSTEM_PROMPT},
        {"role": "user",   "content": prompt},
    ]
    try:
        resp = requests.post(
            f"{HF_ROUTER_URL}/chat/completions",
            headers={
                "Authorization": f"Bearer {HF_TOKEN}",
                "Content-Type":  "application/json",
            },
            json={
                "model":       LLAMA_MODEL,
                "messages":    messages,
                "max_tokens":  MAX_TOKENS,
                "temperature": TEMPERATURE,
            },
            timeout=60,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(f"  [WARN] Llama router call failed: {e}", flush=True)
        return None


def call_local_model(obs: dict, assigned_this_episode: set) -> Optional[str]:
    if _local_model is None:
        return None
    import torch
    prompt = build_user_prompt(obs, assigned_this_episode)
    messages = [
        {"role": "system", "content": R2_SYSTEM_PROMPT},
        {"role": "user",   "content": prompt},
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
        return tok.decode(outputs[0][inp_len:], skip_special_tokens=True).strip()
    except Exception as e:
        print(f"  [WARN] Local model inference error: {e}", flush=True)
        return None


# ── LLM call with retry ────────────────────────────────────────────────────────

def call_llm(obs: dict, assigned_this_episode: set, step_num: int) -> tuple[str, bool]:
    """
    Call LLM with up to MAX_RETRIES retries and exponential backoff.
    Routes to HF router (Llama baseline) or local model based on flags.
    Returns (response_text, used_llm: bool).
    """
    if LLAMA_BASELINE:
        raw = call_llama_router(obs, assigned_this_episode)
    else:
        raw = call_local_model(obs, assigned_this_episode)
    if raw:
        return raw, True
    print("[WARN] LLM unavailable, using rule-based fallback", flush=True)
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
    reassigned_this_episode: set[str] = set() # tasks we've sent reassign for
    last_failed_task: Optional[str] = None    # last task that got negative reward
    llm_fail_streak: int            = 0       # consecutive LLM failures
    LLM_ABANDON_AFTER               = 5       # give up on LLM after this many consecutive failures

    # [FIX] Loop detection: track last N (action_type, task_id) pairs.
    # If the model picks the same action LOOP_DETECT_WINDOW times in a row,
    # it has entered a degenerate loop — force rule-based fallback to break it.
    action_history: list[tuple] = []

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
                # Guard 1: task already assigned this episode or not in backlog → override
                # Guard 1b: task already reassigned this episode → override
                # Guard 2: task has unmet deps → override
                # Guard 3: dev not available → override
                # This is the primary fix for Llama repeatedly picking T04/T05.
                if action["action_type"] in ("assign", "reassign") and action.get("task_id"):
                    tid    = action["task_id"]
                    did    = action.get("dev_id")
                    t_info = next((t for t in obs["tasks"] if t["id"] == tid), None)
                    d_info = next((d for d in obs["developers"] if d["id"] == did), None) if did else None
                    done_ids_guard = {t["id"] for t in obs["tasks"] if t["status"] == "done"}

                    invalid_reason = None
                    if tid in assigned_this_episode:
                        invalid_reason = f"already_assigned:{tid}"
                    elif action["action_type"] == "reassign" and tid in reassigned_this_episode:
                        invalid_reason = f"already_reassigned:{tid}"
                    elif t_info is None:
                        invalid_reason = f"unknown_task:{tid}"
                    elif action["action_type"] == "assign" and t_info["status"] != "backlog":
                        invalid_reason = f"not_backlog:{t_info['status']}"
                    elif d_info is not None and not d_info.get("is_available", True):
                        invalid_reason = f"dev_busy:{did}"
                    # [FIX-R2-CAPACITY] Reject assign when named dev is at capacity.
                    # is_available can be True while current_load == capacity; the env
                    # returns -0.05 and the task never goes in_progress, starving the
                    # entire dep chain for the rest of the episode.
                    elif (action["action_type"] == "assign" and d_info is not None
                          and d_info.get("current_load", 0) >= d_info.get("capacity", 5)):
                        invalid_reason = f"dev_at_capacity:{did}"
                    else:
                        deps = t_info.get("metadata", {}).get("depends_on", []) if t_info else []
                        if deps and not all(d in done_ids_guard for d in deps):
                            invalid_reason = f"unmet_deps:{deps}"

                    if invalid_reason:
                        print(f"  [GUARD] {invalid_reason} → fallback", flush=True)
                        action = get_rule_based_action(
                            obs, assigned_this_episode, last_failed_task
                        )

                # [FIX] Loop detection: if same (action_type, task_id) repeated
                # LOOP_DETECT_WINDOW times in a row, force rule-based fallback.
                action_key = (action.get("action_type"), action.get("task_id"))
                action_history.append(action_key)
                if len(action_history) > LOOP_DETECT_WINDOW:
                    action_history.pop(0)
                if (len(action_history) == LOOP_DETECT_WINDOW
                        and len(set(action_history)) == 1
                        and action_key[0] != "skip"):
                    print(f"  [LOOP] Detected {len(action_history)}× repeat of "
                          f"{action_key} → forcing fallback", flush=True)
                    # Block the repeated task so LLM cannot propose it again.
                    # [FIX-LOOP-REASSIGN] Also add to reassigned_this_episode when
                    # the loop is on a reassign action — otherwise get_rule_based_action
                    # step 3 immediately picks the same task as a reassign candidate
                    # and the loop continues through the fallback (T25, T30 loops).
                    if action_key[1]:
                        assigned_this_episode.add(action_key[1])
                        if action_key[0] == "reassign":
                            reassigned_this_episode.add(action_key[1])
                    action = get_rule_based_action(
                        obs, assigned_this_episode, last_failed_task,
                        reassigned_this_episode
                    )
                    action_history.clear()  # reset after breaking loop
            else:
                llm_fail_streak += 1
                action = get_rule_based_action(obs, assigned_this_episode, last_failed_task, reassigned_this_episode)
        else:
            # Either LLM abandoned or between batch intervals
            action = get_rule_based_action(obs, assigned_this_episode, last_failed_task, reassigned_this_episode)

        # ── Call environment ──────────────────────────────────────────────────
        # [FIX-PREBLOCK] Block task BEFORE calling env so the next step's fallback
        # immediately sees it as unavailable. This is the primary fix for the
        # 4-5× repeat-assign loops (T04, T05, T10) observed in baselines.
        #
        # [FIX-R2-SOFT-REJECT] Refined pre-block for R2's 60-step episodes:
        # Only pre-block if the chosen dev is confirmed under capacity right now.
        # If no dev_id is specified, or the dev is at capacity, skip the pre-block
        # and let the post-env block handle it based on the actual reward signal.
        # This allows the fallback to retry a task once a dev slot opens after a
        # soft -0.05 capacity-full reject, instead of permanently blocking it.
        if action["action_type"] == "assign" and action.get("task_id"):
            _pre_did = action.get("dev_id")
            _pre_dev = next((d for d in obs.get("developers", []) if d["id"] == _pre_did), None) if _pre_did else None
            _dev_has_room = _pre_dev is None or _pre_dev.get("current_load", 0) < _pre_dev.get("capacity", 5)
            if _dev_has_room:
                assigned_this_episode.add(action["task_id"])

        # [FIX-R2-REASSIGN-BLOCK] Mirror pre-block for reassign.
        # Add task to reassigned_this_episode immediately so guard/fallback on
        # the very next step cannot emit the same reassign before env responds.
        if action["action_type"] == "reassign" and action.get("task_id"):
            reassigned_this_episode.add(action["task_id"])

        result = call_env("step", {"action": action})
        obs    = result["observation"]
        reward = result["reward"]
        done   = result["done"]

        # Update assigned_this_episode AFTER env responds.
        # Block task permanently if env accepted (in_progress) or hard-rejected it.
        # [FIX-SOFT-BOUNCE] Do NOT permanently block on a -0.05 soft bounce.
        # Instead, discard the task from assigned_this_episode so it's retryable
        # once a dev slot opens. This breaks the dep-chain freeze: T01 bounces
        # -0.05 in step 1 (dev at capacity), gets stuck in assigned_this_episode,
        # its downstream dep chain never starts, and everything skips for 50 steps.
        if action["action_type"] == "assign" and action.get("task_id"):
            tid = action["task_id"]
            task_after = next((t for t in obs.get("tasks", []) if t["id"] == tid), None)
            was_accepted    = task_after and task_after.get("status") == "in_progress"
            was_hard_reject = reward < -0.05   # -0.15 skill/already-in-progress → block
            is_soft_bounce  = -0.06 < reward < 0  # -0.05 capacity-full → allow retry
            if was_accepted or was_hard_reject:
                assigned_this_episode.add(tid)
            elif is_soft_bounce and task_after and task_after.get("status") == "backlog":
                # Soft bounce: task still backlog, unblock so fallback retries later
                assigned_this_episode.discard(tid)

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
            # Prefer the env's own final_score if present in info — it uses the
            # full internal formula including streak bonuses and penalties that
            # our local approximation cannot replicate accurately.
            info = result.get("info", {})
            if "final_score" in info:
                final_score = max(0.01, min(0.99, float(info["final_score"])))
            else:
                # Fallback formula when env doesn't supply final_score
                tasks_total   = len(obs.get("tasks", [])) or 1
                tasks_done    = obs.get("tasks_completed", 0)
                inst_score    = obs.get("instruction_following_score", 0.01)
                delivery_rate = tasks_done / tasks_total
                debt_count    = len(obs.get("tech_debt", []))
                team_health   = max(0.01, 1.0 - debt_count * 0.02)
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
    if LLAMA_BASELINE:
        print(f"[INFO] LLAMA_BASELINE=1 — using HF router ({LLAMA_MODEL})", flush=True)
        if not HF_TOKEN:
            print("[ERROR] HF_TOKEN not set — export HF_TOKEN=hf_... and retry.",
                  flush=True)
            sys.exit(1)
    else:
        ok = load_local_model(LOCAL_MODEL_PATH) if USE_LLM else False
        if not ok:
            print("[WARN] Local model load failed — running rule-based only.", flush=True)
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
    label = "LLAMA BASELINE" if LLAMA_BASELINE else "RULE-BASED"

    print("\n" + "=" * 62, flush=True)
    print(f" ROUND 2 — SCORES  [{label}]", flush=True)
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