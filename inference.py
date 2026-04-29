"""
inference.py — Round 1 LLM agent for AI Sprint Manager
=======================================================
Loads the trained model LOCALLY via Unsloth (fast 4-bit) or HF Transformers.
The HF Router returns 400 "model not supported" for fine-tuned private models.

Environment variables:
  HF_TOKEN          : HuggingFace token (read access to the model repo).
  LOCAL_MODEL_PATH  : HF repo ID or local path. Default: sejal-k/multi-sprint-model
  ENV_BASE_URL      : Sprint env server. Default: HF Space URL.
  USE_LLM           : "0" to run rule-based only (default "1").
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
import warnings
from typing import Optional

# ── Silence harmless deprecation warnings in inference logs ──────────────────
warnings.filterwarnings("ignore", message=".*max_new_tokens.*max_length.*")
warnings.filterwarnings("ignore", message=".*AttentionMaskConverter.*")
warnings.filterwarnings("ignore", message=".*attention mask API.*")
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")
warnings.filterwarnings("ignore", category=UserWarning, message=".*max_new_tokens.*")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

import requests

# ── Config ────────────────────────────────────────────────────────────────────
LOCAL_MODEL_PATH = os.getenv("LOCAL_MODEL_PATH", "sejal-k/multi-sprint-model")
HF_TOKEN         = os.getenv("HF_TOKEN", "")
ENV_BASE_URL     = os.getenv("ENV_BASE_URL", "https://sejal-k-ai-sprint-manager.hf.space")

_use_llm_raw = os.getenv("USE_LLM", "1").strip().lower()
USE_LLM      = _use_llm_raw not in ("0", "false", "no", "off")

MAX_STEPS   = 12
TEMPERATURE = 0.3
MAX_TOKENS  = 96
TASKS       = ["easy_sprint", "medium_sprint", "hard_sprint"]

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
    "  unblock: unblock a BLOCKED task (not backlog)\n"
    "  skip: do nothing\n"
    "Output ONLY the JSON object. No markdown, no explanation."
)

# ── Local model state ─────────────────────────────────────────────────────────
_local_model     = None
_local_tokenizer = None
_generation_cfg  = None   # cached GenerationConfig with max_length cleared


def _patch_generation_config(model):
    """Remove max_length from generation_config so max_new_tokens has no conflict."""
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

    # Try Unsloth first (2× faster on T4)
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

    # Fallback: HF Transformers fp16
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
        print(f"[ERROR] Both loaders failed. Last error: {e2}", flush=True)
        return False


def call_local_model(obs: dict) -> Optional[str]:
    if _local_model is None:
        return None
    import torch
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": build_user_prompt(obs)},
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


# ── Prompt builder ────────────────────────────────────────────────────────────

def build_user_prompt(obs: dict) -> str:
    tasks_summary = "\n".join(
        f"  [{t['id']}] P{t.get('priority','?')} | effort={t.get('effort','?')} "
        f"| due=Day{t.get('deadline','?')} | status={t.get('status','?')} "
        f"| dev={t.get('assigned_to','none')}"
        for t in obs.get("tasks", [])
    )
    devs_summary = "\n".join(
        f"  [{d['id']}] skill={d.get('skill','?')} "
        f"| load={d.get('current_load',0)}/{d.get('capacity',5)} "
        f"| available={d.get('is_available',False)}"
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


# ── Action parser ─────────────────────────────────────────────────────────────

_VALID_TASK_ID = re.compile(r"^T\d+$")

def _clean_id(val: Optional[str]) -> Optional[str]:
    """
    Extract bare task/dev ID from whatever the model outputs.
    Handles '[T1] User Login API' → 'T1', 'T01' → 'T01', etc.
    """
    if val is None:
        return None
    s = str(val).strip()
    if s.lower() in {"null", "none", "", "undefined", "n/a", "nil"}:
        return None
    # Extract first T\d+ or dev\d+ pattern
    m = re.search(r'\b(T\d+)\b', s)
    if m:
        return m.group(1)
    m = re.search(r'\b(dev\d+)\b', s, re.IGNORECASE)
    if m:
        return m.group(1).lower()
    # Return raw if nothing matches — env will reject, fallback will catch
    return s if s else None


def parse_action(text: str) -> dict:
    _skip = {"action_type": "skip", "task_id": None, "dev_id": None, "new_priority": None}
    if not text:
        return _skip

    text = text.strip()
    text = re.sub(r"^```[a-z]*\s*", "", text)
    text = re.sub(r"\s*```$",       "", text)

    # Find last balanced {...} block
    depth = 0; obj_start = -1; last_start = -1; last_end = -1
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                obj_start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and obj_start >= 0:
                last_start, last_end = obj_start, i + 1

    d = None
    if last_start >= 0:
        try:
            d = json.loads(text[last_start:last_end])
        except json.JSONDecodeError:
            pass
    if d is None:
        try:
            d = json.loads(text)
        except Exception:
            return _skip

    if isinstance(d, list):
        d = d[0] if d and isinstance(d[0], dict) else None
    if not isinstance(d, dict):
        return _skip

    action_type = str(d.get("action_type", "skip")).lower().strip()
    if action_type not in {"assign", "reassign", "reprioritize", "unblock", "skip"}:
        action_type = "skip"

    task_id = _clean_id(d.get("task_id"))
    dev_id  = _clean_id(d.get("dev_id"))

    if action_type in ("assign", "reassign") and (not task_id or not dev_id):
        action_type = "skip"
    if action_type == "unblock" and not task_id:
        action_type = "skip"

    np_ = d.get("new_priority")
    if np_ is not None:
        try:
            np_ = int(np_)
            np_ = np_ if 1 <= np_ <= 5 else None
        except (ValueError, TypeError):
            np_ = None

    return {"action_type": action_type, "task_id": task_id,
            "dev_id": dev_id, "new_priority": np_}


# ── Rule-based fallback ───────────────────────────────────────────────────────

def _validate_action_against_obs(action: dict, obs: dict) -> tuple[bool, str]:
    """
    Hard semantic check against live observation state.
    Returns (is_valid, reason). Invalid actions are sent to rule_based_fallback.
    This guards against the trained model's reward-hacking patterns:
    - 'unblock' on a task that isn't blocked
    - 'assign'/'reassign' on a task that isn't backlog
    - referencing nonexistent task/dev IDs
    """
    at  = action.get("action_type", "skip")
    tid = action.get("task_id")
    did = action.get("dev_id")

    if at == "skip":
        return True, "ok"

    tasks_by_id = {t["id"]: t for t in obs.get("tasks", [])}
    devs_by_id  = {d["id"]: d for d in obs.get("developers", [])}

    if tid and tid not in tasks_by_id:
        return False, f"unknown_task:{tid}"

    task = tasks_by_id.get(tid) if tid else None

    if at == "unblock":
        if task is None:
            return False, "unblock_no_task"
        if task.get("status") != "blocked":
            return False, f"unblock_not_blocked:{task.get('status')}"
        return True, "ok"

    if at in ("assign", "reassign"):
        if task is None:
            return False, "assign_no_task"
        if task.get("status") != "backlog":
            return False, f"assign_bad_status:{task.get('status')}"
        if did and did not in devs_by_id:
            return False, f"unknown_dev:{did}"
        dev = devs_by_id.get(did) if did else None
        if dev and not dev.get("is_available", False):
            return False, "dev_unavailable"
        return True, "ok"

    if at == "reprioritize":
        if task is None:
            return False, "reprio_no_task"
        return True, "ok"

    return True, "ok"


# ── Rule-based fallback ───────────────────────────────────────────────────────

def rule_based_fallback(obs: dict) -> dict:
    """
    Deterministic fallback action: assign the highest-priority backlog task
    to the best available developer. Returns skip if nothing is actionable.
    """
    tasks   = obs.get("tasks", [])
    devs    = obs.get("developers", [])
    backlog = sorted(
        [t for t in tasks if t.get("status") == "backlog"],
        key=lambda t: (t.get("priority", 9), t.get("deadline", 99))
    )
    if not backlog:
        return {"action_type": "skip", "task_id": None, "dev_id": None, "new_priority": None}
    task  = backlog[0]
    skill = task.get("required_skill", "")
    avail = [d for d in devs
             if d.get("is_available", False)
             and d.get("current_load", 0) < d.get("capacity", 5)]
    matched = [d for d in avail if d.get("skill") == skill or d.get("skill") == "fullstack"]
    dev = matched[0] if matched else (avail[0] if avail else None)
    if not dev:
        return {"action_type": "skip", "task_id": None, "dev_id": None, "new_priority": None}
    return {"action_type": "assign", "task_id": task["id"],
            "dev_id": dev["id"], "new_priority": None}


# ── Environment helpers ───────────────────────────────────────────────────────

def call_env(endpoint: str, payload: dict = None, method: str = "POST") -> dict:
    url = f"{ENV_BASE_URL}/{endpoint}"
    if method == "GET":
        resp = requests.get(url, timeout=30)
    else:
        resp = requests.post(url, json=payload or {}, timeout=30)
    resp.raise_for_status()
    return resp.json()


def compute_final_score(obs: dict, info: dict) -> float:
    if "final_score" in info:
        return max(0.01, min(0.99, float(info["final_score"])))
    tasks  = obs.get("tasks", [])
    total  = len(tasks) or 1
    done   = sum(1 for t in tasks if t.get("status") == "done")
    missed = sum(1 for t in tasks if t.get("status") == "missed")
    return round(max(0.01, min(0.99, done / total - missed / total * 0.3)), 4)


# ── Episode runner ────────────────────────────────────────────────────────────

def run_episode(task_name: str) -> float:
    print(f"\n[START] task={task_name}", flush=True)
    obs         = call_env("reset", {"task_name": task_name, "seed": 42})
    final_score = 0.01
    step_num    = 0

    for step_num in range(1, MAX_STEPS + 1):
        if obs.get("done", False):
            break

        if USE_LLM and _local_model is not None:
            raw = call_local_model(obs)
            if raw:
                action = parse_action(raw)
                ok, reason = _validate_action_against_obs(action, obs)
                if not ok:
                    print(f"  [GUARD] model action invalid ({reason}) → fallback",
                          flush=True)
                    action = rule_based_fallback(obs)
            else:
                action = rule_based_fallback(obs)
        else:
            action = rule_based_fallback(obs)

        result = call_env("step", {"action": action})
        obs    = result.get("observation", result)
        reward = result.get("reward", 0.0)
        done   = result.get("done", False)
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

    if step_num == MAX_STEPS and final_score == 0.01:
        final_score = compute_final_score(obs, {})

    tasks_done   = sum(1 for t in obs.get("tasks", []) if t.get("status") == "done")
    tasks_missed = sum(1 for t in obs.get("tasks", []) if t.get("status") == "missed")
    print(
        f"[END] task={task_name} score={final_score:.4f} steps={step_num} "
        f"completed={tasks_done} missed={tasks_missed}",
        flush=True,
    )
    return final_score


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    mode = "rule-based-only"
    if USE_LLM:
        ok = load_local_model(LOCAL_MODEL_PATH)
        if ok:
            mode = f"local-model ({LOCAL_MODEL_PATH})"
        else:
            print("[WARN] Local model load failed — running rule-based only.", flush=True)
    else:
        print("[INFO] USE_LLM=0 — running rule-based only", flush=True)

    print(f"[INFO] mode={mode}", flush=True)
    print(f"[INFO] server={ENV_BASE_URL}", flush=True)

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
            scores[task] = run_episode(task)
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