"""
inference.py — Round 1 LLM agent for AI Sprint Manager
=======================================================
Loads the trained model LOCALLY (Unsloth 4-bit → HF+PEFT fallback).
The HF Router API returns 400 "model not supported" for fine-tuned
private models — local loading is required.

Environment variables:
  HF_TOKEN          : HuggingFace token (read access to the model repo).
  LOCAL_MODEL_PATH  : HF repo ID or local path of the trained model.
                      Default: sejal-k/multi-sprint-model
  ENV_BASE_URL      : Sprint env server. Default: HF Space URL.
  USE_LLM           : Set to "0" to run rule-based only (default "1").
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from typing import Optional

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
    "  unblock: unblock a blocked task\n"
    "  skip: do nothing\n"
    "Output ONLY the JSON object. No markdown, no explanation."
)

# ── Local model state ─────────────────────────────────────────────────────────
_local_model     = None
_local_tokenizer = None


def load_local_model(model_path: str) -> bool:
    """
    Load the fine-tuned model locally.
    Tries Unsloth 4-bit first (fastest on T4), falls back to HF + PEFT.
    Mirrors the loader in inference_r2.py exactly.
    """
    global _local_model, _local_tokenizer
    if _local_model is not None:
        return True

    print(f"[INFO] Loading model locally: {model_path}", flush=True)

    # ── Try Unsloth (preferred — 2× faster inference on T4) ───────────────────
    try:
        from unsloth import FastLanguageModel
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_path,
            max_seq_length=2048,
            dtype=None,
            load_in_4bit=True,
            token=HF_TOKEN or None,
        )
        FastLanguageModel.for_inference(model)
        _local_model, _local_tokenizer = model, tokenizer
        print("[INFO] ✅ Loaded via Unsloth (4-bit inference mode)", flush=True)
        return True
    except Exception as e:
        print(f"[WARN] Unsloth load failed: {e}", flush=True)

    # ── Fallback: HF Transformers in fp16 ────────────────────────────────────
    # Works for a full merged checkpoint (not an adapter-only repo).
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_path, token=HF_TOKEN or None)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            token=HF_TOKEN or None,
        )
        model.eval()
        _local_model, _local_tokenizer = model, tokenizer
        print("[INFO] ✅ Loaded via HF Transformers (fp16, device_map=auto)", flush=True)
        return True
    except Exception as e2:
        print(f"[ERROR] HF Transformers load also failed: {e2}", flush=True)
        return False


def call_local_model(obs: dict) -> Optional[str]:
    """Run a forward pass through the local model and return raw text."""
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
        f"  [{t['id']}] {t.get('name','?')} | P{t.get('priority','?')} | "
        f"effort={t.get('effort','?')} | due=Day{t.get('deadline','?')} | "
        f"status={t.get('status','?')} | dev={t.get('assigned_to','none')} | "
        f"progress={t.get('progress', 0):.0%}"
        for t in obs.get("tasks", [])
    )
    devs_summary = "\n".join(
        f"  [{d['id']}] {d.get('name','?')} | skill={d.get('skill','?')} | "
        f"load={d.get('current_load',0)}/{d.get('capacity',5)} | "
        f"available={d.get('is_available',False)}"
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

def parse_action(text: str) -> dict:
    """Extract JSON action from LLM output, handling markdown fences."""
    _skip = {"action_type": "skip", "task_id": None, "dev_id": None, "new_priority": None}
    if not text:
        return _skip

    text = text.strip()
    text = re.sub(r"^```[a-z]*\s*", "", text)
    text = re.sub(r"\s*```$",       "", text)

    # Find the last balanced {...} block
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

    null_vals = {"null", "none", "None", "Null", "", "undefined", "N/A", "nil"}
    for key in ("task_id", "dev_id", "new_priority"):
        v = d.get(key)
        if v is not None and str(v).strip() in null_vals:
            d[key] = None

    if action_type in ("assign", "reassign") and (not d.get("task_id") or not d.get("dev_id")):
        action_type = "skip"

    return {
        "action_type":  action_type,
        "task_id":      d.get("task_id"),
        "dev_id":       d.get("dev_id"),
        "new_priority": d.get("new_priority"),
    }


# ── Rule-based fallback ───────────────────────────────────────────────────────

def rule_based_fallback(obs: dict) -> dict:
    """Best-effort rule-based action when LLM is unavailable or disabled."""
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
    dev     = matched[0] if matched else (avail[0] if avail else None)

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
    raw    = done / total - missed / total * 0.3
    return round(max(0.01, min(0.99, raw)), 4)


# ── Episode runner ────────────────────────────────────────────────────────────

def run_episode(task_name: str) -> float:
    print(f"\n[START] task={task_name}", flush=True)
    obs        = call_env("reset", {"task_name": task_name, "seed": 42})
    final_score = 0.01
    step_num    = 0

    for step_num in range(1, MAX_STEPS + 1):
        if obs.get("done", False):
            break

        # ── Choose action ──────────────────────────────────────────────────────
        if USE_LLM and _local_model is not None:
            raw_text = call_local_model(obs)
            if raw_text:
                action = parse_action(raw_text)
                # Demote to fallback if LLM produced a structurally bad action
                if action["action_type"] in ("assign", "reassign"):
                    if not action.get("task_id") or not action.get("dev_id"):
                        print("  [WARN] LLM bad action → fallback", flush=True)
                        action = rule_based_fallback(obs)
            else:
                print("  [WARN] LLM returned nothing → fallback", flush=True)
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
            print("       Make sure the model is pushed correctly to HF Hub "
                  "and HF_TOKEN is set with read access.", flush=True)
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