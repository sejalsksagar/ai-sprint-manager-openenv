"""
evaluate_r2.py — Round 2 Evaluation Script
============================================
Compares baseline (rule-based) vs trained LLM on both R1 and R2 tasks.
Produces the before/after improvement table.
Saves results to results/r2_evaluation.json.

Usage:
    # Baseline only (no model needed):
    python evaluate_r2.py --baseline-only

    # Full comparison using the HF Hub trained model:
    python evaluate_r2.py --model sejal-k/ai-sprint-manager-trained

    # Full comparison using a local model directory:
    python evaluate_r2.py --model results/trained_model

    # Quick 1-episode-per-task run:
    python evaluate_r2.py --model sejal-k/ai-sprint-manager-trained --episodes 1

Fixes vs original:
  - Removed broken `from client import SprintEnvClient` /
    `from project_client import ProjectEnvClient` — replaced with plain requests.
  - run_r1_episode / run_r2_episode now use raw requests.post(), no custom client.
  - rule_based_r2: depends_on read from task top-level AND task.metadata.
  - build_llm_policy: local model loads with PEFT if LoRA adapter detected.
  - score_r2_obs: uses tasks_completed field if available (faster than counting).
  - No more .close() calls on non-existent client objects.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Callable, Optional

import requests

ENV_BASE_URL = os.getenv("ENV_BASE_URL", "https://sejal-k-ai-sprint-manager.hf.space")
HF_TOKEN     = os.getenv("HF_TOKEN", "")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME", "sejal-k/ai-sprint-manager-trained")

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

R1_TASKS = ["easy_sprint", "medium_sprint", "hard_sprint"]
R2_TASKS = ["project_easy", "project_medium", "project_hard"]

# ── Measured baselines (Llama-3.1-8B zero-shot) ──────────────────────────────
LLAMA_BASELINE_R1 = {
    "easy_sprint":   0.0100,
    "medium_sprint": 0.4583,
    "hard_sprint":   0.0100,
    "average":       0.1594,
}

LLAMA_BASELINE_R2 = {
    "project_easy":   0.3198,
    "project_medium": 0.2443,
    "project_hard":   0.2520,
    "average":        0.2720,
}

TRAINING_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"


# ── Env helpers (raw requests — no custom client class needed) ─────────────────

def _r1_reset(task_name: str, seed: int = 42) -> dict:
    r = requests.post(f"{ENV_BASE_URL}/reset", json={"task_name": task_name, "seed": seed}, timeout=30)
    r.raise_for_status()
    return r.json()


def _r1_step(action: dict) -> dict:
    r = requests.post(f"{ENV_BASE_URL}/step", json={"action": action}, timeout=30)
    r.raise_for_status()
    return r.json()


def _r2_reset(task_name: str, seed: int = 42) -> dict:
    r = requests.post(f"{ENV_BASE_URL}/project/reset", json={"task_name": task_name, "seed": seed}, timeout=30)
    r.raise_for_status()
    return r.json()


def _r2_step(action: dict) -> dict:
    r = requests.post(f"{ENV_BASE_URL}/project/step", json={"action": action}, timeout=30)
    r.raise_for_status()
    return r.json()


def _parse_llm_action(raw: str) -> dict:
    """Extract JSON action from LLM output, handling markdown fences."""
    import re
    if not raw:
        return {"action_type": "skip", "task_id": None, "dev_id": None, "new_priority": None}
    raw = raw.strip()
    raw = re.sub(r"^```[a-z]*\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    start = raw.find("{")
    end   = raw.rfind("}") + 1
    if start >= 0 and end > start:
        try:
            obj = json.loads(raw[start:end])
            # Normalise list-valued IDs (LLM bug protection)
            for key in ("task_id", "dev_id"):
                v = obj.get(key)
                if isinstance(v, list):
                    obj[key] = v[0] if v else None
            return obj
        except Exception:
            pass
    return {"action_type": "skip", "task_id": None, "dev_id": None, "new_priority": None}


# ── Rule-based policies ───────────────────────────────────────────────────────

def rule_based_r1(obs: dict) -> dict:
    tasks  = obs.get("tasks", [])
    devs   = obs.get("developers", [])
    avail  = [d for d in devs if d["is_available"] and d["current_load"] < d["capacity"]]
    backlog = sorted(
        [t for t in tasks if t["status"] == "backlog"],
        key=lambda t: (t["priority"], t["deadline"])
    )
    for task in backlog:
        match = [
            d for d in avail
            if d["skill"] == task.get("required_skill") or d["skill"] == "fullstack"
        ]
        dev = match[0] if match else (avail[0] if avail else None)
        if dev:
            return {"action_type": "assign", "task_id": task["id"],
                    "dev_id": dev["id"], "new_priority": None}
    return {"action_type": "skip", "task_id": None, "dev_id": None, "new_priority": None}


def _get_depends_on(task: dict) -> list:
    """Read depends_on from task top-level OR task.metadata (handles both schema versions)."""
    top = task.get("depends_on", [])
    if top:
        return top
    meta = task.get("metadata", {})
    if isinstance(meta, dict):
        return meta.get("depends_on", [])
    return []


def rule_based_r2(obs: dict) -> dict:
    tasks    = obs.get("tasks", [])
    devs     = obs.get("developers", [])
    done_ids = {t["id"] for t in tasks if t["status"] == "done"}
    avail    = [d for d in devs if d.get("is_available", True) and
                d.get("current_load", 0) < d.get("capacity", 5) * 2]

    def best_dev(task):
        m = [d for d in avail if d["skill"] == task.get("required_skill") or d["skill"] == "fullstack"]
        return m[0] if m else (avail[0] if avail else None)

    def deps_ok(task):
        return all(dep in done_ids for dep in _get_depends_on(task))

    # Instruction-prioritised tasks first
    for inst in [i for i in obs.get("instruction_queue", []) if not i.get("followed", False)]:
        for tid in inst.get("affects_tasks", []):
            t = next((t for t in tasks if t["id"] == tid and t["status"] == "backlog"), None)
            if t and deps_ok(t):
                dev = best_dev(t)
                if dev:
                    return {"action_type": "assign", "task_id": t["id"],
                            "dev_id": dev["id"], "new_priority": None}

    # Remaining backlog
    backlog = sorted(
        [t for t in tasks if t["status"] == "backlog" and deps_ok(t)],
        key=lambda t: (t.get("priority", 99), t.get("deadline", 99))
    )
    for t in backlog:
        dev = best_dev(t)
        if dev:
            return {"action_type": "assign", "task_id": t["id"],
                    "dev_id": dev["id"], "new_priority": None}

    return {"action_type": "skip", "task_id": None, "dev_id": None, "new_priority": None}


# ── Score calculators ─────────────────────────────────────────────────────────

def score_r1_obs(obs: dict) -> float:
    tasks  = obs.get("tasks", [])
    total  = len(tasks) or 1
    done   = sum(1 for t in tasks if t["status"] == "done")
    missed = sum(1 for t in tasks if t["status"] == "missed")
    raw    = done / total - missed / total * 0.3
    return round(max(0.01, min(0.99, raw)), 4)


def score_r2_obs(obs: dict) -> float:
    """
    R2 score: delivery×0.55 + instruction_following×0.30 + team_health×0.15
    Uses tasks_completed field if present (set by env), otherwise counts manually.
    """
    tasks       = obs.get("tasks", [])
    total       = len(tasks) or 1
    # Prefer the env's pre-computed counter, fall back to manual count
    tasks_done  = obs.get("tasks_completed") or sum(1 for t in tasks if t.get("status") == "done")
    inst_score  = obs.get("instruction_following_score", 0.01)
    debt_count  = len(obs.get("tech_debt", [])) if isinstance(obs.get("tech_debt"), list) else 0
    delivery    = tasks_done / total
    team_health = max(0.01, 1.0 - debt_count * 0.02)
    raw = delivery * 0.55 + inst_score * 0.30 + team_health * 0.15
    return round(max(0.01, min(0.99, raw)), 4)


# ── Episode runners (plain requests — no custom client class) ─────────────────

def run_r1_episode(task_name: str, policy_fn: Callable) -> dict:
    obs = _r1_reset(task_name, seed=42)
    rewards, actions = [], []

    for _ in range(12):
        if obs.get("done", False):
            break
        action = policy_fn(obs)
        result = _r1_step(action)
        obs    = result["observation"]
        rewards.append(result["reward"])
        actions.append(action["action_type"])
        if result["done"]:
            break

    return {
        "task":              task_name,
        "score":             score_r1_obs(obs),
        "cumulative_reward": round(sum(rewards), 4),
        "steps":             len(rewards),
        "tasks_completed":   obs.get("tasks_completed", 0),
        "tasks_missed":      obs.get("tasks_missed", 0),
        "action_breakdown":  {a: actions.count(a) for a in set(actions)},
    }


def run_r2_episode(task_name: str, policy_fn: Callable) -> dict:
    obs_data = _r2_reset(task_name, seed=42)
    obs = obs_data.get("observation", obs_data)
    rewards, actions, sprint_rewards = [], [], []

    for _ in range(60):
        if obs.get("done", False):
            break
        action = policy_fn(obs)
        result = _r2_step(action)
        obs    = result.get("observation", result)
        rew    = result.get("reward", 0.0)
        done   = result.get("done", obs.get("done", False))
        rewards.append(rew)
        actions.append(action.get("action_type", "skip"))
        sprint_rewards = obs.get("sprint_rewards", [])
        if done:
            break

    return {
        "task":                        task_name,
        "score":                       score_r2_obs(obs),
        "cumulative_reward":           round(sum(rewards), 4),
        "steps":                       len(rewards),
        "tasks_completed":             obs.get("tasks_completed", 0),
        "tasks_missed":                obs.get("tasks_missed", 0),
        "instruction_following_score": obs.get("instruction_following_score", 0.0),
        "tech_debt_count":             len(obs.get("tech_debt", [])) if isinstance(obs.get("tech_debt"), list) else 0,
        "sprint_rewards":              sprint_rewards,
        "action_breakdown":            {a: actions.count(a) for a in set(actions)},
    }


# ── LLM policy builders ───────────────────────────────────────────────────────

def _build_api_policy(model_id: str, system_prompt: str) -> Callable:
    """Policy that calls any OpenAI-compatible API endpoint."""
    from openai import OpenAI
    llm = OpenAI(api_key=HF_TOKEN or "dummy", base_url=API_BASE_URL)

    def policy(obs: dict) -> dict:
        user_msg = f"Current state:\n{json.dumps(obs, indent=2)}\nOutput JSON action only."
        try:
            resp = llm.chat.completions.create(
                model=model_id,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_msg},
                ],
                max_tokens=80,
                temperature=0.1,
            )
            raw = resp.choices[0].message.content or ""
            return _parse_llm_action(raw)
        except Exception:
            pass
        return {"action_type": "skip", "task_id": None, "dev_id": None, "new_priority": None}

    return policy


def build_llm_policy(model_path: str, system_prompt: str) -> Callable:
    """
    Build LLM policy.
    - If model_path is a local directory AND it's a full model: load with transformers.
    - If it contains only a LoRA adapter (adapter_config.json present): load base + PEFT.
    - Otherwise: treat as HF Hub model ID and call the API.
    """
    local_path = Path(model_path)
    if local_path.exists():
        try:
            import torch
            from transformers import AutoTokenizer

            adapter_config = local_path / "adapter_config.json"
            if adapter_config.exists():
                # LoRA-only checkpoint — load base model + adapter
                import json as _json
                with open(adapter_config) as f:
                    cfg = _json.load(f)
                base_model_id = cfg.get("base_model_name_or_path", TRAINING_MODEL)
                print(f"[INFO] Loading base model {base_model_id} + LoRA from {local_path}", flush=True)

                from transformers import AutoModelForCausalLM
                from peft import PeftModel

                tokenizer = AutoTokenizer.from_pretrained(str(local_path))
                base = AutoModelForCausalLM.from_pretrained(
                    base_model_id, torch_dtype=torch.float16, device_map="auto"
                )
                model = PeftModel.from_pretrained(base, str(local_path))
            else:
                # Full merged model
                print(f"[INFO] Loading full model from {local_path}", flush=True)
                from transformers import AutoModelForCausalLM
                tokenizer = AutoTokenizer.from_pretrained(str(local_path))
                model = AutoModelForCausalLM.from_pretrained(
                    str(local_path), torch_dtype=torch.float16, device_map="auto"
                )

            def local_policy(obs: dict) -> dict:
                prompt = f"{system_prompt}\n\nState:\n{json.dumps(obs)}\nAction:"
                inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
                with torch.no_grad():
                    out = model.generate(
                        **inputs, max_new_tokens=80,
                        temperature=0.1, do_sample=True
                    )
                raw = tokenizer.decode(
                    out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
                )
                return _parse_llm_action(raw)

            return local_policy

        except Exception as e:
            print(f"[WARN] Could not load local model: {e}", flush=True)

    # Fall back to HF API
    print(f"[INFO] Using HF API for model {model_path}", flush=True)
    return _build_api_policy(model_path, system_prompt)


# ── Main evaluation ───────────────────────────────────────────────────────────

R1_SYSTEM_PROMPT = (
    "You are an expert Tech Lead managing an agile sprint. "
    'Output a JSON action: {"action_type":"<assign|reassign|reprioritize|unblock|skip>",'
    '"task_id":"<id or null>","dev_id":"<id or null>","new_priority":<1-5 or null>}. '
    "Only output JSON. Assign backlog tasks to available developers, skill match preferred."
)

R2_SYSTEM_PROMPT = (
    "You are a tech lead managing a 6-sprint software project. "
    "Each step output ONE JSON action. Respond ONLY with valid JSON, no markdown.\n"
    'Schema: {"action_type":"<assign|reprioritize|unblock|skip>",'
    '"task_id":"<T## or null>","dev_id":"<dev# or null>","new_priority":<1-5 or null>}'
)


def evaluate(
    model_path: Optional[str] = None,
    n_episodes: int = 3,
    baseline_only: bool = False,
):
    print(f"\n{'='*60}", flush=True)
    print(f" AI Sprint Manager — Evaluation", flush=True)
    print(f" Env:   {ENV_BASE_URL}", flush=True)
    print(f" Model: {model_path or 'rule-based only'}", flush=True)
    print(f" Training model: {TRAINING_MODEL}", flush=True)
    print(f"{'='*60}", flush=True)

    # Health check both endpoints
    try:
        requests.get(f"{ENV_BASE_URL}/health", timeout=10).raise_for_status()
        requests.get(f"{ENV_BASE_URL}/project/health", timeout=10).raise_for_status()
        print("[OK] Environment is live", flush=True)
    except Exception as e:
        print(f"[ERROR] Server unreachable: {e}", flush=True)
        sys.exit(1)

    results = {
        "metadata": {
            "timestamp":      time.strftime("%Y-%m-%d %H:%M:%S"),
            "model":          model_path or "rule-based",
            "training_model": TRAINING_MODEL,
            "env_url":        ENV_BASE_URL,
            "n_episodes":     n_episodes,
            "baseline_only":  baseline_only,
        },
        "r1_llama_baseline": LLAMA_BASELINE_R1,
        "r2_llama_baseline": LLAMA_BASELINE_R2,
        "r1_rule_based": {},
        "r1_llm":        {},
        "r2_rule_based": {},
        "r2_llm":        {},
        "improvement":   {},
    }

    # ── R1 rule-based ─────────────────────────────────────────────────────────
    print(f"\n{'─'*55}", flush=True)
    print(" R1 — Rule-based baseline", flush=True)
    print(f"{'─'*55}", flush=True)
    for task in R1_TASKS:
        ep_results = []
        for ep in range(n_episodes):
            r = run_r1_episode(task, rule_based_r1)
            ep_results.append(r)
            print(f"  {task} ep{ep+1}: score={r['score']:.4f} "
                  f"done={r['tasks_completed']} reward={r['cumulative_reward']:.2f}", flush=True)
        avg_score = sum(r["score"] for r in ep_results) / n_episodes
        results["r1_rule_based"][task] = {"avg_score": round(avg_score, 4), "episodes": ep_results}

    # ── R2 rule-based ─────────────────────────────────────────────────────────
    print(f"\n{'─'*55}", flush=True)
    print(" R2 — Rule-based baseline", flush=True)
    print(f"{'─'*55}", flush=True)
    for task in R2_TASKS:
        ep_results = []
        for ep in range(n_episodes):
            r = run_r2_episode(task, rule_based_r2)
            ep_results.append(r)
            print(f"  {task} ep{ep+1}: score={r['score']:.4f} "
                  f"inst={r['instruction_following_score']:.2f}", flush=True)
        avg_score = sum(r["score"] for r in ep_results) / n_episodes
        results["r2_rule_based"][task] = {"avg_score": round(avg_score, 4), "episodes": ep_results}

    if not baseline_only and model_path:
        llm_r1_policy = _build_api_policy(model_path, R1_SYSTEM_PROMPT)
        llm_r2_policy = build_llm_policy(model_path, R2_SYSTEM_PROMPT)

        # ── R1 LLM ───────────────────────────────────────────────────────────
        print(f"\n{'─'*55}", flush=True)
        print(f" R1 — LLM ({model_path})", flush=True)
        print(f"{'─'*55}", flush=True)
        for task in R1_TASKS:
            ep_results = []
            for ep in range(n_episodes):
                r = run_r1_episode(task, llm_r1_policy)
                ep_results.append(r)
                print(f"  {task} ep{ep+1}: score={r['score']:.4f}", flush=True)
            avg_score = sum(r["score"] for r in ep_results) / n_episodes
            results["r1_llm"][task] = {"avg_score": round(avg_score, 4), "episodes": ep_results}

        # ── R2 LLM ───────────────────────────────────────────────────────────
        print(f"\n{'─'*55}", flush=True)
        print(f" R2 — LLM ({model_path})", flush=True)
        print(f"{'─'*55}", flush=True)
        for task in R2_TASKS:
            ep_results = []
            for ep in range(n_episodes):
                r = run_r2_episode(task, llm_r2_policy)
                ep_results.append(r)
                print(f"  {task} ep{ep+1}: score={r['score']:.4f} "
                      f"inst={r['instruction_following_score']:.2f}", flush=True)
            avg_score = sum(r["score"] for r in ep_results) / n_episodes
            results["r2_llm"][task] = {"avg_score": round(avg_score, 4), "episodes": ep_results}

        # ── Improvement table ─────────────────────────────────────────────────
        for task in R2_TASKS:
            base  = LLAMA_BASELINE_R2.get(task, 0)
            llm   = results["r2_llm"].get(task, {}).get("avg_score", base)
            delta = round(llm - base, 4)
            results["improvement"][task] = {
                "llama_baseline":    base,
                "trained_llm":       llm,
                "delta_vs_llama":    delta,
                "pct_gain_vs_llama": round(delta / max(base, 0.01) * 100, 1),
            }

    _print_summary(results, baseline_only)

    out_path = RESULTS_DIR / "r2_evaluation.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[INFO] Results saved to {out_path}", flush=True)
    return results


def _print_summary(results: dict, baseline_only: bool):
    print(f"\n{'='*65}", flush=True)
    print(" EVALUATION SUMMARY", flush=True)
    print(f"{'='*65}", flush=True)

    # R1
    print(f"\n{'R1 SCORES':─<65}", flush=True)
    print(f"  {'Task':<22} {'Llama Baseline':>15} {'Rule-Based':>11} {'LLM Trained':>12}", flush=True)
    for task in R1_TASKS:
        llama = results["r1_llama_baseline"].get(task, 0)
        rule  = results["r1_rule_based"].get(task, {}).get("avg_score", 0)
        llm   = results["r1_llm"].get(task, {}).get("avg_score", 0)
        rule_s = f"{rule:.4f}" if rule else "—"
        llm_s  = f"{llm:.4f}"  if llm  else "—"
        print(f"  {task:<22} {llama:>15.4f} {rule_s:>11} {llm_s:>12}", flush=True)
    avg_r1 = results["r1_llama_baseline"].get("average", 0)
    print(f"  {'AVERAGE':<22} {avg_r1:>15.4f}", flush=True)

    # R2
    print(f"\n{'R2 SCORES':─<65}", flush=True)
    print(f"  {'Task':<22} {'Llama Baseline':>15} {'Rule-Based':>11} {'LLM Trained':>12} {'Δ vs Llama':>10}", flush=True)
    for task in R2_TASKS:
        llama = results["r2_llama_baseline"].get(task, 0)
        rule  = results["r2_rule_based"].get(task, {}).get("avg_score", 0)
        llm   = results["r2_llm"].get(task, {}).get("avg_score", 0)
        imp   = results["improvement"].get(task, {})
        delta = imp.get("delta_vs_llama", 0)
        rule_s  = f"{rule:.4f}"  if rule  else "—"
        llm_s   = f"{llm:.4f}"   if llm   else "—"
        delta_s = f"+{delta:.4f}" if delta > 0 else (f"{delta:.4f}" if delta else "—")
        print(f"  {task:<22} {llama:>15.4f} {rule_s:>11} {llm_s:>12} {delta_s:>10}", flush=True)
    avg_r2 = results["r2_llama_baseline"].get("average", 0)
    print(f"  {'AVERAGE':<22} {avg_r2:>15.4f}", flush=True)

    print(f"\n{'='*65}", flush=True)
    print(f"  Training model: {TRAINING_MODEL} (GRPO, 4-bit QLoRA)", flush=True)
    print(f"  Baselines:      Llama-3.1-8B zero-shot (via HF Router)", flush=True)
    print(f"{'='*65}", flush=True)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Evaluate R1+R2 before/after training")
    parser.add_argument(
        "--model", type=str, default=None,
        help="HF model ID or local path to trained model (e.g. sejal-k/ai-sprint-manager-trained)"
    )
    parser.add_argument(
        "--baseline-only", action="store_true",
        help="Run rule-based baseline only, no model needed"
    )
    parser.add_argument(
        "--episodes", type=int, default=3,
        help="Episodes per task (default: 3)"
    )
    args = parser.parse_args()

    if not args.baseline_only and not args.model:
        print("[INFO] No --model specified. Defaulting to trained model on HF Hub.", flush=True)
        args.model = MODEL_NAME

    evaluate(
        model_path=args.model,
        n_episodes=args.episodes,
        baseline_only=args.baseline_only,
    )


if __name__ == "__main__":
    main()