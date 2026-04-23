"""
evaluate_r2.py — Round 2 Evaluation Script
============================================
Compares baseline (rule-based) vs trained LLM on both R1 and R2 tasks.
Produces the before/after improvement table judges want to see.
Saves results to results/r2_evaluation.json.

Usage:
    # Baseline only (rule-based, no model needed):
    python evaluate_r2.py --baseline-only

    # Full comparison (trained model vs baseline):
    python evaluate_r2.py --model results/trained_model

    # Quick 1-episode-per-task run:
    python evaluate_r2.py --model results/trained_model --episodes 1
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import requests

ENV_BASE_URL = os.getenv("ENV_BASE_URL", "http://localhost:7860")
HF_TOKEN     = os.getenv("HF_TOKEN", "")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct")

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

R1_TASKS = ["easy_sprint", "medium_sprint", "hard_sprint"]
R2_TASKS = ["project_easy", "project_medium", "project_hard"]

# Qwen2.5-1.5B baseline scores (from inference.py run — your actual measured baseline)
QWEN_BASELINE_R1 = {
    "easy_sprint":   0.9900,
    "medium_sprint": 0.6667,
    "hard_sprint":   0.3716,
}

# R2 rule-based baseline (from evaluate_r2.py --baseline-only, 3 episodes each)
RULE_BASED_BASELINE_R2 = {
    "project_easy":   0.2727,
    "project_medium": 0.2063,
    "project_hard":   0.2610,
}


# ── Rule-based policies ───────────────────────────────────────────────────────

def rule_based_r1(obs: dict) -> dict:
    tasks  = obs.get("tasks", [])
    devs   = obs.get("developers", [])
    avail  = [d for d in devs if d["is_available"] and d["current_load"] < d["capacity"]]
    backlog = sorted([t for t in tasks if t["status"] == "backlog"],
                     key=lambda t: (t["priority"], t["deadline"]))
    for task in backlog:
        match = [d for d in avail if d["skill"] == task.get("required_skill") or d["skill"] == "fullstack"]
        dev = match[0] if match else (avail[0] if avail else None)
        if dev:
            return {"action_type": "assign", "task_id": task["id"],
                    "dev_id": dev["id"], "new_priority": None}
    return {"action_type": "skip", "task_id": None, "dev_id": None, "new_priority": None}


def rule_based_r2(obs: dict) -> dict:
    tasks    = obs.get("tasks", [])
    devs     = obs.get("developers", [])
    done_ids = {t["id"] for t in tasks if t["status"] == "done"}
    avail    = [d for d in devs if d["is_available"] and d["current_load"] < d["capacity"] * 2]

    def best_dev(task):
        m = [d for d in avail if d["skill"] == task.get("required_skill") or d["skill"] == "fullstack"]
        return m[0] if m else (avail[0] if avail else None)

    for inst in [i for i in obs.get("instruction_queue", []) if not i.get("followed", False)]:
        for tid in inst.get("affects_tasks", []):
            t = next((t for t in tasks if t["id"] == tid and t["status"] == "backlog"), None)
            if t and all(d in done_ids for d in t.get("metadata", {}).get("depends_on", [])):
                dev = best_dev(t)
                if dev:
                    return {"action_type": "assign", "task_id": t["id"],
                            "dev_id": dev["id"], "new_priority": None}

    backlog = sorted([t for t in tasks if t["status"] == "backlog"],
                     key=lambda t: (t["priority"], t["deadline"]))
    for t in backlog:
        if all(d in done_ids for d in t.get("metadata", {}).get("depends_on", [])):
            dev = best_dev(t)
            if dev:
                return {"action_type": "assign", "task_id": t["id"],
                        "dev_id": dev["id"], "new_priority": None}

    return {"action_type": "skip", "task_id": None, "dev_id": None, "new_priority": None}


# ── Score calculators ─────────────────────────────────────────────────────────

def score_r1_obs(obs: dict) -> float:
    """Extract R1 final score from terminal observation."""
    done   = sum(1 for t in obs.get("tasks", []) if t["status"] == "done")
    total  = len(obs.get("tasks", [])) or 1
    missed = sum(1 for t in obs.get("tasks", []) if t["status"] == "missed")
    raw    = done / total - missed / total * 0.3
    return round(max(0.01, min(0.99, raw)), 4)


def score_r2_obs(obs: dict) -> float:
    """Compute R2 project score from terminal observation."""
    tasks_total   = len(obs.get("tasks", [])) or 1
    tasks_done    = obs.get("tasks_completed", 0)
    inst_score    = obs.get("instruction_following_score", 0.01)
    delivery_rate = tasks_done / tasks_total
    debt_count    = len(obs.get("tech_debt", []))
    team_health   = max(0.01, 1.0 - debt_count * 0.02)
    raw = delivery_rate * 0.55 + inst_score * 0.30 + team_health * 0.15
    return round(max(0.01, min(0.99, raw)), 4)


# ── Episode runners ───────────────────────────────────────────────────────────

def run_r1_episode(r1_client, task_name: str, policy_fn) -> dict:
    """Run one R1 episode. Calls /step directly as dict to avoid model_dump() issue."""
    import requests as _req
    obs = r1_client.reset(task_name=task_name, seed=42)
    rewards, actions = [], []
    base_url = r1_client.base_url
    for _ in range(12):
        if obs.get("done", False):
            break
        action = policy_fn(obs)
        resp = _req.post(f"{base_url}/step", json={"action": action}, timeout=30)
        resp.raise_for_status()
        result = resp.json()
        obs = result["observation"]
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


def run_r2_episode(r2_client, task_name: str, policy_fn) -> dict:
    obs = r2_client.reset(task_name=task_name, seed=42)
    rewards, actions, sprint_rewards = [], [], []
    for _ in range(60):
        if obs.get("done", False):
            break
        action = policy_fn(obs)
        result = r2_client.step(action)
        obs    = result.observation if hasattr(result, "observation") else result["observation"]
        rew    = result.reward      if hasattr(result, "reward")      else result["reward"]
        done   = result.done        if hasattr(result, "done")         else result["done"]
        rewards.append(rew)
        actions.append(action["action_type"])
        sprint_rewards = obs.get("sprint_rewards", [])
        if done:
            break
    return {
        "task":                       task_name,
        "score":                      score_r2_obs(obs),
        "cumulative_reward":          round(sum(rewards), 4),
        "steps":                      len(rewards),
        "tasks_completed":            obs.get("tasks_completed", 0),
        "tasks_missed":               obs.get("tasks_missed", 0),
        "instruction_following_score": obs.get("instruction_following_score", 0.0),
        "tech_debt_count":            len(obs.get("tech_debt", [])),
        "sprint_rewards":             sprint_rewards,
        "action_breakdown":           {a: actions.count(a) for a in set(actions)},
    }


# ── LLM policy builder ────────────────────────────────────────────────────────

def build_llm_policy(model_path: str, system_prompt: str):
    """
    Load a trained model from disk and return a policy function obs->action dict.
    Supports: local LoRA checkpoint, HF Hub model ID, or HF router API.
    """
    # Try loading locally with transformers
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
        import torch

        print(f"[INFO] Loading model from {model_path}...", flush=True)
        tokenizer = AutoTokenizer.from_pretrained(model_path, token=HF_TOKEN or None)

        # Try Unsloth first for LoRA merge
        try:
            from unsloth import FastLanguageModel
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=model_path,
                max_seq_length=2048,
                load_in_4bit=True,
                token=HF_TOKEN or None,
            )
            FastLanguageModel.for_inference(model)
            print(f"[INFO] Loaded with Unsloth (4-bit inference mode)", flush=True)
        except Exception:
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map="auto",
                torch_dtype=torch.bfloat16,
                token=HF_TOKEN or None,
            )
            print(f"[INFO] Loaded with standard transformers", flush=True)

        pipe = pipeline("text-generation", model=model, tokenizer=tokenizer,
                        max_new_tokens=80, temperature=0.1, do_sample=True)

        def policy_fn(obs: dict) -> dict:
            from inference_r2 import _build_r2_prompt, _parse_action
            prompt = _build_r2_prompt(obs)
            messages = [{"role": "system", "content": system_prompt},
                        {"role": "user",   "content": prompt}]
            try:
                out = pipe(messages)[0]["generated_text"]
                text = out[-1]["content"] if isinstance(out, list) else str(out)
                return _parse_action(text)
            except Exception:
                return {"action_type": "skip", "task_id": None, "dev_id": None, "new_priority": None}

        return policy_fn

    except Exception as e:
        print(f"[WARN] Could not load local model ({e}). Using OpenAI API with {model_path}.", flush=True)
        return _build_api_policy(model_path, system_prompt)


def _build_api_policy(model_name: str, system_prompt: str):
    """Fallback: use OpenAI-compatible API (HF router)."""
    from openai import OpenAI
    llm_client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN or "dummy")

    def policy_fn(obs: dict) -> dict:
        from inference_r2 import _build_r2_prompt, _parse_action
        try:
            resp = llm_client.chat.completions.create(
                model=model_name,
                messages=[{"role": "system", "content": system_prompt},
                          {"role": "user",   "content": _build_r2_prompt(obs)}],
                temperature=0.1,
                max_tokens=80,
            )
            return _parse_action(resp.choices[0].message.content or "")
        except Exception:
            return {"action_type": "skip", "task_id": None, "dev_id": None, "new_priority": None}

    return policy_fn


# ── Main evaluation ───────────────────────────────────────────────────────────

def evaluate(model_path: str | None, n_episodes: int = 3, baseline_only: bool = False):
    from client import SprintEnvClient
    from project_client import ProjectEnvClient

    # Health checks
    try:
        requests.get(f"{ENV_BASE_URL}/health", timeout=10).raise_for_status()
        requests.get(f"{ENV_BASE_URL}/project/health", timeout=10).raise_for_status()
    except Exception as e:
        print(f"[ERROR] Server unreachable: {e}", flush=True)
        sys.exit(1)

    results = {
        "metadata": {
            "timestamp":     time.strftime("%Y-%m-%d %H:%M:%S"),
            "model":         model_path or "rule-based",
            "env_url":       ENV_BASE_URL,
            "n_episodes":    n_episodes,
            "baseline_only": baseline_only,
        },
        "r1_baseline_qwen":  QWEN_BASELINE_R1,
        "r1_rule_based":     {},
        "r1_llm":            {},
        "r2_baseline_rules": RULE_BASED_BASELINE_R2,
        "r2_rule_based":     {},
        "r2_llm":            {},
        "improvement":       {},
    }

    r1_client = SprintEnvClient(base_url=ENV_BASE_URL)
    r2_client = ProjectEnvClient(base_url=ENV_BASE_URL)

    # ── R1 rule-based baseline ────────────────────────────────────────────────
    print(f"\n{'─'*55}", flush=True)
    print(f" R1 — Rule-based baseline", flush=True)
    print(f"{'─'*55}", flush=True)
    for task in R1_TASKS:
        ep_results = []
        for ep in range(n_episodes):
            r = run_r1_episode(r1_client, task, rule_based_r1)
            ep_results.append(r)
            print(f"  {task} ep{ep+1}: score={r['score']:.4f} "
                  f"done={r['tasks_completed']} reward={r['cumulative_reward']:.2f}", flush=True)
        avg_score = sum(r["score"] for r in ep_results) / n_episodes
        results["r1_rule_based"][task] = {
            "avg_score": round(avg_score, 4),
            "episodes":  ep_results,
        }

    # ── R2 rule-based baseline ────────────────────────────────────────────────
    print(f"\n{'─'*55}", flush=True)
    print(f" R2 — Rule-based baseline", flush=True)
    print(f"{'─'*55}", flush=True)
    for task in R2_TASKS:
        ep_results = []
        for ep in range(n_episodes):
            r = run_r2_episode(r2_client, task, rule_based_r2)
            ep_results.append(r)
            print(f"  {task} ep{ep+1}: score={r['score']:.4f} "
                  f"done={r['tasks_completed']} inst={r['instruction_following_score']:.2f} "
                  f"debt={r['tech_debt_count']}", flush=True)
        avg_score = sum(r["score"] for r in ep_results) / n_episodes
        results["r2_rule_based"][task] = {
            "avg_score": round(avg_score, 4),
            "episodes":  ep_results,
        }

    if not baseline_only and model_path:
        from inference_r2 import R2_SYSTEM_PROMPT

        # R1 system prompt (inline to avoid import dependency on inference.py internals)
        R1_SYSTEM_PROMPT = (
            "You are an expert Tech Lead managing an agile sprint. "
            "Output a JSON action: {\"action_type\":\"<assign|reassign|reprioritize|unblock|skip>\","
            "\"task_id\":\"<id or null>\",\"dev_id\":\"<id or null>\",\"new_priority\":<1-5 or null>}. "
            "Only output JSON. Assign backlog tasks to available developers, skill match preferred."
        )

        llm_r1_policy = _build_api_policy(model_path, R1_SYSTEM_PROMPT)
        llm_r2_policy = build_llm_policy(model_path, R2_SYSTEM_PROMPT)

        # ── R1 LLM ───────────────────────────────────────────────────────────
        print(f"\n{'─'*55}", flush=True)
        print(f" R1 — LLM ({model_path})", flush=True)
        print(f"{'─'*55}", flush=True)
        for task in R1_TASKS:
            ep_results = []
            for ep in range(n_episodes):
                r = run_r1_episode(r1_client, task, llm_r1_policy)
                ep_results.append(r)
                print(f"  {task} ep{ep+1}: score={r['score']:.4f}", flush=True)
            avg_score = sum(r["score"] for r in ep_results) / n_episodes
            results["r1_llm"][task] = {
                "avg_score": round(avg_score, 4),
                "episodes":  ep_results,
            }

        # ── R2 LLM ───────────────────────────────────────────────────────────
        print(f"\n{'─'*55}", flush=True)
        print(f" R2 — LLM ({model_path})", flush=True)
        print(f"{'─'*55}", flush=True)
        for task in R2_TASKS:
            ep_results = []
            for ep in range(n_episodes):
                r = run_r2_episode(r2_client, task, llm_r2_policy)
                ep_results.append(r)
                print(f"  {task} ep{ep+1}: score={r['score']:.4f} "
                      f"inst={r['instruction_following_score']:.2f}", flush=True)
            avg_score = sum(r["score"] for r in ep_results) / n_episodes
            results["r2_llm"][task] = {
                "avg_score": round(avg_score, 4),
                "episodes":  ep_results,
            }

        # ── Improvement table ─────────────────────────────────────────────────
        for task in R2_TASKS:
            base  = results["r2_rule_based"][task]["avg_score"]
            llm   = results["r2_llm"].get(task, {}).get("avg_score", base)
            delta = round(llm - base, 4)
            pct   = round(delta / max(base, 0.01) * 100, 1)
            results["improvement"][task] = {"baseline": base, "llm": llm,
                                            "delta": delta, "pct_gain": pct}

    r1_client.close()
    r2_client.close()

    # ── Print summary table ───────────────────────────────────────────────────
    _print_summary(results, baseline_only)

    # ── Save JSON ─────────────────────────────────────────────────────────────
    out_path = RESULTS_DIR / "r2_evaluation.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[INFO] Results saved to {out_path}", flush=True)
    return results


def _print_summary(results: dict, baseline_only: bool):
    print(f"\n{'='*60}", flush=True)
    print(f" EVALUATION SUMMARY", flush=True)
    print(f"{'='*60}", flush=True)

    print(f"\n{'R1 SCORES':─<55}", flush=True)
    print(f"  {'Task':<22} {'Qwen Baseline':>14} {'LLM Trained':>12}", flush=True)
    for task in R1_TASKS:
        qwen  = results["r1_baseline_qwen"].get(task, 0)
        llm   = results["r1_llm"].get(task, {}).get("avg_score", 0)
        llm_s = f"{llm:.4f}" if llm else "—"
        print(f"  {task:<22} {qwen:>14.4f} {llm_s:>12}", flush=True)

    print(f"\n{'R2 SCORES':─<55}", flush=True)
    print(f"  {'Task':<22} {'Rule-based':>12} {'LLM Trained':>12} {'Δ':>7} {'%gain':>7}", flush=True)
    for task in R2_TASKS:
        rule  = results["r2_rule_based"].get(task, {}).get("avg_score",
                results["r2_baseline_rules"].get(task, 0))
        llm   = results["r2_llm"].get(task, {}).get("avg_score", 0)
        imp   = results["improvement"].get(task, {})
        delta = imp.get("delta", 0)
        pct   = imp.get("pct_gain", 0)
        llm_s   = f"{llm:.4f}" if llm else "—"
        delta_s = f"+{delta:.4f}" if delta > 0 else (f"{delta:.4f}" if delta else "—")
        pct_s   = f"+{pct:.1f}%" if pct > 0 else (f"{pct:.1f}%" if pct else "—")
        print(f"  {task:<22} {rule:>12.4f} {llm_s:>12} {delta_s:>7} {pct_s:>7}", flush=True)

    print(f"\n{'='*60}", flush=True)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Evaluate R1+R2 before/after training")
    parser.add_argument("--model",         type=str, default=None,
                        help="Path to trained model dir or HF model ID")
    parser.add_argument("--baseline-only", action="store_true",
                        help="Run rule-based baseline only (no model needed)")
    parser.add_argument("--episodes",      type=int, default=3,
                        help="Episodes per task (default: 3)")
    args = parser.parse_args()

    if not args.baseline_only and not args.model:
        print("[INFO] No --model specified. Running baseline-only evaluation.", flush=True)
        args.baseline_only = True

    evaluate(
        model_path=args.model,
        n_episodes=args.episodes,
        baseline_only=args.baseline_only,
    )


if __name__ == "__main__":
    main()