"""
evaluate.py — Compare baseline vs trained RL policy
=====================================================
Run after train.py completes.
"""
from __future__ import annotations
import os
import json
import requests
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

ENV_URL        = os.getenv("ENV_BASE_URL", "https://sejal-k-ai-sprint-manager.hf.space")
BASELINE_MODEL = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-1.5B-Instruct")
API_BASE_URL   = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
API_KEY        = os.getenv("HF_TOKEN", "dummy")
POLICY_PATH    = "./results/best_policy.json"
TASKS          = ["easy_sprint", "medium_sprint", "hard_sprint"]
N_SEEDS        = 5

SYSTEM = """You are a Tech Lead. Output ONLY JSON.
{"action_type": "assign", "task_id": "T1", "dev_id": "dev1", "new_priority": null}
Only assign backlog tasks to available skill-matched developers."""


# ── Shared helpers ────────────────────────────────────────────────────────────

def env_reset(task_name, seed=42):
    r = requests.post(f"{ENV_URL}/reset",
                      json={"task_name": task_name, "seed": seed}, timeout=30)
    r.raise_for_status()
    return r.json()

def env_step(action):
    r = requests.post(f"{ENV_URL}/step", json={"action": {
        "action_type":  action.get("action_type", "skip"),
        "task_id":      action.get("task_id"),
        "dev_id":       action.get("dev_id"),
        "new_priority": action.get("new_priority"),
    }}, timeout=30)
    r.raise_for_status()
    return r.json()

def run_episode(act_fn, task_name, seed=42):
    obs    = env_reset(task_name, seed)
    result = {"info": {}}
    for _ in range(12):
        if obs.get("done"):
            break
        action = act_fn(obs)
        result = env_step(action)
        obs    = result["observation"]
    return max(0.01, min(0.99, result.get("info", {}).get("final_score", 0.01)))

def evaluate(act_fn, name):
    print(f"\n  {name}")
    print(f"  {'─'*45}")
    scores = {}
    for task in TASKS:
        vals = []
        for seed in range(N_SEEDS):
            try:
                vals.append(run_episode(act_fn, task, seed=seed*13+42))
            except Exception:
                vals.append(0.01)
        avg = sum(vals) / len(vals)
        scores[task] = round(avg, 4)
        bar = "█" * int(avg * 20)
        print(f"  {task:<20} {avg:.4f}  {bar}")
    overall = sum(scores.values()) / len(scores)
    scores["average"] = round(overall, 4)
    print(f"  {'AVERAGE':<20} {overall:.4f}")
    return scores


# ── Trained policy agent ──────────────────────────────────────────────────────

class TrainedPolicy:
    def __init__(self, path):
        with open(path) as f:
            w = json.load(f)
        self.priority_weight = w["priority_weight"]
        self.deadline_weight = w["deadline_weight"]
        self.skill_weight    = w["skill_weight"]
        self.load_weight     = w["load_weight"]

    def act(self, obs):
        day     = obs.get("current_day", 1)
        backlog = [t for t in obs["tasks"] if t["status"] == "backlog"]
        avail   = [d for d in obs["developers"]
                   if d["is_available"] and d["current_load"] < d["capacity"]]
        if not backlog or not avail:
            return {"action_type": "skip", "task_id": None,
                    "dev_id": None, "new_priority": None}

        best, bt, bd = float("-inf"), None, None
        for t in backlog:
            for d in avail:
                s  = self.priority_weight * (6 - t["priority"])
                s += self.deadline_weight * (10 / max(1, t["deadline"] - day))
                if d["skill"] == t["required_skill"]:
                    s += self.skill_weight * 3
                elif d["skill"] == "fullstack":
                    s += self.skill_weight * 2
                else:
                    s -= self.skill_weight * 2
                s -= self.load_weight * (d["current_load"] / max(d["capacity"], 1)) * 2
                if s > best:
                    best, bt, bd = s, t, d
        if bt and bd:
            return {"action_type": "assign", "task_id": bt["id"],
                    "dev_id": bd["id"], "new_priority": None}
        return {"action_type": "skip", "task_id": None,
                "dev_id": None, "new_priority": None}


# ── Baseline LLM agent ────────────────────────────────────────────────────────

def make_baseline_act():
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    def act(obs):
        backlog = sorted([t for t in obs["tasks"] if t["status"] == "backlog"],
                         key=lambda t: (t["priority"], t["deadline"]))
        avail   = [d for d in obs["developers"]
                   if d["is_available"] and d["current_load"] < d["capacity"]]
        prompt  = (
            f"Day {obs['current_day']}/{obs['sprint_length']}\n"
            f"Backlog: {[t['id']+':'+t['required_skill'] for t in backlog[:4]]}\n"
            f"Devs: {[d['id']+':'+d['skill'] for d in avail]}\n"
            f"JSON action:"
        )
        try:
            resp = client.chat.completions.create(
                model=BASELINE_MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM},
                    {"role": "user",   "content": prompt},
                ],
                temperature=0.1, max_tokens=80,
            )
            text = resp.choices[0].message.content or ""
            a = json.loads(text.strip())
            if a.get("action_type") in ("assign","reassign","skip","unblock"):
                return a
        except Exception:
            pass
        return {"action_type": "skip", "task_id": None,
                "dev_id": None, "new_priority": None}
    return act


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    try:
        r = requests.get(f"{ENV_URL}/health", timeout=10)
        print(f"Server: {r.json()}")
    except Exception:
        print("ERROR: Start server first: python ui.py")
        return

    print("\n" + "="*55)
    print("  EVALUATION: Baseline vs Trained RL Policy")
    print("="*55)

    results = {}

    # Baseline
    results["baseline"] = evaluate(make_baseline_act(), f"Baseline LLM ({BASELINE_MODEL})")

    # Trained policy
    if os.path.exists(POLICY_PATH):
        policy = TrainedPolicy(POLICY_PATH)
        results["trained"] = evaluate(policy.act, f"Trained RL Policy ({POLICY_PATH})")

        # Comparison table
        print(f"\n{'='*55}")
        print("  IMPROVEMENT SUMMARY")
        print(f"{'='*55}")
        print(f"  {'Task':<20} {'Baseline':>10} {'Trained':>10} {'Delta':>10}")
        print(f"  {'─'*48}")
        for task in TASKS + ["average"]:
            b     = results["baseline"].get(task, 0)
            t     = results["trained"].get(task, 0)
            delta = t - b
            sign  = "+" if delta >= 0 else ""
            print(f"  {task:<20} {b:>10.4f} {t:>10.4f} {sign}{delta:>9.4f}")
    else:
        print(f"\n  No trained policy at {POLICY_PATH}")
        print("  Run python train.py first.")

    # Save
    os.makedirs("results", exist_ok=True)
    with open("results/evaluation.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved → results/evaluation.json")


if __name__ == "__main__":
    main()