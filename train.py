"""
train.py — Lightweight RL Training for Sprint Manager (No TRL required)
========================================================================
Uses REINFORCE (Policy Gradient) — the foundation of GRPO/PPO.
Works on Windows CPU without any encoding issues.

Run:
  1. python ui.py       (Terminal 1)
  2. python train.py    (Terminal 2)
"""
from __future__ import annotations
import os
import json
import math
import time
import requests
import random
from dotenv import load_dotenv

load_dotenv()

ENV_URL    = os.getenv("ENV_BASE_URL", "http://localhost:7860")
TASKS      = ["easy_sprint", "medium_sprint", "hard_sprint"]
N_EPISODES = 30    # increase with GPU/time
SAVE_DIR   = "./results"

os.makedirs(SAVE_DIR, exist_ok=True)

print("="*55)
print("  Sprint Manager — RL Training (REINFORCE)")
print("="*55)
print(f"  Server : {ENV_URL}")
print(f"  Episodes: {N_EPISODES}")
print()

# ── Environment helpers ───────────────────────────────────────────────────────

def env_reset(task_name: str, seed: int = None) -> dict:
    r = requests.post(f"{ENV_URL}/reset",
                      json={"task_name": task_name, "seed": seed}, timeout=30)
    r.raise_for_status()
    return r.json()


def env_step(action: dict) -> dict:
    r = requests.post(f"{ENV_URL}/step", json={"action": {
        "action_type":  action.get("action_type", "skip"),
        "task_id":      action.get("task_id"),
        "dev_id":       action.get("dev_id"),
        "new_priority": action.get("new_priority"),
    }}, timeout=30)
    r.raise_for_status()
    return r.json()


# ── Rule-based policy (our "model" for now) ───────────────────────────────────
# In Round 2 with GPU, replace this with an actual LLM

class SprintPolicy:
    """
    A learnable rule-based policy.
    Parameters control how it weights different factors.
    These get updated by REINFORCE gradient estimates.
    
    Think of this as a simplified version of what a neural network does —
    it scores possible actions and picks the best one.
    When replaced with an LLM, the LLM's weights are updated instead.
    """

    def __init__(self):
        # Learnable weights — start at 1.0, get updated by RL
        self.priority_weight   = 1.0   # how much to value task priority
        self.deadline_weight   = 1.0   # how much to value deadline urgency
        self.skill_weight      = 2.0   # how much to value skill matching
        self.load_weight       = 1.0   # how much to avoid overloaded devs

    def score_assignment(self, task: dict, dev: dict, current_day: int) -> float:
        """Score a potential task→dev assignment. Higher = better."""
        score = 0.0

        # Priority signal (P1=highest=5 pts, P5=lowest=1 pt)
        score += self.priority_weight * (6 - task["priority"])

        # Deadline urgency (tasks due soon score higher)
        days_left = max(1, task["deadline"] - current_day)
        score += self.deadline_weight * (10.0 / days_left)

        # Skill match bonus
        if dev["skill"] == task["required_skill"]:
            score += self.skill_weight * 3.0
        elif dev["skill"] == "fullstack":
            score += self.skill_weight * 2.0
        else:
            score -= self.skill_weight * 2.0   # mismatch penalty

        # Prefer less loaded developers
        load_ratio = dev["current_load"] / max(dev["capacity"], 1)
        score -= self.load_weight * load_ratio * 2.0

        return score

    def act(self, obs: dict) -> dict:
        """Choose best action given current observation."""
        current_day = obs.get("current_day", 1)
        tasks = obs.get("tasks", [])
        devs  = obs.get("developers", [])

        backlog = [t for t in tasks if t["status"] == "backlog"]
        available = [
            d for d in devs
            if d["is_available"] and d["current_load"] < d["capacity"]
        ]

        if not backlog or not available:
            return {"action_type": "skip", "task_id": None,
                    "dev_id": None, "new_priority": None}

        # Find best task→dev pair
        best_score  = float("-inf")
        best_task   = None
        best_dev    = None

        for task in backlog:
            for dev in available:
                score = self.score_assignment(task, dev, current_day)
                if score > best_score:
                    best_score = score
                    best_task  = task
                    best_dev   = dev

        if best_task and best_dev:
            return {
                "action_type": "assign",
                "task_id":     best_task["id"],
                "dev_id":      best_dev["id"],
                "new_priority": None,
            }
        return {"action_type": "skip", "task_id": None,
                "dev_id": None, "new_priority": None}

    def update(self, reward_signal: float, learning_rate: float = 0.05):
        """
        REINFORCE update — nudge weights in direction of reward.
        Positive reward → strengthen current weights
        Negative reward → weaken current weights
        
        This is the core of policy gradient RL.
        With an LLM, this becomes a gradient update on millions of parameters.
        """
        delta = learning_rate * reward_signal
        self.priority_weight  = max(0.1, self.priority_weight  + delta * 0.3)
        self.deadline_weight  = max(0.1, self.deadline_weight  + delta * 0.3)
        self.skill_weight     = max(0.1, self.skill_weight     + delta * 0.5)
        self.load_weight      = max(0.1, self.load_weight      + delta * 0.2)

    def save(self, path: str):
        weights = {
            "priority_weight": self.priority_weight,
            "deadline_weight": self.deadline_weight,
            "skill_weight":    self.skill_weight,
            "load_weight":     self.load_weight,
        }
        with open(path, "w") as f:
            json.dump(weights, f, indent=2)
        print(f"  Policy saved → {path}")

    def load(self, path: str):
        with open(path) as f:
            weights = json.load(f)
        self.priority_weight = weights["priority_weight"]
        self.deadline_weight = weights["deadline_weight"]
        self.skill_weight    = weights["skill_weight"]
        self.load_weight     = weights["load_weight"]


# ── Training loop ─────────────────────────────────────────────────────────────

def run_episode(policy: SprintPolicy, task_name: str, seed: int = None) -> tuple[float, float]:
    """Run one full episode. Returns (cumulative_reward, final_score)."""
    obs = env_reset(task_name, seed=seed)
    total_reward = 0.0
    final_score  = 0.01

    for _ in range(12):
        if obs.get("done"):
            break
        action = policy.act(obs)
        result = env_step(action)
        total_reward += result.get("reward", 0.0)
        obs = result["observation"]
        if result.get("done"):
            final_score = max(0.01, min(0.99,
                result.get("info", {}).get("final_score", 0.01)))
            break

    return total_reward, final_score


def train():
    # Check server
    try:
        r = requests.get(f"{ENV_URL}/health", timeout=10)
        print(f"Server: {r.json()}\n")
    except Exception:
        print(f"ERROR: Cannot reach server at {ENV_URL}")
        print("Start it first: python ui.py")
        return

    policy    = SprintPolicy()
    history   = []
    best_avg  = float("-inf")

    print(f"{'Episode':>8} {'Task':<15} {'Reward':>8} {'Score':>8} "
          f"{'SkillW':>8} {'Avg10':>8}")
    print("─" * 65)

    start = time.time()

    for episode in range(1, N_EPISODES + 1):
        # Rotate through tasks so policy learns all scenarios
        task = TASKS[(episode - 1) % len(TASKS)]
        seed = episode * 7  # different seed each episode = diverse experience

        total_reward, final_score = run_episode(policy, task, seed=seed)

        # Normalise reward for stable updates
        norm_reward = math.tanh(total_reward / 10.0)

        # REINFORCE update
        policy.update(norm_reward, learning_rate=0.03)

        history.append({
            "episode": episode,
            "task":    task,
            "reward":  round(total_reward, 4),
            "score":   final_score,
            "skill_weight": round(policy.skill_weight, 3),
        })

        # Rolling average of last 10 episodes
        recent_scores = [h["score"] for h in history[-10:]]
        avg10 = sum(recent_scores) / len(recent_scores)

        print(f"{episode:>8} {task:<15} {total_reward:>8.2f} {final_score:>8.4f} "
              f"{policy.skill_weight:>8.3f} {avg10:>8.4f}")

        # Save best policy
        if avg10 > best_avg:
            best_avg = avg10
            policy.save(f"{SAVE_DIR}/best_policy.json")

        # Save checkpoint every 10 episodes
        if episode % 10 == 0:
            policy.save(f"{SAVE_DIR}/policy_ep{episode}.json")

    # Final save
    policy.save(f"{SAVE_DIR}/final_policy.json")

    # Save training history
    with open(f"{SAVE_DIR}/training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    elapsed = time.time() - start
    print("\n" + "="*55)
    print("  TRAINING COMPLETE")
    print("="*55)
    print(f"  Episodes : {N_EPISODES}")
    print(f"  Best avg : {best_avg:.4f}")
    print(f"  Runtime  : {elapsed:.1f}s")
    print(f"  Policy   : {SAVE_DIR}/best_policy.json")
    print(f"  History  : {SAVE_DIR}/training_history.json")
    print("\nNext: python evaluate.py")

    # Print final weights
    print("\n  Learned policy weights:")
    print(f"    priority_weight : {policy.priority_weight:.3f}")
    print(f"    deadline_weight : {policy.deadline_weight:.3f}")
    print(f"    skill_weight    : {policy.skill_weight:.3f}")
    print(f"    load_weight     : {policy.load_weight:.3f}")


if __name__ == "__main__":
    train()