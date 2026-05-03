---
title: AI Sprint Manager
emoji: 🤖
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
tags: [openenv, reinforcement-learning, agile, sprint-management, fastapi, gradio]
---

# 🤖 AI Sprint Manager — OpenEnv

> **A reinforcement learning environment where an AI agent acts as a Tech Lead managing agile software sprints — from a single 10-day sprint (Round 1) to a full 60-day, 6-sprint project (Round 2).**

---

## 🎯 What Is This?

Modern software teams spend enormous time on sprint planning decisions:
- Which developer gets which task?
- What do you do when someone goes sick mid-sprint?
- How do you handle a stakeholder instruction that arrives on day 23 and invalidates your whole plan?

This environment simulates these real-world decisions so an AI agent can **learn optimal sprint management strategies** through reinforcement learning.

The agent plays the role of a Tech Lead. Each step it observes the full sprint state (tasks, developers, workloads, deadlines, active instructions) and takes an action. The environment responds with a reward signal that guides learning.

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────┐
│     RL Agent / LLM / Training Loop      │
│   inference.py (R1) / inference_r2.py   │
└──────────────────┬──────────────────────┘
                   │ HTTP  reset / step / state
                   ▼
┌─────────────────────────────────────────┐
│         FastAPI Server (port 7860)      │
│    /reset  /step  /state  /health       │
│    /project/reset  /project/step  (R2)  │
└──────────────────┬──────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────┐
│      Sprint Environment (core logic)    │
│  • Task/developer simulation            │
│  • Reward calculation (3-layer for R2)  │
│  • Random events (bugs, absences)       │
│  • 3 graders: easy / medium / hard      │
│  • Tech debt & instruction following    │
└──────────────────┬──────────────────────┘
                   │ data loaded from
                   ▼
┌─────────────────────────────────────────┐
│      data/sprint_data.json              │
│  (customizable — bring your own data!)  │
└─────────────────────────────────────────┘
```

---

## 🎮 Live Demo

1. Select a sprint scenario (easy / medium / hard)
2. Click **🔄 Reset Sprint**
3. Use the **Skill → Dev Guide** to assign tasks correctly
4. Or click **🤖 Auto-Assign All** to let the system decide
5. Watch the reward history and task status update in real time

The **Round 2 tab** shows the full 60-day project view: sprint timeline, instruction queue, tech debt tracker, and live LLM agent log.

---

## 📐 Action Space

| Field | Type | Values |
|---|---|---|
| `action_type` | string | `assign`, `reassign`, `reprioritize`, `unblock`, `skip` |
| `task_id` | string | Task ID e.g. `"T1"`, `"T6"` |
| `dev_id` | string | Developer ID e.g. `"dev1"`, `"dev3"` |
| `new_priority` | int | 1–5 (1=highest, for reprioritize only) |

## 📊 Observation Space

| Field | Type | Description |
|---|---|---|
| `current_day` | int | Day in sprint (1–10) or project (1–60) |
| `current_sprint` | int | Active sprint number (R2 only) |
| `sprint_length` | int | Total sprint length |
| `developers` | list | Each dev's skill, capacity, load, tasks, availability |
| `tasks` | list | Each task's type, priority, effort, deadline, status, progress, deps |
| `reward` | float | Step reward |
| `cumulative_reward` | float | Total reward this episode |
| `tasks_completed/missed/in_progress/backlog` | int | Status counts |
| `workload_balance_score` | float | 0=unbalanced, 1=perfect |
| `instruction_following_score` | float | Running avg: 1.0=all instructions followed (R2) |
| `tech_debt` | list | Tasks that missed sprint deadlines (R2) |
| `active_instructions` | list | Current stakeholder instructions (R2) |
| `events` | list | Events that just happened (completions, misses, absences) |
| `done` | bool | Whether episode is complete |

---

## 🎯 Tasks (Scenarios)

### Round 1 — Single Sprint

| ID | Difficulty | Devs | Tasks | Random Events |
|---|---|---|---|---|
| `easy_sprint` | 🟢 Easy | 3 | 5 | None |
| `medium_sprint` | 🟡 Medium | 4 | 8 | Dev absences, bugs expire |
| `hard_sprint` | 🔴 Hard | 5 | 12 | Urgent bugs mid-sprint, cascading failures |

### Round 2 — Full 60-Day Project

| ID | Difficulty | Devs | Tasks | Features |
|---|---|---|---|---|
| `project_easy` | 🟢 Easy | 4 | 25 | Cross-sprint deps, time-released instructions |
| `project_medium` | 🟡 Medium | 5 | 30 | Absences, burnout, tech debt |
| `project_hard` | 🔴 Hard | 6 | 40 | All of the above + cascading failures |

---

## 📈 Benchmark Scores

### Round 1

| Model | easy_sprint | medium_sprint | hard_sprint | Average |
|---|---|---|---|---|
| Rule-Based | 0.9900 | 0.6667 | 0.3716 | **0.6761** |
| Llama-3.1-8B (baseline) | 0.9900 | 0.5000 | 0.2858 | 0.5919 |
| **Trained Qwen2.5-1.5B** | **0.9900** | **0.6667** | 0.2858 | **0.6475** |

### Round 2

| Model | project_easy | project_medium | project_hard | Average |
|---|---|---|---|---|
| Rule-Based | 0.2308 | 0.1693 | 0.1049 | 0.1683 |
| Llama-3.1-8B (baseline) | 0.2567 | 0.1647 | 0.0870 | 0.1694 |
| **Trained Qwen2.5-1.5B** | **0.2567** | **0.2027** | 0.0750 | **0.1781** |

The trained 1.5B model achieves the **highest average in Round 2**, beating the 5× larger Llama baseline. The `project_medium` improvement (+20% over rule-based, +23% over Llama) reflects effective instruction-following learned through GRPO.

---

## 💰 Reward Function

### Round 1 (Step Rewards)

| Event | Reward |
|---|---|
| Assign task (skill match) | +0.8 to +1.3 |
| Assign task (skill mismatch penalty) | +0.1 to +0.6 |
| Wrong skill / over capacity | -0.15 |
| Task completed on time | +0.5 to +2.5 |
| Task completed late | +0.1 |
| Task missed deadline | -0.3 to -1.5 |
| Urgent bug missed | -0.25 extra |
| Skip (no action) | -0.05 |
| Final score bonus | score × 10.0 |

### Round 2 (Three-Layer Rewards)

| Layer | Trigger | Signal |
|---|---|---|
| Step reward | Every action | Same as R1 + instruction weight |
| Sprint boundary | Every 10 days | ±delivery rate, burnout penalty, miss penalty |
| Final score | Day 60 | delivery_rate×0.55 + inst_score×0.30 + team_health×0.15 |

GRPO training signal: `combined = step_norm × 0.6 + inst_score × 0.4`

---

## 🛡️ Inference: Guard / Loop / Stall System

The inference layer has three defence mechanisms to handle invalid or degenerate model outputs:

**GUARD** — validates every action before executing it. Checks: task not already assigned this episode, dependency tasks are complete, task is in backlog, developer has capacity. Fires fallback on any violation.

**LOOP** — detects repetition. If the model emits the same `(action_type, task_id)` pair twice consecutively, overrides with rule-based fallback. Prevents the greedy-decoding lock-on failure mode.

**STALL** — detects long-running tasks. If a task has been in_progress for more than `max(5, effort×1.5)` days, reassigns to a higher-productivity developer. Unblocks frozen dependency chains before they cascade.

---

## 🔌 API Reference

```bash
# Health check
GET /health → {"status": "ok", "env": "ai-sprint-manager"}

# Round 1
POST /reset    Body: {"task_name": "easy_sprint", "seed": 42}
POST /step     Body: {"action": {"action_type": "assign", "task_id": "T1", "dev_id": "dev1"}}
GET  /state
GET  /tasks

# Round 2 (multi-sprint project)
POST /project/reset    Body: {"task_name": "project_easy", "seed": 42}
POST /project/step     Body: {"action": {"action_type": "assign", "task_id": "T01", "dev_id": "dev1"}}
GET  /project/state
```

---

## 🐍 Python Client Usage

```python
from client import SprintEnvClient
from sprint_env.models import SprintAction

# Connect to live Space
with SprintEnvClient(base_url="https://sejal-k-ai-sprint-manager.hf.space") as env:
    obs = env.reset(task_name="medium_sprint", seed=42)

    while not obs["done"]:
        action = SprintAction(
            action_type="assign",
            task_id="T1",
            dev_id="dev1",
        )
        result = env.step(action)
        print(result)  # StepResult(reward=+1.20, done=False, day=2, completed=0)
        obs = result.observation
```

---

## 🗂️ Project Structure

```
ai-sprint-manager-openenv/
├── openenv.yaml              # OpenEnv spec metadata
├── pyproject.toml            # Project dependencies
├── Dockerfile                # Container definition
├── requirements.txt          # Python dependencies
├── inference.py              # R1 LLM agent (trained model + guard/loop)
├── inference_r2.py           # R2 LLM agent (60-day + guard/loop/stall)
├── train_llm.py              # SFT + GRPO training pipeline
├── client.py                 # Typed Python client
├── project_client.py                 # R2 Typed Python client (for RL training)
├── ui.py                     # Gradio UI + FastAPI combined server
├── start.sh                  # Container startup script
│
├── data/
│   └── sprint_data.json      # All scenario data (customizable!)
│
├── sprint_env/
│   ├── __init__.py
│   ├── models.py             # Pydantic Action/Observation/State
│   ├── tasks.py              # Task & Developer dataclasses
│   ├── environment.py        # Core RL environment logic (R1)
│   ├── project_graders.py            # R2 Scoring functions (easy/medium/hard)
│   ├── project_models.py             # R2 Pydantic Action/Observation/State
│   ├── project_environment.py     # Extended environment (R2, 60-day)
│   ├── project_data_loader.py        # R2 JSON data loader with caching
│   └── data_loader.py        # JSON data loader with caching
│
└── server/
    ├── __init__.py
    ├── app.py                # OpenEnv-compliant FastAPI server entry
    └── project_app.py         # R2 OpenEnv-compliant FastAPI server entry
```

---

## 🤖 Training Your Own Model

```python
# Example training loop skeleton (TRL/GRPO compatible)
from client import SprintEnvClient
from sprint_env.models import SprintAction

env = SprintEnvClient(base_url="http://localhost:7860")

for episode in range(1000):
    obs = env.reset(task_name="medium_sprint")
    trajectory = []

    while not obs["done"]:
        action = policy.sample(obs)           # your policy here
        result = env.step(action)
        trajectory.append((obs, action, result.reward))
        obs = result.observation

    policy.update(trajectory)                 # GRPO/PPO update
```

Full training pipeline (SFT warm-up → GRPO curriculum):

```bash
# Smoke test first (no GPU needed)
python train_llm.py --smoke-test

# Full training (T4 GPU)
python train_llm.py --phase both --episodes 300 \
       --sft-epochs 2 --gpu-tier t4 \
       --output results/trained_model --push
```

---

## 🔧 Bring Your Own Data

Edit `data/sprint_data.json`:

```json
{
  "scenarios": {
    "my_custom_sprint": {
      "description": "My team's actual sprint",
      "difficulty": "medium",
      "developers": [
        {"id": "dev1", "name": "Your Name", "skill": "backend", "capacity": 5, "productivity": 1.0}
      ],
      "tasks": [
        {"id": "T1", "name": "Your Task", "task_type": "feature", "priority": 1,
         "effort": 3, "deadline": 5, "required_skill": "backend"}
      ]
    }
  }
}
```

Or point to your own file:
```bash
export SPRINT_DATA_PATH=/path/to/your/data.json
python ui.py
```

---

## 🚀 Setup & Run

```bash
# Clone
git clone https://github.com/sejalsksagar/ai-sprint-manager-openenv.git
cd ai-sprint-manager-openenv

# Install
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Configure
cp .env.example .env
# Edit .env with your HF_TOKEN

# Run locally
python ui.py
# Open http://localhost:7860

# Docker
docker build -t ai-sprint-manager .
docker run -p 7860:7860 ai-sprint-manager

# Run inference (R1)
python inference.py

# Run inference (R2 — full 60-day project)
python inference_r2.py
```

---

## 👥 Team

Built for the **Meta PyTorch OpenEnv Hackathon × SST | India AI Hackathon '26**