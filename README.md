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

> **A reinforcement learning environment where an AI agent acts as a Tech Lead managing agile software sprints.**

[![OpenEnv](https://img.shields.io/badge/OpenEnv-Compatible-blue)](https://github.com/meta-pytorch/OpenEnv)
[![HF Space](https://img.shields.io/badge/🤗-Live%20Demo-yellow)](https://huggingface.co/spaces/sejal-k/ai-sprint-manager)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 🎯 What Is This?

Modern software teams spend enormous time on sprint planning decisions:
- Which developer gets which task?
- What do you do when someone goes sick mid-sprint?
- How do you handle an urgent production bug that appears on day 5?

This environment simulates these real-world decisions so an AI agent can **learn optimal sprint management strategies** through reinforcement learning.

The agent plays the role of a Tech Lead. Each step it observes the full sprint state (tasks, developers, workloads, deadlines) and takes an action. The environment responds with a reward signal that guides learning.

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────┐
│     RL Agent / LLM / Training Loop      │
│         (uses client.py)                │
└──────────────────┬──────────────────────┘
                   │ HTTP  reset / step / state
                   ▼
┌─────────────────────────────────────────┐
│         FastAPI Server (port 7860)      │
│    /reset  /step  /state  /health       │
└──────────────────┬──────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────┐
│      Sprint Environment (core logic)    │
│  • Task/developer simulation            │
│  • Reward calculation                   │
│  • Random events (bugs, absences)       │
│  • 3 graders: easy / medium / hard      │
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

👉 **[Try it on Hugging Face Spaces](https://huggingface.co/spaces/sejal-k/ai-sprint-manager)**

1. Select a sprint scenario (easy / medium / hard)
2. Click **🔄 Reset Sprint**
3. Use the **Skill → Dev Guide** to assign tasks correctly
4. Or click **🤖 Auto-Assign All** to let the system decide
5. Watch the reward history and task status update in real time

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
| `current_day` | int | Day in sprint (1–10) |
| `sprint_length` | int | Total sprint length |
| `developers` | list | Each dev's skill, capacity, load, tasks, availability |
| `tasks` | list | Each task's type, priority, effort, deadline, status, progress |
| `reward` | float | Step reward |
| `cumulative_reward` | float | Total reward this episode |
| `tasks_completed/missed/in_progress/backlog` | int | Status counts |
| `workload_balance_score` | float | 0=unbalanced, 1=perfect |
| `events` | list | Events that just happened (completions, misses, absences) |
| `done` | bool | Whether episode is complete |

---

## 🎯 Tasks (Scenarios)

| ID | Difficulty | Devs | Tasks | Random Events |
|---|---|---|---|---|
| `easy_sprint` | 🟢 Easy | 3 | 5 | None |
| `medium_sprint` | 🟡 Medium | 4 | 8 | Dev absences, bugs expire |
| `hard_sprint` | 🔴 Hard | 5 | 12 | Urgent bugs mid-sprint, cascading failures |

### Baseline Scores (meta-llama/Llama-3.1-8B-Instruct)

| Task | Score |
|---|---|
| `easy_sprint` | 1.00 ████████████████████ |
| `medium_sprint` | 0.42 ████████ |
| `hard_sprint` | 0.00 |
| **Average** | **0.47** |

---

## 💰 Reward Function

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

---

## 🔌 API Reference

```bash
# Health check
GET /health → {"status": "ok", "env": "ai-sprint-manager"}

# Start new episode
POST /reset
Body: {"task_name": "easy_sprint", "seed": 42}

# Take one action
POST /step
Body: {"action": {"action_type": "assign", "task_id": "T1", "dev_id": "dev1"}}

# Get full state
GET /state

# List scenarios
GET /tasks
```

---

## 🐍 Python Client Usage

```python
from client import SprintEnvClient
from sprint_env.models import SprintAction

# Connect to live Space
with SprintEnvClient(base_url="https://sejal-k-ai-sprint-manager.hf.space") as env:
    # Reset
    obs = env.reset(task_name="medium_sprint", seed=42)

    # Agent loop
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
├── inference.py              # Baseline LLM agent script
├── client.py                 # Typed Python client (for RL training)
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
│   ├── environment.py        # Core RL environment logic
│   ├── graders.py            # Scoring functions (easy/medium/hard)
│   └── data_loader.py        # JSON data loader with caching
│
└── server/
    ├── __init__.py
    └── app.py                # OpenEnv-compliant FastAPI server entry
```

---

## 🔧 Bring Your Own Data

Don't want to use our sample scenarios? Edit `data/sprint_data.json`:

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

# Run inference
python inference.py
```

---

## 🤖 Can an RL Agent Learn From This?

Yes. The environment is designed for policy gradient training (GRPO, PPO):

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

The shaped reward function provides learning signal at every step — not just at episode end — which is critical for efficient RL training.

---

## 📋 Submission Checklist

- ✅ HF Space deploys and responds to `/reset`
- ✅ OpenEnv spec compliance (`openenv validate` passes)
- ✅ Docker build works
- ✅ Baseline inference script runs and produces `[START]/[STEP]/[END]` output
- ✅ 3 tasks with graders (easy/medium/hard), scores 0.0–1.0
- ✅ Meaningful reward function (shaped, not sparse)
- ✅ `openenv.yaml` present and valid
- ✅ `pyproject.toml` with `openenv-core>=0.2.0`
- ✅ `client.py` for RL training integration
- ✅ External data file (`data/sprint_data.json`) — bring your own data

---

## 👥 Team

Built for the **Meta PyTorch OpenEnv Hackathon x SST | India AI Hackathon '26**