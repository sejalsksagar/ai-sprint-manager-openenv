# 🤖 AI Sprint Manager — OpenEnv

An RL environment where an AI agent acts as a **Tech Lead** managing agile sprints — assigning tasks to developers, balancing workloads, and responding to unexpected events.

## Why This Exists

Software teams lose countless hours to poor sprint planning. This environment enables RL agents to learn optimal task allocation strategies in a realistic simulation of real engineering team dynamics.

## Action Space

| Field | Type | Description |
|---|---|---|
| `action_type` | string | `assign`, `reassign`, `reprioritize`, `unblock`, `skip` |
| `task_id` | string | ID of task to act on (e.g. `"T1"`) |
| `dev_id` | string | ID of developer (e.g. `"dev1"`) |
| `new_priority` | int 1-5 | New priority for reprioritize action |

## Observation Space

| Field | Description |
|---|---|
| `current_day` | Day in sprint (1–10) |
| `sprint_length` | Total sprint length |
| `developers` | List of developer states (skill, capacity, load, assigned tasks) |
| `tasks` | All tasks with status, priority, effort, deadline, progress |
| `reward` | Step reward |
| `tasks_completed/missed/in_progress/backlog` | Counts |
| `workload_balance_score` | 0=unbalanced, 1=perfectly balanced |
| `events` | List of events that just happened |

## Tasks

| ID | Difficulty | Description |
|---|---|---|
| `easy_sprint` | Easy | 3 devs, 5 tasks, no surprises. Baseline testing. |
| `medium_sprint` | Medium | 4 devs, 8 tasks, bugs + dev absences. |
| `hard_sprint` | Hard | 5 devs, 12 tasks, urgent bugs appear mid-sprint. |

### Baseline Scores (random agent)

| Task | Score |
|---|---|
| easy_sprint | ~0.30 |
| medium_sprint | ~0.22 |
| hard_sprint | ~0.12 |

## Setup
```bash
# Local
pip install -r requirements.txt
uvicorn server:app --host 0.0.0.0 --port 8000

# Docker
docker build -t ai-sprint-manager .
docker run -p 8000:8000 ai-sprint-manager

# Inference
export HF_TOKEN=your_token
export MODEL_NAME=meta-llama/Llama-3.1-8B-Instruct
export API_BASE_URL=https://router.huggingface.co/v1
python inference.py
```

## API
```bash
# Reset
curl -X POST http://localhost:8000/reset \
  -H "Content-Type: application/json" \
  -d '{"task_name": "easy_sprint", "seed": 42}'

# Step
curl -X POST http://localhost:8000/step \
  -H "Content-Type: application/json" \
  -d '{"action": {"action_type": "assign", "task_id": "T1", "dev_id": "dev1"}}'

# State
curl http://localhost:8000/state
```