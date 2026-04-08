# 📖 Complete Guide — AI Sprint Manager OpenEnv

**What we built, how to test it, how to demo it, and how to explain it.**

---

## 🤔 What Are We Building — Plain English

Imagine you're the Tech Lead of a software company. Every 2 weeks (called a "sprint"), your team gets a list of tasks — new features, bug fixes, tech debt. Your job is to decide:

- Which developer gets which task?
- What's most urgent when a new bug appears?
- What do you do when someone calls in sick?
- How do you avoid burning out your best developer?

**We built a simulation of this scenario** so an AI agent can practice making these decisions thousands of times and learn to get better — just like how AlphaGo learned to play Go by playing millions of games against itself.

The AI plays the role of the Tech Lead. It looks at the sprint state and decides what to do. The environment tells it how well it did (reward). Over time, it learns better strategies.

---

## 🏗️ How It's Built — Layer by Layer

```
YOU / AI AGENT
  ↓  makes decisions (assign, skip, reprioritize...)
FASTAPI SERVER  (ui.py / server/app.py)
  ↓  receives actions, returns results
SPRINT ENVIRONMENT  (sprint_env/environment.py)
  ↓  core logic: tracks tasks, devs, days, rewards
DATA  (data/sprint_data.json)
  ↓  tasks and developers — fully customizable
GRADIO UI  (ui.py)
     visual sprint board, charts, controls
```

---

## 📁 What Each File Does

| File | Purpose | Change it to... |
|------|---------|----------------|
| `data/sprint_data.json` | All scenario data | Add your own tasks/devs |
| `sprint_env/models.py` | Data contracts (Action/Observation/State) | Add new fields |
| `sprint_env/tasks.py` | Task & Developer classes | Add new task types |
| `sprint_env/environment.py` | Core RL logic | Change simulation rules |
| `sprint_env/graders.py` | Scoring (easy/medium/hard) | Change scoring weights |
| `sprint_env/data_loader.py` | Loads JSON data with caching | Point to custom data |
| `server/app.py` | OpenEnv HTTP API entry point | Add new endpoints |
| `client.py` | Typed Python client for RL training | Use in training scripts |
| `ui.py` | Gradio UI + combined server | Change UI layout |
| `inference.py` | Baseline LLM agent | Change model/strategy |
| `openenv.yaml` | OpenEnv spec metadata | Update task list |

---

## 🔄 What Happens Each Step

```
Day 1 → Day 2 → Day 3 → ... → Day 10 → DONE
  ↑        ↑        ↑
agent    agent    agent
acts     acts     acts
```

**One step = one day in the sprint:**

1. Agent receives observation (all tasks, all devs, current day)
2. Agent picks an action (e.g. "assign T1 to dev1")
3. Environment validates the action
4. Developers work on assigned tasks — progress increases
5. Random events fire (dev goes sick, new bug appears)
6. Reward is calculated and returned
7. Repeat until Day 10 or all tasks resolved

---

## 💰 Reward Design — Why It Works for RL

The reward function is **shaped** (signal at every step) not **sparse** (only at the end):

```
Good actions  →  positive reward immediately
Bad actions   →  negative reward immediately
Task done on time  →  bonus
Task missed deadline  →  penalty
Sprint ends  →  final_score × 10 bonus
```

This means a learning agent gets feedback on every single decision — critical for efficient RL training with GRPO, PPO, or any policy gradient algorithm.

---

## ✅ How To Know Everything Is Working

### Quick 60-second check

```bash
# 1. Start server
python ui.py

# 2. In another terminal — health check
curl http://localhost:7860/health
# Expected: {"status":"ok","env":"ai-sprint-manager"}

# 3. Reset
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task_name":"easy_sprint","seed":42}'
# Expected: JSON with current_day=1, 5 tasks in backlog

# 4. Step
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{"action":{"action_type":"assign","task_id":"T1","dev_id":"dev1"}}'
# Expected: reward around +1.2, T1 now in_progress

# 5. Validate
openenv validate
# Expected: [OK] ai-sprint-manager: Ready for multi-mode deployment

# 6. Run inference
python inference.py
# Expected: [START]/[STEP]/[END] lines, scores for all 3 tasks
```

### Full test checklist

| # | Test | How | Pass Condition |
|---|------|-----|---------------|
| 1 | Server health | `GET /health` | `{"status":"ok"}` |
| 2 | Reset works | `POST /reset` | day=1, tasks in backlog |
| 3 | Assign works | `POST /step` assign T1→dev1 | reward +1.2, T1 in_progress |
| 4 | Skill mismatch rejected | Assign backend task to frontend dev | reward -0.15, error message |
| 5 | Sprint ends | 10 skip steps | `done: true` |
| 6 | Grader runs | Check final_score in info | value between 0.0 and 1.0 |
| 7 | OpenEnv valid | `openenv validate` | `[OK]` message |
| 8 | Inference output | `python inference.py` | `[START]` `[STEP]` `[END]` lines |
| 9 | Docker build | `docker build .` | Exit code 0 |
| 10 | Docker run | `docker run -p 7860:7860 ...` then health | `{"status":"ok"}` |
| 11 | Live Space | `curl https://sejal-k-ai-sprint-manager.hf.space/health` | `{"status":"ok"}` |
| 12 | UI loads | Open http://localhost:7860 | Gradio UI visible |
| 13 | UI reset | Click Reset Sprint | Sprint board populates |
| 14 | Auto-assign | Click Auto-Assign All | Tasks move to in_progress |
| 15 | Reward chart | Take 3+ actions | Sparkline appears |

---

## 🎤 Project Demo Script (10 minutes)

### Before Demo
```bash
python ui.py
# Open http://localhost:7860 in browser — full screen
# Have terminal with inference.py output ready
```

### [0:00 — 1:30] The Problem
> "Software teams waste hours every sprint on planning. Which developer gets which task? What happens when someone goes sick? What if a critical bug appears on day 5? These decisions directly affect delivery speed and developer burnout."

> "We built an RL environment that simulates exactly this — so an AI agent can learn to make these decisions better."

### [1:30 — 3:00] Show the UI
- Select `easy_sprint` → **🔄 Reset Sprint**
- Point to sprint board: *"5 tasks in backlog, 3 developers, 10 day sprint"*
- Point to Skill Guide: *"This tells you which dev is right for which task. Backend tasks need Alice, frontend tasks need Bob."*

### [3:00 — 4:30] Manual Play — Good vs Bad Decision
- Assign T3 (frontend) → dev1 (backend): *"Wrong skill — negative reward, task rejected"*
- Assign T3 (frontend) → dev2 (frontend): *"Correct match — positive reward, task starts"*
- Point to reward chart: *"See the reward signal? This is exactly what the AI learns from."*

### [4:30 — 6:00] Auto-Assign
- Click **🤖 Auto-Assign All**
- *"Rule-based auto-assign picks the best skill match for every task."*
- Click **▶️ Take Action** (skip) a few times
- *"Each day the sprint advances, tasks progress, deadlines approach."*

### [6:00 — 7:30] Hard Sprint
- Reset with `hard_sprint`
- *"12 tasks, 5 developers, random events — developers go sick, urgent bugs appear mid-sprint."*
- Auto-assign, then skip a few times
- When a 🚨 event fires: *"There — urgent bug on day 4. A trained agent needs to react and reassign resources."*

### [7:30 — 9:00] Inference Output
- Show terminal with inference.py output
- *"This is our Llama 3.1 baseline running against all 3 scenarios automatically."*
- Point to structured output: *"`[START]` `[STEP]` `[END]` — machine-parseable format the judges require."*
- Point to scores: *"Easy sprint perfect score of 1.0 — validates the environment works. Hard sprint 0.0 — shows it genuinely challenges frontier models."*

### [9:00 — 10:00] Technical Highlights
> "What makes this submission stand out:"

- **Real-world domain** — not CartPole, not a game — actual software engineering
- **External data file** — `data/sprint_data.json` — anyone can plug in their own team
- **Typed Python client** — `client.py` makes it plug-and-play with TRL, Stable-Baselines3
- **OpenEnv compliant** — passes all 3 validation checks
- **Shaped rewards** — signal at every step, enables efficient RL training

---

## 🐛 Common Issues & Fixes

| Problem | Cause | Fix |
|---------|-------|-----|
| `ModuleNotFoundError` | Missing package | `pip install -r requirements.txt` |
| Port 7860 in use | Other process | Kill it or change port in ui.py |
| `401 Unauthorized` in inference | Bad HF token | Regenerate at hf.co/settings/tokens |
| Validate step 3 fails | openenv not in PATH | Activate venv before running script |
| Tasks not progressing | No devs assigned | Auto-Assign or assign manually |
| Score always 0.0 | All tasks missed | Assign earlier, prioritize urgent tasks |
| Docker timeout | venv in context | Check `.dockerignore` has `venv/` |

---

## 🔬 Is This a Real RL Environment?

| Criterion | Our Environment |
|-----------|----------------|
| Sequential decisions | ✅ Each day depends on previous assignments |
| Large state space | ✅ Tasks × Developers × Day — combinatorial |
| Non-trivial action space | ✅ 5 types × 12 tasks × 5 devs |
| Shaped reward | ✅ Signal every step, not just episode end |
| Stochastic transitions | ✅ Random dev absences, mid-sprint bugs |
| Clean episode boundaries | ✅ reset() gives fresh state every time |
| Partial observability | ✅ Agent can't predict future events |
| Trainable with RL | ✅ GRPO / PPO / any policy gradient |

**A trained RL agent (not zero-shot LLM) should score 0.7+ on medium and 0.4+ on hard** after sufficient training — currently it scores 0.42 and 0.0 with baseline Llama 3.1. That's the gap RL training is meant to close.

---

## 📊 Baseline Score Interpretation

```
easy_sprint:   1.00  ← LLM figured out skill matching perfectly
medium_sprint: 0.42  ← Partial success, random events hurt performance
hard_sprint:   0.00  ← Cascade failures overwhelm baseline LLM
average:       0.47
```

These scores are intentional and show the difficulty curve works correctly:
- Easy solvable by any agent → environment is correct
- Medium shows partial success → reward shaping works
- Hard challenges frontier models → difficulty is genuine