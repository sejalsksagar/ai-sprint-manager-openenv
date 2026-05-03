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
| `inference.py` | R1 LLM agent script | Change model/strategy |
| `inference_r2.py` | R2 LLM agent script (60-day) | Change model/strategy |
| `train_llm.py` | SFT + GRPO training pipeline | Change hyperparameters |
| `openenv.yaml` | OpenEnv spec metadata | Update task list |

---

## 🖥️ UI Components — Round 1 (Sprint Manager Tab)

The UI has two main tabs. Here is every component explained.

### Top Controls Row

```
[🎯 Sprint Scenario ▾]  [🔄 Reset Sprint]  [🤖 Auto-Assign All]
```

| Component | What it does |
|-----------|-------------|
| **Sprint Scenario dropdown** | Choose `easy_sprint` (3 devs, 5 tasks), `medium_sprint` (4 devs, 8 tasks, random events), or `hard_sprint` (5 devs, 12 tasks, urgent bugs). Always reset after changing. |
| **🔄 Reset Sprint** | Starts a new fresh episode. Clears the board, resets reward history, sets day to 1. Click this first before anything else. |
| **🤖 Auto-Assign All** | Runs the rule-based heuristic (skill-match + priority sort) to auto-assign every backlog task in one click. Useful to see what a decent baseline looks like. |

### Main Board Row

```
┌─────────────────────────┬───────────────────┐
│  📋 Sprint Board         │  👥 Team Workload  │
│  (tasks by status)       │  🎯 Skill Guide    │
│                          │  📊 Sprint Metrics │
└─────────────────────────┴───────────────────┘
```

**📋 Sprint Board** — the main view. Shows all tasks grouped by status:
- `🔄 IN PROGRESS` — assigned to a dev, progress bar filling each day
- `📋 BACKLOG` — not yet assigned, waiting
- `✅ DONE` — completed before deadline
- `❌ MISSED` — deadline passed without completion
- `🚫 BLOCKED` — waiting on a dependency

Each task card shows: type emoji | task ID | name | priority | effort (story points) | deadline day | required skill | assigned dev | progress bar.

**👥 Team Workload** — one row per developer showing:
- Status: ✅ available, 🤒 sick/absent
- Load bar: how many story points assigned vs capacity
- 🟢FREE / 🟡BUSY / 🔴FULL colour indicator

**🎯 Skill → Dev Guide** — quick lookup table. Tells you which dev has the right skill for a task. ✅ = available and has capacity, ❌ = full or sick.

**📊 Sprint Metrics** — cumulative reward, workload balance score, and task counts at a glance.

### Charts Row

```
┌───────────────────────┬───────────────────────┐
│  📈 Reward History     │  📊 Task Status        │
│  sparkline + last 10  │  bar chart by status   │
└───────────────────────┴───────────────────────┘
```

**📈 Reward History** — two sparklines: one for cumulative reward (should go up), one for per-step reward (varies). Below that, the last 10 steps are shown as text bars so you can see which actions gave positive or negative reward.

**📊 Task Status** — visual breakdown of task counts: done / in progress / backlog / missed / blocked. Includes a completion percentage bar.

### Manual Action Row

```
[Action ▾] [Task ID] [Dev ID] [Priority ▾] [▶️ Take Action]
```

| Field | What to put |
|-------|------------|
| **Action** | `assign`, `reassign`, `reprioritize`, `unblock`, or `skip` |
| **Task ID** | e.g. `T1`, `T3` — must match a task on the board |
| **Dev ID** | e.g. `dev1`, `dev2` — must match a developer |
| **Priority** | 1–5 only for `reprioritize` action, leave blank otherwise |

**📜 Event Log** below the action row shows what just happened — task completions, dev absences, bugs, reward amount.

---

## 🖥️ UI Components — Round 2 (Project Manager Tab)

### Top Controls Row

```
[🎯 Project Scenario ▾]  [🔄 Reset Project]  [🤖 Auto-Assign Sprint]  [⏩ Advance Day]
```

| Component | What it does |
|-----------|-------------|
| **Project Scenario dropdown** | `project_easy` (25 tasks / 6 sprints), `project_medium` (30 tasks), `project_hard` (40 tasks). |
| **🔄 Reset Project** | Starts fresh 60-day project. Always click this first. |
| **🤖 Auto-Assign Sprint** | Assigns all backlog tasks for the *current sprint only* using skill-match heuristic. Does NOT advance the day. |
| **⏩ Advance Day** | Sends a `skip` action — advances one day without assigning. Use to let the day tick forward, release scheduled instructions, and let in-progress tasks make progress. |

### Three-Column Board Row

```
┌──────────────┬────────────────────┬──────────────┐
│ 🗓️ Timeline   │ 📋 Sprint Board     │ 👥 Team       │
│ 6 sprints    │ current sprint only │ Workload     │
└──────────────┴────────────────────┴──────────────┘
```

**🗓️ Sprint Timeline** — shows all 6 sprints at a glance:
- ✅ = sprint completed ≥70% delivery
- ⚠️ = 40–69% delivery
- ❌ = below 40% delivery
- 🏃 = current sprint (with day progress bar)
- ⏳ = future sprint

Also shows the overall project delivery bar (all 6 sprints combined).

**📋 Current Sprint Board** — same format as R1 board but scoped to the current sprint's tasks. Shows dependency info (`Deps: T01,T02`) so you know which tasks are blocked.

**👥 Team Workload** — same as R1 but also shows `⚠️prod=0.8` warning when a developer's productivity has dropped due to tech debt.

### Metrics Row

```
┌──────────────────────┬───────────────────┬──────────────────┐
│ 📋 Instruction Queue │ 🔴 Tech Debt       │ 📊 Project Metrics│
└──────────────────────┴───────────────────┴──────────────────┘
```

**📋 Instruction Queue** — shows stakeholder instructions that have been released. Each instruction has:
- ⚠️ = not yet followed (act on this!)
- ✅ = already followed
- Release day and target sprint

These are the instruction-following tasks that feed the `inst_score` metric. Ignoring them costs you 30% of the final score.

**🔴 Tech Debt Tracker** — lists every task that was missed at a sprint boundary. Each missed task permanently reduces team productivity by 2%. This is how cascade failures happen — miss 5 tasks in sprint 1 and every developer is 10% slower for the rest of the project.

**📊 Project Metrics** — all key numbers: cumulative reward, team balance, instruction-following score (0–1), tech debt count, average sprint score, task counts.

### Cross-Sprint Reward Chart

Shows per-sprint score bars (✅/⚠️/❌ thresholds) plus a cumulative sparkline across all 60 steps.


### R2 Manual Action Row

Same as R1, plus two new fields:
- **sprint_plan** action: batch-plan multiple tasks for the sprint in one call
- **Task IDs (sprint_plan)**: comma-separated list, e.g. `T01,T02,T03`

---

## 🔄 What Happens Each Step

```
Day 1 → Day 2 → Day 3 → ... → Day 10 → DONE (R1)
Day 1 → Day 2 → ... → Day 60 → DONE (R2, 6 sprints)
  ↑        ↑        ↑
agent    agent    agent
acts     acts     acts
```

**One step = one day:**

1. Agent receives observation (all tasks, all devs, current day, active instructions)
2. Agent picks an action
3. Environment validates the action (guard checks)
4. Developers work — progress increases on assigned tasks
5. Random events fire (dev sick, new bug, stall detection)
6. Instructions release if their day has come
7. Reward is calculated and returned
8. Repeat until done

---

## 💰 Reward Design — Why It Works for RL

```
Good actions  →  positive reward immediately
Bad actions   →  negative reward immediately
Task done on time  →  bonus
Task missed deadline  →  penalty
Sprint ends (R2)  →  sprint delivery score
Day 60 (R2)  →  final project score
```

Signal at every step = efficient RL training. A learning agent doesn't have to wait until the end to know if it's doing well.

---

## 🧪 Session Isolation — How to Test It's Working

Each browser session gets its own environment instance stored in `gr.State`. No shared global. Here is how to verify:

### Test 1: Two tabs don't share state

```bash
# Open two browser tabs both pointing to http://localhost:7860
# Tab A: Select easy_sprint → Reset Sprint → click Auto-Assign All
# Tab B: Select hard_sprint → Reset Sprint
# Expected: Tab B shows hard_sprint board, Tab A still shows easy_sprint board
# If they interfere: global env bug (not fixed)
```

### Test 2: API sessions are independent

```bash
# Terminal 1 — start a session
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task_name":"easy_sprint","seed":42}'
# Copy the episode_id from the response

# Terminal 2 — start a different session at the same time
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task_name":"hard_sprint","seed":99}'
# Copy that episode_id too

# Now step each session independently using their episode_ids
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{"episode_id":"<id_from_terminal_1>","action":{"action_type":"assign","task_id":"T1","dev_id":"dev1"}}'

curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{"episode_id":"<id_from_terminal_2>","action":{"action_type":"skip"}}'

# Expected: each returns state from its own episode, no cross-contamination
```

### Test 4: Concurrent UI users (load test)

```python
# run_concurrent_test.py
import threading, requests, time

def user_session(user_id, scenario):
    base = "http://localhost:7860"
    r = requests.post(f"{base}/reset", json={"task_name": scenario, "seed": user_id})
    eid = r.json()["episode_id"]
    print(f"User {user_id} ({scenario}) episode_id: {eid}")
    for step in range(5):
        r = requests.post(f"{base}/step", json={
            "episode_id": eid,
            "action": {"action_type": "skip"}
        })
        day = r.json()["observation"]["current_day"]
        print(f"User {user_id} step {step+1}: day={day}")
        time.sleep(0.1)

threads = [
    threading.Thread(target=user_session, args=(i, s))
    for i, s in enumerate(["easy_sprint", "medium_sprint", "hard_sprint"])
]
for t in threads: t.start()
for t in threads: t.join()
# Expected: each user's day counter increments independently
# If days jump: shared state bug
```

---

## ✅ How To Know Everything Is Working

### Quick 60-second check

```bash
# 1. Start server
python ui.py

# 2. Health check
curl http://localhost:7860/health
# Expected: {"status":"ok","env":"ai-sprint-manager"}

# 3. Reset with episode_id
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task_name":"easy_sprint","seed":42}'
# Expected: JSON with episode_id, current_day=1, 5 tasks in backlog

# 4. Step (use episode_id from step 3)
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{"episode_id":"<your-id>","action":{"action_type":"assign","task_id":"T1","dev_id":"dev1"}}'
# Expected: reward around +1.2, T1 now in_progress

# 5. Run trained agent
python inference.py
# Expected: [LLM] and [FB] tagged lines, attribution summary at end

# 6. Validate
openenv validate
# Expected: [OK] ai-sprint-manager: Ready
```

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
| Per-session isolation | ✅ Each UI user has independent env |
| Attribution-transparent | ✅ [LLM] / [FB] tags on every agent step |

---

## 📊 Actual Benchmark Scores

```
Round 1 — Single Sprint (10 steps)
  easy_sprint   : Rule-based 0.99 | Llama-8B 0.99 | Trained-1.5B 0.99
  medium_sprint : Rule-based 0.67 | Llama-8B 0.50 | Trained-1.5B 0.67
  hard_sprint   : Rule-based 0.37 | Llama-8B 0.29 | Trained-1.5B 0.29
  average       : Rule-based 0.68 | Llama-8B 0.59 | Trained-1.5B 0.65

Round 2 — 60-Day Project (6 sprints)
  project_easy   : Rule-based 0.23 | Llama-8B 0.26 | Trained-1.5B 0.26
  project_medium : Rule-based 0.17 | Llama-8B 0.16 | Trained-1.5B 0.20 ← best
  project_hard   : Rule-based 0.10 | Llama-8B 0.09 | Trained-1.5B 0.08
  average        : Rule-based 0.17 | Llama-8B 0.17 | Trained-1.5B 0.18 ← best overall
```

The trained 1.5B model achieves the highest R2 average. `project_medium` is the clearest win (+19.7% over rule-based, +23.1% over Llama). To understand how much of this is the trained model vs the fallback, check the [LLM%] in the attribution summary when you run the agent.