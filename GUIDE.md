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
| `inference.py` | R1 LLM agent script (Kaggle/local) | Change model/strategy |
| `inference_r2.py` | R2 LLM agent script (Kaggle/local) | Change model/strategy |
| `train_llm.py` | SFT + GRPO training pipeline | Change hyperparameters |
| `openenv.yaml` | OpenEnv spec metadata | Update task list |

> **Note:** The trained LLM agent (`sejal-k/multi-sprint-model`) is not served in the live HF Spaces UI as of now. The demo supports manual actions and rule-based auto-assign.

---

## 🖥️ UI Overview

The UI has two tabs served at `http://localhost:7860` (or your HF Space URL). There is **no LLM agent button** in the deployed UI — all interaction is via manual actions or the rule-based Auto-Assign button.

---

## 🖥️ UI Components — Round 1 (Sprint Manager Tab)

### Top Controls Row

```
[🎯 Sprint Scenario ▾]  [🔄 Reset Sprint]  [🤖 Auto-Assign All]
```

| Component | What it does |
|-----------|-------------|
| **🎯 Sprint Scenario** | Choose `easy_sprint` (3 devs, 5 tasks), `medium_sprint` (4 devs, 8 tasks, random events), or `hard_sprint` (5 devs, 12 tasks, urgent bugs). Always reset after changing scenario. |
| **🔄 Reset Sprint** | Starts a fresh episode. Clears the board, resets reward history, sets day to 1. Click this first before anything else. |
| **🤖 Auto-Assign All** | Runs the rule-based heuristic (skill-match + priority sort) to assign every backlog task in one click. Assigns all remaining sprints too — useful as a strong baseline. |

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
| **Priority** | 1–5, only for `reprioritize` action — leave blank otherwise |

**📜 Event Log** below the action row shows what just happened — task completions, dev absences, bugs, reward amount.

---

## 🖥️ UI Components — Round 2 (Project Manager Tab)

### Top Controls Row

```
[🎯 Project Scenario ▾]  [🔄 Reset Project]  [🤖 Auto-Assign Sprint]  [⏩ Advance Day]
```

| Component | What it does |
|-----------|-------------|
| **🎯 Project Scenario** | `project_easy` (25 tasks / 6 sprints), `project_medium` (30 tasks), `project_hard` (40 tasks). |
| **🔄 Reset Project** | Starts a fresh 60-day project. Always click this first. |
| **🤖 Auto-Assign Sprint** | Assigns all backlog tasks for the *current sprint only* using skill-match heuristic. Does **not** advance the day — use ⏩ after to progress. |
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

These feed the `inst_score` metric. Ignoring them costs 30% of the final score.

**🔴 Tech Debt Tracker** — lists every task that was missed at a sprint boundary. Each missed task permanently reduces team productivity by 2%. Miss 5 tasks in sprint 1 and every developer is 10% slower for the rest of the project.

**📊 Project Metrics** — all key numbers: cumulative reward, team balance, instruction-following score (0–1), tech debt count, average sprint score, task counts.

### Cross-Sprint Reward Chart

Shows per-sprint score bars (✅/⚠️/❌ thresholds) plus a cumulative sparkline across all 60 steps.

### R2 Manual Action Row

Same fields as R1, plus two new additions:
- **`sprint_plan`** action: batch-plans multiple tasks for the sprint in one call
- **Task IDs (sprint_plan)**: comma-separated list, e.g. `T01,T02,T03`

---

## 🧪 Testing Checklist — Round 1 (Sprint Manager)

Follow these steps in order to confirm Round 1 is working correctly.

### Step 1 — Basic reset and board load

1. Open the **🏃 Round 1 — Sprint Manager** tab
2. Select `easy_sprint` from the scenario dropdown
3. Click **🔄 Reset Sprint**
4. ✅ **Expected:** Sprint Board populates with tasks in `BACKLOG`, Team Workload shows all devs as 🟢FREE, Metrics shows Day 1, reward 0.0

### Step 2 — Manual assign

1. Set Action = `assign`, Task ID = `T1`, Dev ID = `dev1`
2. Click **▶️ Take Action**
3. ✅ **Expected:** T1 moves to `IN PROGRESS` on the board, dev1 shows 🟡BUSY, Event Log shows the assignment and a positive reward (~+1.0 to +1.5), Reward History updates

### Step 3 — Auto-Assign All

1. Click **🔄 Reset Sprint** to start fresh
2. Click **🤖 Auto-Assign All**
3. ✅ **Expected:** All assignable backlog tasks move to `IN PROGRESS` in one click, Team Workload bars fill up, Sprint Metrics shows increased reward, Reward History shows multiple steps

### Step 4 — Skip to advance days

1. After Auto-Assign, click **▶️ Take Action** with Action = `skip` several times
2. ✅ **Expected:** Day counter increments each step, progress bars on IN PROGRESS tasks fill, tasks eventually move to `DONE` when complete

### Step 5 — Invalid action guard

1. Try to assign a task to the wrong skill dev (e.g. a backend task to a frontend-only dev)
2. ✅ **Expected:** Event Log shows a guard/invalid message, reward is 0 or negative, board does not change incorrectly

### Step 6 — All three scenarios

Repeat Steps 1–3 for `medium_sprint` and `hard_sprint`:
- `medium_sprint`: More tasks (8), random events may fire (dev sick, bug appears mid-sprint). Check that Event Log shows random event messages.
- `hard_sprint`: 12 tasks, urgent bugs. Some tasks will have very short deadlines — check that MISSED tasks appear in the board if you skip too many days.

---

## 🧪 Testing Checklist — Round 2 (Project Manager)

Follow these steps in order to confirm Round 2 is working correctly.

### Step 1 — Basic reset and timeline load

1. Open the **🚀 Round 2 — Project Manager** tab
2. Select `project_easy` from the scenario dropdown
3. Click **🔄 Reset Project**
4. ✅ **Expected:** Sprint Timeline shows Sprint 1 as 🏃 (current), sprints 2–6 as ⏳, Sprint Board shows Sprint 1 tasks in BACKLOG, Instruction Queue is empty (instructions release later), Tech Debt shows none

### Step 2 — Auto-Assign Sprint (Sprint 1)

1. Click **🤖 Auto-Assign Sprint**
2. ✅ **Expected:** All Sprint 1 backlog tasks move to IN PROGRESS, Team Workload bars fill, Project Metrics reward increases. The Sprint Timeline and Cross-Sprint Reward Chart also update.

### Step 3 — Advance days through Sprint 1

1. Click **⏩ Advance Day** repeatedly (or use Action = `skip` in the manual row) until Sprint 1 ends (Day 10)
2. ✅ **Expected:**
   - Tasks complete and move to DONE as the days pass
   - Instruction Queue starts populating as instructions are released on their scheduled days
   - At Day 10, Sprint Timeline updates Sprint 1 to ✅ / ⚠️ / ❌ depending on delivery rate
   - Sprint Board switches to Sprint 2 tasks
   - Cross-Sprint Reward Chart shows Sprint 1 bar

### Step 4 — Auto-Assign Sprint for all remaining sprints

Repeat for each sprint (2 through 6):
1. Click **🤖 Auto-Assign Sprint** at the start of each sprint
2. Click **⏩ Advance Day** through the 10 days of that sprint
3. ✅ **Expected per sprint:**
   - Sprint Timeline updates each sprint's status icon after it completes
   - If tasks were missed in a previous sprint, Tech Debt Tracker lists them and Team Workload shows `⚠️prod=X.XX` warnings
   - Instruction Queue items get ✅ when followed, ⚠️ when ignored
   - Cross-Sprint Reward Chart grows one bar per sprint

### Step 5 — Follow an instruction

1. When the Instruction Queue shows a ⚠️ item, read the target task
2. Manually assign or reprioritize that task as instructed
3. ✅ **Expected:** The ⚠️ in the Instruction Queue changes to ✅, Event Log shows instruction-following reward, Project Metrics `inst_score` increases

### Step 6 — Full project completion

After Day 60 (Sprint 6 ends):
1. ✅ **Expected:** All 6 sprints show a status icon in the Timeline, Project Metrics shows a final delivery score, overall project delivery bar reflects total completion

### Step 7 — All three scenarios

Repeat Steps 1–6 for `project_medium` and `project_hard`:
- `project_medium` (30 tasks): More complex dependency chains — some tasks will be BLOCKED until dependencies are done. Use `unblock` after resolving them.
- `project_hard` (40 tasks): Aggressive deadlines. Expect tech debt to accumulate. Watch the `⚠️prod` warnings grow and observe the productivity penalty in the Reward Chart.

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
# If they interfere: global env bug
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

# Step each session independently using their episode_ids
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{"episode_id":"<id_from_terminal_1>","action":{"action_type":"assign","task_id":"T1","dev_id":"dev1"}}'

curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{"episode_id":"<id_from_terminal_2>","action":{"action_type":"skip"}}'

# Expected: each returns state from its own episode, no cross-contamination
```

### Test 3: Concurrent UI users (load test)

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

## ✅ Quick Smoke Test — 60 Seconds

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

# 5. Validate
openenv validate
# Expected: [OK] ai-sprint-manager: Ready
```

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
| Per-session isolation | ✅ Each UI user has independent env |

---