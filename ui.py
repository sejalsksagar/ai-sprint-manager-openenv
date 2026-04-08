"""
AI Sprint Manager — Gradio UI + FastAPI (optimized for speed)
- In-memory data cache (no repeated disk reads)
- Lightweight ASCII charts instead of Plotly
- Minimal recomputation per step
"""
import gradio as gr
import json
from fastapi import FastAPI
import uvicorn

from sprint_env.environment import SprintManagerEnv
from sprint_env.models import SprintAction
from sprint_env.data_loader import load_sprint_data, get_scenario_names

# ── Boot-time init (runs once) ────────────────────────────────────────────────
env = SprintManagerEnv()
SCENARIO_NAMES = get_scenario_names()   # cached after first call
_sprint_data = load_sprint_data()       # warms the cache immediately

# ── FastAPI ───────────────────────────────────────────────────────────────────
api = FastAPI(title="AI Sprint Manager — OpenEnv", version="1.0.0")

@api.post("/reset")
def api_reset(req: dict = {}):
    obs = env.reset(
        task_name=req.get("task_name", "easy_sprint"),
        seed=req.get("seed"),
        episode_id=req.get("episode_id"),
    )
    return obs.model_dump()

@api.post("/step")
def api_step(req: dict):
    action = SprintAction(**req.get("action", {}))
    obs, reward, done, info = env.step(action)
    return {"observation": obs.model_dump(), "reward": reward, "done": done, "info": info}

@api.get("/state")
def api_state():
    return env.state.model_dump()

@api.get("/health")
def api_health():
    return {"status": "ok", "env": "ai-sprint-manager"}

@api.get("/tasks")
def api_tasks():
    return {"tasks": [
        {"id": k, "description": v.get("description",""), "difficulty": v.get("difficulty","")}
        for k, v in _sprint_data["scenarios"].items()
    ]}

# ── In-memory state ───────────────────────────────────────────────────────────
reward_history: list[dict] = []

# ── Lightweight chart builders (pure text — instant render) ───────────────────

def make_reward_chart() -> str:
    """ASCII sparkline reward chart — renders instantly."""
    if len(reward_history) < 2:
        return "📈 Reward chart will appear after first action."

    cumulative = [r["cumulative"] for r in reward_history]
    step_rewards = [r["reward"] for r in reward_history]

    # Sparkline using block characters
    def sparkline(values: list[float]) -> str:
        if not values:
            return ""
        blocks = "▁▂▃▄▅▆▇█"
        mn, mx = min(values), max(values)
        span = mx - mn or 1
        return "".join(blocks[int((v - mn) / span * (len(blocks) - 1))] for v in values)

    cum_spark = sparkline(cumulative)
    step_spark = sparkline(step_rewards)

    lines = [
        f"📈 REWARD HISTORY (Step 0 → {len(reward_history)-1})",
        "─" * 45,
        f"Cumulative : {cum_spark}",
        f"             min={min(cumulative):+.2f}  max={max(cumulative):+.2f}  "
        f"current={cumulative[-1]:+.2f}",
        "",
        f"Per Step   : {step_spark}",
        f"             min={min(step_rewards):+.2f}  max={max(step_rewards):+.2f}  "
        f"last={step_rewards[-1]:+.2f}",
        "",
    ]

    # Mini bar chart for last 10 steps
    recent = step_rewards[-10:]
    lines.append("Last 10 steps:")
    for i, r in enumerate(recent):
        bar_len = int(abs(r) * 8)
        bar = ("+" if r >= 0 else "-") * min(bar_len, 20)
        lines.append(f"  s{len(step_rewards)-len(recent)+i+1:02d}: {bar} {r:+.2f}")

    return "\n".join(lines)


def make_task_chart(obs: dict) -> str:
    """ASCII task status chart — renders instantly."""
    if not obs or "tasks" not in obs:
        return "📊 Task chart will appear after reset."

    counts = {"done": 0, "in_progress": 0, "backlog": 0, "missed": 0, "blocked": 0}
    total = len(obs["tasks"])
    for t in obs["tasks"]:
        s = t["status"]
        if s in counts:
            counts[s] += 1

    config = [
        ("done",        "✅ Done       ", "#"),
        ("in_progress", "🔄 In Progress", "="),
        ("backlog",     "📋 Backlog    ", "·"),
        ("missed",      "❌ Missed     ", "!"),
        ("blocked",     "🚫 Blocked    ", "?"),
    ]

    lines = [f"📊 TASK STATUS ({total} total)", "─" * 40]
    for key, label, char in config:
        count = counts[key]
        if total > 0:
            bar_len = int((count / total) * 24)
            pct = count / total * 100
        else:
            bar_len, pct = 0, 0
        bar = char * bar_len + "·" * (24 - bar_len)
        lines.append(f"{label}: [{bar}] {count} ({pct:.0f}%)")

    lines.append("")
    lines.append(f"Sprint completion: {counts['done']}/{total} tasks done")
    if total > 0:
        comp_pct = int(counts["done"] / total * 20)
        lines.append(f"[{'█' * comp_pct}{'░' * (20 - comp_pct)}] {counts['done']/total*100:.0f}%")

    return "\n".join(lines)


# ── Formatters (optimized — no repeated dict lookups) ─────────────────────────

_TYPE_EMOJI  = {"feature": "🔧", "bug": "🐛", "urgent_bug": "🚨", "tech_debt": "🔩"}
_PRIO_LABEL  = ["", "🔴P1", "🟠P2", "🟡P3", "🟢P4", "⚪P5"]
_SKILL_EMOJI = {"backend": "⚙️", "frontend": "🎨", "devops": "🚀", "fullstack": "💎"}


def format_sprint_board(obs: dict) -> str:
    if not obs or "tasks" not in obs:
        return "👆 Select a scenario and click Reset Sprint to begin!"

    sections: dict[str, list[str]] = {
        "in_progress": [], "backlog": [], "done": [], "missed": [], "blocked": []
    }
    for t in obs["tasks"]:
        s = t["status"]
        if s not in sections:
            s = "backlog"
        filled = int(t["progress"] * 10)
        bar = "█" * filled + "░" * (10 - filled)
        te  = _TYPE_EMOJI.get(t["task_type"], "📌")
        pl  = _PRIO_LABEL[t["priority"]] if t["priority"] <= 5 else ""
        sections[s].append(
            f"  {te} [{t['id']}] {t['name']}\n"
            f"     {pl} | Effort:{t['effort']}sp | Due:Day{t['deadline']} | {t['required_skill']}\n"
            f"     Dev:{t['assigned_to'] or '—'} | [{bar}] {t['progress']:.0%}"
        )

    day      = int(obs.get("current_day", 1))
    slen     = int(obs.get("sprint_length", 10))
    day_bar  = "▓" * day + "░" * (slen - day)

    lines = [
        f"📅 Day {day}/{slen}  [{day_bar}]",
        f"✅{obs['tasks_completed']} 🔄{obs['tasks_in_progress']} "
        f"📋{obs['tasks_backlog']} ❌{obs['tasks_missed']}",
        "─" * 50,
    ]
    for key, label in [
        ("in_progress","🔄 IN PROGRESS"), ("backlog","📋 BACKLOG"),
        ("done","✅ DONE"), ("missed","❌ MISSED"), ("blocked","🚫 BLOCKED")
    ]:
        items = sections[key]
        if items:
            lines.append(f"\n{label} ({len(items)})")
            lines.extend(items)

    return "\n".join(lines)


def format_developers(obs: dict) -> str:
    if not obs or "developers" not in obs:
        return ""
    lines = ["👥 TEAM WORKLOAD", "─" * 38, ""]
    for d in obs["developers"]:
        load, cap = d["current_load"], d["capacity"]
        pct    = load / cap if cap > 0 else 0
        filled = min(int(pct * 10), 10)
        bar    = "█" * filled + "░" * (10 - filled)
        status = "✅" if d["is_available"] else "🤒"
        load_s = "🔴FULL" if pct >= 1.0 else ("🟡BUSY" if pct >= 0.6 else "🟢FREE")
        se     = _SKILL_EMOJI.get(d["skill"], "👤")
        tasks  = ", ".join(d["assigned_tasks"]) if d["assigned_tasks"] else "—"
        lines += [
            f"{status} {d['name']} {se} ({d['skill']})",
            f"  [{bar}] {load}/{cap}sp {load_s}",
            f"  Tasks: {tasks}",
            "",
        ]
    return "\n".join(lines)


def format_skill_table(obs: dict) -> str:
    if not obs or "developers" not in obs:
        return ""
    lines = ["🎯 SKILL → DEV GUIDE", "─" * 38, ""]
    skill_groups: dict[str, list[str]] = {}
    for d in obs["developers"]:
        s = d["skill"]
        avail = "✅" if d["is_available"] and d["current_load"] < d["capacity"] else "❌"
        skill_groups.setdefault(s, []).append(
            f"  {avail} {d['name']} ({d['id']}) {d['current_load']}/{d['capacity']}sp"
        )
    for skill, devs in skill_groups.items():
        lines.append(f"{_SKILL_EMOJI.get(skill,'👤')} {skill.upper()} tasks:")
        lines.extend(devs)
        lines.append("")
    lines += ["💎 fullstack can take ANY task", "❌ = unavailable or full"]
    return "\n".join(lines)


def format_events(obs: dict) -> str:
    events = obs.get("events", [])
    return "\n".join(f"• {e}" for e in events) if events else "No events yet."


def format_metrics(obs: dict) -> str:
    if not obs:
        return ""
    bal   = obs.get("workload_balance_score", 0)
    filled = int(bal * 10)
    bar    = "█" * filled + "░" * (10 - filled)
    return (
        f"📊 Cumulative Reward : {obs.get('cumulative_reward', 0):+.2f}\n"
        f"⚖️  Balance           : [{bar}] {bal:.2f}\n"
        f"✅ Done              : {obs.get('tasks_completed', 0)}\n"
        f"❌ Missed            : {obs.get('tasks_missed', 0)}\n"
        f"🔄 In Progress       : {obs.get('tasks_in_progress', 0)}\n"
        f"📋 Backlog           : {obs.get('tasks_backlog', 0)}"
    )


# ── All outputs in one tuple (keeps wiring DRY) ───────────────────────────────
def _make_outputs(obs_dict: dict, event_text: str):
    return (
        format_sprint_board(obs_dict),
        format_developers(obs_dict),
        format_skill_table(obs_dict),
        event_text,
        format_metrics(obs_dict),
        make_reward_chart(),
        make_task_chart(obs_dict),
        obs_dict,
    )


# ── Gradio handler functions ──────────────────────────────────────────────────

def reset_env(task_name: str):
    global reward_history
    reward_history = []
    obs     = env.reset(task_name=task_name, seed=42)
    obs_dict = obs.model_dump()
    reward_history.append({"step": 0, "reward": 0.0, "cumulative": 0.0})
    return _make_outputs(obs_dict, "• Sprint started! Assign tasks to begin.")


def take_action(action_type, task_id, dev_id, new_priority, current_obs):
    try:
        action = SprintAction(
            action_type=action_type,
            task_id=task_id or None,
            dev_id=dev_id or None,
            new_priority=int(new_priority) if new_priority else None,
        )
        obs, reward, done, info = env.step(action)
        obs_dict = obs.model_dump()
        reward_history.append({
            "step": len(reward_history),
            "reward": reward,
            "cumulative": obs_dict["cumulative_reward"],
        })
        ev = format_events(obs_dict)
        if reward > 0:
            ev += f"\n💰 Reward: +{reward:.2f}"
        elif reward < 0:
            ev += f"\n💸 Reward: {reward:.2f}"
        if done:
            ev += f"\n\n🏁 SPRINT COMPLETE! Score: {info.get('final_score',0):.2f}/1.0"
        return _make_outputs(obs_dict, ev)
    except Exception as e:
        return _make_outputs(current_obs, f"❌ Error: {e}")


def auto_assign(current_obs: dict):
    if not current_obs or "tasks" not in current_obs:
        return _make_outputs({}, "⚠️ Reset the sprint first!")

    tasks    = current_obs.get("tasks", [])
    devs     = current_obs.get("developers", [])
    backlog  = sorted(
        [t for t in tasks if t["status"] == "backlog"],
        key=lambda t: (t["priority"], t["deadline"])
    )

    if not backlog:
        return _make_outputs(current_obs, "✅ No backlog tasks to assign!")

    obs_dict = current_obs
    events_log = []

    for task in backlog:
        available = [d for d in devs if d["is_available"] and d["current_load"] < d["capacity"]]
        skill_match = [d for d in available if d["skill"] == task["required_skill"] or d["skill"] == "fullstack"]
        chosen = skill_match[0] if skill_match else (available[0] if available else None)

        if chosen:
            action = SprintAction(action_type="assign", task_id=task["id"], dev_id=chosen["id"])
            obs, reward, done, info = env.step(action)
            obs_dict = obs.model_dump()
            devs = obs_dict["developers"]
            reward_history.append({
                "step": len(reward_history),
                "reward": reward,
                "cumulative": obs_dict["cumulative_reward"],
            })
            events_log.append(f"✅ {task['id']} → {chosen['name']} (reward {reward:+.2f})")
        else:
            events_log.append(f"⚠️ No available dev for {task['id']}")

    return _make_outputs(obs_dict, "\n".join(events_log))


# ── Build UI ──────────────────────────────────────────────────────────────────
CSS = """
.gradio-container { max-width: 1400px; margin: auto; }
footer { display: none !important; }
"""

with gr.Blocks(title="🤖 AI Sprint Manager") as demo:

    current_obs = gr.State({})

    gr.Markdown("""
    # 🤖 AI Sprint Manager — OpenEnv
    **RL environment for agile sprint management.**
    Assign tasks, balance workload, beat deadlines — watch the AI learn!
    """)

    with gr.Row():
        task_selector = gr.Dropdown(
            choices=SCENARIO_NAMES, value=SCENARIO_NAMES[0],
            label="🎯 Sprint Scenario", scale=2
        )
        reset_btn = gr.Button("🔄 Reset Sprint",    variant="primary",   scale=1)
        auto_btn  = gr.Button("🤖 Auto-Assign All", variant="secondary",  scale=1)

    with gr.Row():
        with gr.Column(scale=3):
            sprint_board = gr.Textbox(
                label="📋 Sprint Board", lines=26, interactive=False,
                value="👆 Select a scenario and click Reset Sprint to begin!"
            )
        with gr.Column(scale=2):
            dev_panel    = gr.Textbox(label="👥 Team Workload",        lines=9,  interactive=False)
            skill_table  = gr.Textbox(label="🎯 Skill → Dev Guide",    lines=9,  interactive=False)
            metrics_panel= gr.Textbox(label="📊 Sprint Metrics",       lines=8,  interactive=False)

    with gr.Row():
        reward_chart_box = gr.Textbox(label="📈 Reward History",       lines=14, interactive=False)
        task_chart_box   = gr.Textbox(label="📊 Task Status",          lines=14, interactive=False)

    gr.Markdown("### 🎮 Manual Action")
    with gr.Row():
        action_type    = gr.Dropdown(
            choices=["assign","reassign","reprioritize","unblock","skip"],
            value="assign", label="Action", scale=1
        )
        task_id_input  = gr.Textbox(label="Task ID",   placeholder="e.g. T1",   scale=1)
        dev_id_input   = gr.Textbox(label="Dev ID",    placeholder="e.g. dev1", scale=1)
        priority_input = gr.Dropdown(
            choices=["","1","2","3","4","5"], value="",
            label="Priority (reprioritize only)", scale=1
        )
        step_btn = gr.Button("▶️ Take Action", variant="primary", scale=1)

    event_log = gr.Textbox(label="📜 Event Log", lines=4, interactive=False)

    gr.Markdown("""
    ---
    | Action | When | Example |
    |--------|------|---------|
    | `assign` | Put backlog task on a dev | Task=T1, Dev=dev1 |
    | `reassign` | Move in-progress task | Task=T2, Dev=dev3 |
    | `reprioritize` | Change priority | Task=T4, Priority=1 |
    | `skip` | Advance 1 day | — |

    **Skills:** ⚙️ backend → Alice/Eve | 🎨 frontend → Bob | 🚀 devops → Carol | 💎 fullstack → Dave (any task)
    """)

    # ── Wire outputs ──────────────────────────────────────────────────────────
    OUTPUTS = [sprint_board, dev_panel, skill_table, event_log,
               metrics_panel, reward_chart_box, task_chart_box, current_obs]

    reset_btn.click(fn=reset_env,   inputs=[task_selector], outputs=OUTPUTS)
    auto_btn.click( fn=auto_assign, inputs=[current_obs],   outputs=OUTPUTS)
    step_btn.click(
        fn=take_action,
        inputs=[action_type, task_id_input, dev_id_input, priority_input, current_obs],
        outputs=OUTPUTS,
    )

# ── Mount into FastAPI ────────────────────────────────────────────────────────
app = gr.mount_gradio_app(api, demo, path="/")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)