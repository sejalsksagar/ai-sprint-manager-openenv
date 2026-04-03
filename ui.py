"""
Gradio UI for AI Sprint Manager OpenEnv
"""
import gradio as gr
import requests
import json

ENV_URL = "http://localhost:7860"

def reset_env(task_name):
    try:
        r = requests.post(f"{ENV_URL}/reset", json={"task_name": task_name, "seed": 42})
        obs = r.json()
        return format_sprint_board(obs), format_developers(obs), format_events(obs), obs
    except Exception as e:
        return str(e), "", "", {}

def take_action(action_type, task_id, dev_id, new_priority, current_obs):
    try:
        action = {
            "action_type": action_type,
            "task_id": task_id or None,
            "dev_id": dev_id or None,
            "new_priority": int(new_priority) if new_priority else None,
        }
        r = requests.post(f"{ENV_URL}/step", json={"action": action})
        result = r.json()
        obs = result["observation"]
        reward = result["reward"]
        done = result["done"]

        event_text = format_events(obs)
        if reward != 0:
            event_text += f"\n💰 Step Reward: {reward:+.2f}"
        if done:
            event_text += "\n\n🏁 SPRINT COMPLETE!"

        return (
            format_sprint_board(obs),
            format_developers(obs),
            event_text,
            format_metrics(obs),
            obs,
        )
    except Exception as e:
        return str(e), "", "", "", current_obs

def format_sprint_board(obs):
    if not obs or "tasks" not in obs:
        return "No sprint active. Click Reset to start."

    sections = {"backlog": [], "in_progress": [], "done": [], "missed": [], "blocked": []}
    for t in obs["tasks"]:
        s = t["status"]
        if s not in sections:
            s = "backlog"
        bar = "█" * int(t["progress"] * 10) + "░" * (10 - int(t["progress"] * 10))
        sections[s].append(
            f"  [{t['id']}] {t['name']}\n"
            f"       Type:{t['task_type']} | P{t['priority']} | "
            f"Effort:{t['effort']} | Due:Day{t['deadline']}\n"
            f"       Skill:{t['required_skill']} | Dev:{t['assigned_to'] or '-'}\n"
            f"       Progress: [{bar}] {t['progress']:.0%}"
        )

    day = obs.get("current_day", "?")
    sprint_len = obs.get("sprint_length", 10)
    day_bar = "▓" * day + "░" * (sprint_len - day)

    out = f"📅 Day {day}/{sprint_len}  [{day_bar}]\n"
    out += f"✅ Done:{obs['tasks_completed']}  ❌ Missed:{obs['tasks_missed']}  "
    out += f"🔄 In Progress:{obs['tasks_in_progress']}  📋 Backlog:{obs['tasks_backlog']}\n"
    out += "─" * 50 + "\n"

    labels = {
        "backlog": "📋 BACKLOG",
        "in_progress": "🔄 IN PROGRESS",
        "done": "✅ DONE",
        "missed": "❌ MISSED",
        "blocked": "🚫 BLOCKED",
    }
    for key, label in labels.items():
        items = sections[key]
        if items:
            out += f"\n{label} ({len(items)})\n"
            out += "\n".join(items) + "\n"

    return out

def format_developers(obs):
    if not obs or "developers" not in obs:
        return ""
    out = "👥 DEVELOPER WORKLOAD\n" + "─" * 40 + "\n"
    for d in obs["developers"]:
        load = d["current_load"]
        cap = d["capacity"]
        filled = min(int((load / cap) * 10) if cap > 0 else 0, 10)
        bar = "█" * filled + "░" * (10 - filled)
        status = "✅" if d["is_available"] else "🤒"
        out += (
            f"{status} {d['name']} ({d['skill']})\n"
            f"   Load: [{bar}] {load}/{cap} pts\n"
            f"   Tasks: {d['assigned_tasks'] or 'none'}\n\n"
        )
    return out

def format_events(obs):
    events = obs.get("events", [])
    if not events:
        return "No events yet."
    return "\n".join(f"• {e}" for e in events)

def format_metrics(obs):
    if not obs:
        return ""
    return (
        f"📊 Cumulative Reward: {obs.get('cumulative_reward', 0):.2f}\n"
        f"⚖️  Workload Balance: {obs.get('workload_balance_score', 0):.2f}/1.0\n"
        f"✅ Completed: {obs.get('tasks_completed', 0)}\n"
        f"❌ Missed: {obs.get('tasks_missed', 0)}"
    )

# ── Build UI ──────────────────────────────────────────────────────────────────

with gr.Blocks(
    title="🤖 AI Sprint Manager",
    theme=gr.themes.Soft(),
    css=".gradio-container { max-width: 1200px; margin: auto; }"
) as demo:

    current_obs = gr.State({})

    gr.Markdown("""
    # 🤖 AI Sprint Manager — OpenEnv
    **An RL environment where an AI agent manages agile sprints.**
    Assign tasks to developers, balance workload, and hit deadlines!
    """)

    with gr.Row():
        task_selector = gr.Dropdown(
            choices=["easy_sprint", "medium_sprint", "hard_sprint"],
            value="easy_sprint",
            label="Select Sprint Scenario",
        )
        reset_btn = gr.Button("🔄 Reset Sprint", variant="primary")

    with gr.Row():
        with gr.Column(scale=2):
            sprint_board = gr.Textbox(
                label="📋 Sprint Board",
                lines=25,
                interactive=False,
            )
        with gr.Column(scale=1):
            dev_panel = gr.Textbox(
                label="👥 Developers",
                lines=12,
                interactive=False,
            )
            metrics_panel = gr.Textbox(
                label="📊 Metrics",
                lines=6,
                interactive=False,
            )

    gr.Markdown("### 🎮 Take Action")
    with gr.Row():
        action_type = gr.Dropdown(
            choices=["assign", "reassign", "reprioritize", "unblock", "skip"],
            value="assign",
            label="Action Type",
        )
        task_id_input = gr.Textbox(label="Task ID (e.g. T1)", placeholder="T1")
        dev_id_input = gr.Textbox(label="Developer ID (e.g. dev1)", placeholder="dev1")
        priority_input = gr.Dropdown(
            choices=["", "1", "2", "3", "4", "5"],
            value="",
            label="New Priority (reprioritize only)",
        )

    step_btn = gr.Button("▶️ Take Action", variant="primary")

    event_log = gr.Textbox(
        label="📜 Event Log",
        lines=5,
        interactive=False,
    )

    gr.Markdown("""
    ---
    ### 📖 Quick Guide
    | Action | What it does |
    |--------|-------------|
    | `assign` | Put a backlog task onto a developer |
    | `reassign` | Move a task to a different developer |
    | `reprioritize` | Change a task's priority (1=highest) |
    | `unblock` | Unblock a stuck task |
    | `skip` | Let the sprint advance without acting |

    **Tips:** Match developer skills to task requirements for bonus reward!
    Backend tasks → Alice (dev1) | Frontend → Bob (dev2) | Any → Carol (dev3)
    """)

    reset_btn.click(
        fn=reset_env,
        inputs=[task_selector],
        outputs=[sprint_board, dev_panel, event_log, current_obs],
    )

    step_btn.click(
        fn=take_action,
        inputs=[action_type, task_id_input, dev_id_input, priority_input, current_obs],
        outputs=[sprint_board, dev_panel, event_log, metrics_panel, current_obs],
    )

# ── Mount FastAPI routes into Gradio (single port 7860) ───────────────────────
from server import app as fastapi_app

for route in fastapi_app.routes:
    demo.app.router.routes.append(route)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)