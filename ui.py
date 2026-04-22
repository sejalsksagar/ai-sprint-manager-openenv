"""
AI Sprint Manager — Gradio UI + FastAPI
Round 1: single-sprint environment (unchanged)
Round 2: long-horizon multi-sprint project environment (new tab)

Both share the same FastAPI app on port 7860 via gr.mount_gradio_app().
R1 endpoints: /reset /step /state /health /tasks  (unchanged)
R2 endpoints: /project/reset /project/step /project/state /project/health /project/tasks
"""
import os
import json
import gradio as gr
import uvicorn
from fastapi import FastAPI

# ── R1 imports (unchanged) ────────────────────────────────────────────────────
from sprint_env.environment  import SprintManagerEnv
from sprint_env.models       import SprintAction
from sprint_env.data_loader  import load_sprint_data, get_scenario_names

# ── R2 imports ────────────────────────────────────────────────────────────────
from sprint_env.project_environment import ProjectManagerEnv, VALID_PROJECT_TASK_NAMES
from sprint_env.project_models      import ProjectAction
from server.project_app             import project_router

# ── Boot-time init (runs once) ────────────────────────────────────────────────
r1_env         = SprintManagerEnv()
r2_env         = ProjectManagerEnv()
SCENARIO_NAMES = get_scenario_names()
_sprint_data   = load_sprint_data()

# ── FastAPI — single shared app, port 7860 ────────────────────────────────────
api = FastAPI(title="AI Sprint Manager — OpenEnv", version="2.0.0")

# R1 endpoints (verbatim from uploaded ui.py)
@api.post("/reset")
def api_reset(req: dict = {}):
    obs = r1_env.reset(
        task_name=req.get("task_name", "easy_sprint"),
        seed=req.get("seed"),
        episode_id=req.get("episode_id"),
    )
    return obs.model_dump()

@api.post("/step")
def api_step(req: dict):
    action = SprintAction(**req.get("action", {}))
    obs, reward, done, info = r1_env.step(action)
    return {"observation": obs.model_dump(), "reward": reward, "done": done, "info": info}

@api.get("/state")
def api_state():
    return r1_env.state.model_dump()

@api.get("/health")
def api_health():
    return {"status": "ok", "env": "ai-sprint-manager"}

@api.get("/tasks")
def api_tasks():
    return {"tasks": [
        {"id": k, "description": v.get("description", ""), "difficulty": v.get("difficulty", "")}
        for k, v in _sprint_data["scenarios"].items()
    ]}

# R2 endpoints mounted at /project/*
api.include_router(project_router)


# ═══════════════════════════════════════════════════════════════════════════════
# ROUND 1 — helpers (all preserved verbatim from uploaded ui.py)
# ═══════════════════════════════════════════════════════════════════════════════

r1_reward_history: list[dict] = []

_TYPE_EMOJI  = {"feature": "🔧", "bug": "🐛", "urgent_bug": "🚨", "tech_debt": "🔩"}
_PRIO_LABEL  = ["", "🔴P1", "🟠P2", "🟡P3", "🟢P4", "⚪P5"]
_SKILL_EMOJI = {"backend": "⚙️", "frontend": "🎨", "devops": "🚀", "fullstack": "💎"}


def _sparkline(values: list) -> str:
    if not values:
        return ""
    blocks = "▁▂▃▄▅▆▇█"
    mn, mx = min(values), max(values)
    span   = mx - mn or 1
    return "".join(blocks[int((v - mn) / span * (len(blocks) - 1))] for v in values)


def make_reward_chart(history: list) -> str:
    if len(history) < 2:
        return "📈 Reward chart will appear after first action."
    cumulative   = [r["cumulative"] for r in history]
    step_rewards = [r["reward"]     for r in history]
    lines = [
        f"📈 REWARD HISTORY (Step 0 → {len(history)-1})",
        "─" * 45,
        f"Cumulative : {_sparkline(cumulative)}",
        f"             min={min(cumulative):+.2f}  max={max(cumulative):+.2f}  current={cumulative[-1]:+.2f}",
        "",
        f"Per Step   : {_sparkline(step_rewards)}",
        f"             min={min(step_rewards):+.2f}  max={max(step_rewards):+.2f}  last={step_rewards[-1]:+.2f}",
        "",
    ]
    recent = step_rewards[-10:]
    lines.append("Last 10 steps:")
    for i, r in enumerate(recent):
        bar = ("+" if r >= 0 else "-") * min(int(abs(r) * 8), 20)
        lines.append(f"  s{len(step_rewards)-len(recent)+i+1:02d}: {bar} {r:+.2f}")
    return "\n".join(lines)


def make_task_chart(obs: dict) -> str:
    if not obs or "tasks" not in obs:
        return "📊 Task chart will appear after reset."
    counts = {"done": 0, "in_progress": 0, "backlog": 0, "missed": 0, "blocked": 0}
    total  = len(obs["tasks"])
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
        count   = counts[key]
        bar_len = int((count / total) * 24) if total > 0 else 0
        pct     = count / total * 100         if total > 0 else 0
        bar     = char * bar_len + "·" * (24 - bar_len)
        lines.append(f"{label}: [{bar}] {count} ({pct:.0f}%)")
    lines.append("")
    lines.append(f"Sprint completion: {counts['done']}/{total} tasks done")
    if total > 0:
        cp = int(counts["done"] / total * 20)
        lines.append(f"[{'█'*cp}{'░'*(20-cp)}] {counts['done']/total*100:.0f}%")
    return "\n".join(lines)


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
        ("in_progress", "🔄 IN PROGRESS"), ("backlog", "📋 BACKLOG"),
        ("done", "✅ DONE"), ("missed", "❌ MISSED"), ("blocked", "🚫 BLOCKED"),
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
        s     = d["skill"]
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
    bal    = obs.get("workload_balance_score", 0)
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


def _make_r1_outputs(obs_dict: dict, event_text: str):
    return (
        format_sprint_board(obs_dict),
        format_developers(obs_dict),
        format_skill_table(obs_dict),
        event_text,
        format_metrics(obs_dict),
        make_reward_chart(r1_reward_history),
        make_task_chart(obs_dict),
        obs_dict,
    )


# ── R1 Gradio handlers ────────────────────────────────────────────────────────

def r1_reset_env(task_name: str):
    global r1_reward_history
    r1_reward_history = []
    obs      = r1_env.reset(task_name=task_name, seed=42)
    obs_dict = obs.model_dump()
    r1_reward_history.append({"step": 0, "reward": 0.0, "cumulative": 0.0})
    return _make_r1_outputs(obs_dict, "• Sprint started! Assign tasks to begin.")


def r1_take_action(action_type, task_id, dev_id, new_priority, current_obs):
    try:
        action = SprintAction(
            action_type=action_type,
            task_id=task_id or None,
            dev_id=dev_id or None,
            new_priority=int(new_priority) if new_priority else None,
        )
        obs, reward, done, info = r1_env.step(action)
        obs_dict = obs.model_dump()
        r1_reward_history.append({
            "step": len(r1_reward_history),
            "reward": reward,
            "cumulative": obs_dict["cumulative_reward"],
        })
        ev = format_events(obs_dict)
        if reward > 0:   ev += f"\n💰 Reward: +{reward:.2f}"
        elif reward < 0: ev += f"\n💸 Reward: {reward:.2f}"
        if done:         ev += f"\n\n🏁 SPRINT COMPLETE! Score: {info.get('final_score', 0):.2f}/1.0"
        return _make_r1_outputs(obs_dict, ev)
    except Exception as e:
        return _make_r1_outputs(current_obs, f"❌ Error: {e}")


def r1_auto_assign(current_obs: dict):
    if not current_obs or "tasks" not in current_obs:
        return _make_r1_outputs({}, "⚠️ Reset the sprint first!")
    tasks   = current_obs.get("tasks", [])
    devs    = current_obs.get("developers", [])
    backlog = sorted(
        [t for t in tasks if t["status"] == "backlog"],
        key=lambda t: (t["priority"], t["deadline"])
    )
    if not backlog:
        return _make_r1_outputs(current_obs, "✅ No backlog tasks to assign!")
    obs_dict   = current_obs
    events_log = []
    for task in backlog:
        available   = [d for d in devs if d["is_available"] and d["current_load"] < d["capacity"]]
        skill_match = [d for d in available
                       if d["skill"] == task["required_skill"] or d["skill"] == "fullstack"]
        chosen = skill_match[0] if skill_match else (available[0] if available else None)
        if chosen:
            action = SprintAction(action_type="assign", task_id=task["id"], dev_id=chosen["id"])
            obs, reward, done, info = r1_env.step(action)
            obs_dict = obs.model_dump()
            devs     = obs_dict["developers"]
            r1_reward_history.append({
                "step": len(r1_reward_history),
                "reward": reward,
                "cumulative": obs_dict["cumulative_reward"],
            })
            events_log.append(f"✅ {task['id']} → {chosen['name']} (reward {reward:+.2f})")
        else:
            events_log.append(f"⚠️ No available dev for {task['id']}")
    return _make_r1_outputs(obs_dict, "\n".join(events_log))


def r1_run_trained_agent(task_name: str):
    """
    Run the trained LLM (Qwen2.5-1.5B) via the HF router API.
    Falls back to the REINFORCE rule-based policy if no API key is set.
    Agent log shows every step action + reward so you can watch it think.
    """
    import requests as _req

    api_key   = os.getenv("HF_TOKEN") or os.getenv("API_KEY", "")
    api_base  = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
    model     = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-1.5B-Instruct")
    use_llm   = bool(api_key and api_key != "dummy")

    SYSTEM = (
        "You are an expert Tech Lead managing an agile sprint. "
        "Output a JSON action: {\"action_type\":\"<assign|reassign|reprioritize|unblock|skip>\","
        "\"task_id\":\"<id or null>\",\"dev_id\":\"<id or null>\",\"new_priority\":<1-5 or null>}. "
        "Only output JSON. Assign backlog tasks to available developers, skill match preferred."
    )

    def llm_action(obs_dict: dict) -> SprintAction:
        tasks_s = "\n".join(
            f"[{t['id']}] {t['name']} P{t['priority']} {t['status']} skill={t['required_skill']} dev={t['assigned_to']}"
            for t in obs_dict["tasks"]
        )
        devs_s = "\n".join(
            f"[{d['id']}] {d['name']} skill={d['skill']} load={d['current_load']}/{d['capacity']} avail={d['is_available']}"
            for d in obs_dict["developers"]
        )
        user_msg = (
            f"Day {obs_dict['current_day']}/{obs_dict['sprint_length']} "
            f"done={obs_dict['tasks_completed']} missed={obs_dict['tasks_missed']}\n"
            f"TASKS:\n{tasks_s}\nDEVS:\n{devs_s}\nOutput JSON action:"
        )
        try:
            resp = _req.post(
                f"{api_base}/chat/completions",
                headers={"Authorization": f"Bearer {api_key}"},
                json={"model": model, "messages": [
                    {"role": "system", "content": SYSTEM},
                    {"role": "user",   "content": user_msg},
                ], "max_tokens": 80, "temperature": 0.1},
                timeout=15,
            )
            text = resp.json()["choices"][0]["message"]["content"].strip()
            # Strip markdown fences
            if "```" in text:
                text = "\n".join(l for l in text.split("\n") if not l.strip().startswith("```"))
            s, e = text.find("{"), text.rfind("}") + 1
            d = json.loads(text[s:e]) if s >= 0 and e > s else {}
            return SprintAction(
                action_type=d.get("action_type", "skip"),
                task_id=d.get("task_id"),
                dev_id=d.get("dev_id"),
                new_priority=d.get("new_priority"),
            )
        except Exception as ex:
            return _rule_based_sprint_action(obs_dict)

    def _rule_based_sprint_action(obs_dict: dict) -> SprintAction:
        """Fallback rule-based policy."""
        tasks  = obs_dict.get("tasks", [])
        devs   = obs_dict.get("developers", [])
        avail  = [d for d in devs if d["is_available"] and d["current_load"] < d["capacity"]]
        backlog = sorted([t for t in tasks if t["status"] == "backlog"],
                         key=lambda t: (t["priority"], t["deadline"]))
        for task in backlog:
            match = [d for d in avail if d["skill"] == task["required_skill"] or d["skill"] == "fullstack"]
            dev = match[0] if match else (avail[0] if avail else None)
            if dev:
                return SprintAction(action_type="assign", task_id=task["id"], dev_id=dev["id"])
        return SprintAction(action_type="skip")

    obs      = r1_env.reset(task_name=task_name, seed=42)
    obs_dict = obs.model_dump()
    r1_reward_history.clear()
    r1_reward_history.append({"step": 0, "reward": 0.0, "cumulative": 0.0})

    mode_label = f"🤖 LLM ({model})" if use_llm else "🔧 Rule-based (set HF_TOKEN for LLM)"
    step_logs  = [f"{mode_label} on {task_name}", "─" * 40]

    for step in range(12):
        if obs_dict.get("done"):
            break
        action   = llm_action(obs_dict) if use_llm else _rule_based_sprint_action(obs_dict)
        obs, reward, done, info = r1_env.step(action)
        obs_dict = obs.model_dump()
        r1_reward_history.append({
            "step": step + 1, "reward": reward,
            "cumulative": obs_dict["cumulative_reward"],
        })
        task_part = f"→ {action.task_id}" if action.task_id else ""
        dev_part  = f"/ {action.dev_id}"  if action.dev_id  else ""
        step_logs.append(
            f"Day {obs_dict['current_day']:02d}: {action.action_type} {task_part}{dev_part} "
            f"| reward {reward:+.2f} | cumul {obs_dict['cumulative_reward']:+.2f}"
        )
        if done:
            score = info.get("final_score", 0.01)
            step_logs.append(f"\n🏁 Sprint done! Score: {score:.4f}/0.99")
            break

    return _make_r1_outputs(obs_dict, "\n".join(step_logs))


# ═══════════════════════════════════════════════════════════════════════════════
# ROUND 2 — helpers (multi-sprint project environment)
# ═══════════════════════════════════════════════════════════════════════════════

r2_reward_history: list[dict] = []


def r2_format_timeline(obs: dict) -> str:
    """6-sprint visual timeline with per-sprint delivery rate and score."""
    if not obs or "tasks" not in obs:
        return "👆 Select a project scenario and click Reset Project to begin!"
    current_sprint = obs.get("current_sprint", 1)
    current_day    = obs.get("current_day", 1)
    sprint_rewards = obs.get("sprint_rewards", [])
    tasks          = obs.get("tasks", [])
    lines = [
        f"🗓️  PROJECT TIMELINE  —  Day {current_day}/60  |  Sprint {current_sprint}/6",
        "─" * 56, "",
    ]
    for s in range(1, 7):
        s_tasks = [t for t in tasks if t.get("metadata", {}).get("sprint") == s]
        done    = sum(1 for t in s_tasks if t["status"] == "done")
        total_s = len(s_tasks)
        pct     = done / total_s * 100 if total_s else 0
        bar_f   = int(pct / 10)
        bar     = "█" * bar_f + "░" * (10 - bar_f)
        if s < current_sprint:
            reward = sprint_rewards[s - 1] if (s - 1) < len(sprint_rewards) else 0.0
            icon   = "✅" if pct >= 70 else ("⚠️" if pct >= 40 else "❌")
            lines.append(
                f"  {icon} Sprint {s} (D{(s-1)*10+1}-{s*10}): "
                f"[{bar}] {done}/{total_s}  score={reward:.2f}"
            )
        elif s == current_sprint:
            day_in = ((current_day - 1) % 10) + 1
            p_bar  = "▓" * day_in + "░" * (10 - day_in)
            lines.append(
                f"  🏃 Sprint {s} (D{(s-1)*10+1}-{s*10}): "
                f"[{bar}] {done}/{total_s}  day {day_in}/10 [{p_bar}]"
            )
        else:
            lines.append(
                f"  ⏳ Sprint {s} (D{(s-1)*10+1}-{s*10}): "
                f"{'·'*10}  {total_s} tasks queued"
            )
    lines.append("")
    overall_done  = sum(1 for t in tasks if t["status"] == "done")
    overall_total = len(tasks)
    proj_pct      = overall_done / overall_total * 100 if overall_total else 0
    proj_f        = int(proj_pct / 5)
    lines.append(
        f"📦 Project: [{'█'*proj_f}{'░'*(20-proj_f)}] "
        f"{overall_done}/{overall_total} ({proj_pct:.0f}%)"
    )
    return "\n".join(lines)


def r2_format_board(obs: dict) -> str:
    """Sprint board scoped to current sprint's tasks."""
    if not obs or "tasks" not in obs:
        return "Reset the project to see the sprint board."
    current_sprint = obs.get("current_sprint", 1)
    current_day    = obs.get("current_day", 1)
    s_tasks = [t for t in obs["tasks"]
               if t.get("metadata", {}).get("sprint") == current_sprint]
    sections: dict[str, list[str]] = {
        "in_progress": [], "backlog": [], "done": [], "missed": [], "blocked": []
    }
    for t in s_tasks:
        s = t["status"]
        if s not in sections: s = "backlog"
        filled = int(t["progress"] * 10)
        bar = "█" * filled + "░" * (10 - filled)
        te  = _TYPE_EMOJI.get(t["task_type"], "📌")
        pl  = _PRIO_LABEL[t["priority"]] if t["priority"] <= 5 else ""
        deps = t.get("metadata", {}).get("depends_on", [])
        dep_str = f" | Deps:{','.join(deps)}" if deps else ""
        sections[s].append(
            f"  {te} [{t['id']}] {t['name']}\n"
            f"     {pl} | Effort:{t['effort']}sp | Due:Day{t['deadline']}{dep_str}\n"
            f"     Dev:{t['assigned_to'] or '—'} | [{bar}] {t['progress']:.0%}"
        )
    day_in = ((current_day - 1) % 10) + 1
    d_bar  = "▓" * day_in + "░" * (10 - day_in)
    done_c = sum(1 for t in s_tasks if t["status"] == "done")
    lines  = [
        f"📋 SPRINT {current_sprint} BOARD  —  Day {day_in}/10  [{d_bar}]",
        f"✅{done_c} 🔄{sum(1 for t in s_tasks if t['status']=='in_progress')} "
        f"📋{sum(1 for t in s_tasks if t['status']=='backlog')} "
        f"❌{sum(1 for t in s_tasks if t['status']=='missed')}",
        "─" * 50,
    ]
    for key, label in [
        ("in_progress", "🔄 IN PROGRESS"), ("backlog", "📋 BACKLOG"),
        ("done", "✅ DONE"), ("missed", "❌ MISSED"), ("blocked", "🚫 BLOCKED"),
    ]:
        items = sections[key]
        if items:
            lines.append(f"\n{label} ({len(items)})")
            lines.extend(items)
    return "\n".join(lines)


def r2_format_developers(obs: dict) -> str:
    if not obs or "developers" not in obs:
        return ""
    lines = ["👥 TEAM WORKLOAD", "─" * 38, ""]
    for d in obs["developers"]:
        load, cap = d["current_load"], d["capacity"]
        pct    = load / cap if cap > 0 else 0
        filled = min(int(pct * 10), 10)
        bar    = "█" * filled + "░" * (10 - filled)
        status = "✅" if d["is_available"] else "🏖️"
        load_s = "🔴FULL" if pct >= 1.0 else ("🟡BUSY" if pct >= 0.6 else "🟢FREE")
        se     = _SKILL_EMOJI.get(d["skill"], "👤")
        tasks  = ", ".join(d["assigned_tasks"]) if d["assigned_tasks"] else "—"
        prod   = d.get("productivity", 1.0)
        lines += [
            f"{status} {d['name']} {se} ({d['skill']})  prod={prod:.2f}",
            f"  [{bar}] {load}/{cap}sp {load_s}",
            f"  Tasks: {tasks}",
            "",
        ]
    return "\n".join(lines)


def r2_format_instructions(obs: dict) -> str:
    if not obs: return ""
    queue   = obs.get("instruction_queue", [])
    inst_sc = obs.get("instruction_following_score", 1.0)
    i_bar   = "█" * int(inst_sc * 10) + "░" * (10 - int(inst_sc * 10))
    lines   = [
        f"📋 INSTRUCTION QUEUE  [{i_bar}] {inst_sc:.0%} followed",
        "─" * 48, "",
    ]
    if not queue:
        lines.append("  No instructions released yet.")
    else:
        for inst in queue[-12:]:
            followed   = inst.get("followed", False)
            icon       = "✅" if followed else "⚠️ "
            text_short = inst.get("text", "")[:55]
            if len(inst.get("text", "")) > 55: text_short += "…"
            lines.append(
                f"  {icon} [{inst['id']}] Day {inst['release_day']} → Sprint {inst['target_sprint']}"
            )
            lines.append(f"      {text_short}")
            lines.append("")
    return "\n".join(lines)


def r2_format_tech_debt(obs: dict) -> str:
    if not obs: return ""
    debt  = obs.get("tech_debt", [])
    tasks = {t["id"]: t for t in obs.get("tasks", [])}
    lines = [f"🔴 TECH DEBT  ({len(debt)} items)", "─" * 38, ""]
    if not debt:
        lines.append("  ✅ No tech debt — great execution!")
    else:
        for tid in debt:
            t    = tasks.get(tid, {})
            name = t.get("name", tid)
            sp   = t.get("metadata", {}).get("sprint", "?")
            lines.append(f"  🔴 {tid} — {name}  (was Sprint {sp})")
        lines.append("")
        lines.append(f"  ⚠️ {len(debt)} missed tasks dragging productivity")
    return "\n".join(lines)


def r2_format_metrics(obs: dict) -> str:
    if not obs: return ""
    bal    = obs.get("workload_balance_score", 0)
    inst_s = obs.get("instruction_following_score", 1.0)
    debt   = obs.get("tech_debt", [])
    spr_r  = obs.get("sprint_rewards", [])
    avg_sr = sum(spr_r) / len(spr_r) if spr_r else 0.0
    b_bar  = "█" * int(bal    * 10) + "░" * (10 - int(bal    * 10))
    i_bar  = "█" * int(inst_s * 10) + "░" * (10 - int(inst_s * 10))
    return (
        f"📊 Cumulative Reward  : {obs.get('cumulative_reward', 0):+.2f}\n"
        f"⚖️  Team Balance       : [{b_bar}] {bal:.2f}\n"
        f"📋 Inst Following     : [{i_bar}] {inst_s:.2f}\n"
        f"🔴 Tech Debt          : {len(debt)} tasks\n"
        f"🏅 Avg Sprint Score   : {avg_sr:.3f}\n"
        f"✅ Done               : {obs.get('tasks_completed', 0)}\n"
        f"❌ Missed             : {obs.get('tasks_missed', 0)}\n"
        f"🔄 In Progress        : {obs.get('tasks_in_progress', 0)}\n"
        f"📋 Backlog            : {obs.get('tasks_backlog', 0)}"
    )


def r2_make_reward_chart(obs: dict) -> str:
    sprint_rewards = obs.get("sprint_rewards", []) if obs else []
    history        = r2_reward_history
    lines = ["📈 PROJECT REWARD CHART", "─" * 48, ""]
    if sprint_rewards:
        lines.append("Sprint Scores:")
        for i, sc in enumerate(sprint_rewards):
            b_len = int(sc * 20)
            bar   = "█" * b_len + "░" * (20 - b_len)
            icon  = "✅" if sc >= 0.65 else ("⚠️" if sc >= 0.40 else "❌")
            lines.append(f"  {icon} S{i+1}: [{bar}] {sc:.3f}")
        lines.append("")
    if len(history) >= 2:
        cumulative = [r["cumulative"] for r in history]
        spark      = _sparkline(cumulative)
        lines.append(f"Cumulative: {spark}")
        lines.append(
            f"  min={min(cumulative):+.2f}  max={max(cumulative):+.2f}  "
            f"current={cumulative[-1]:+.2f}"
        )
    else:
        lines.append("Cumulative: (take actions to see chart)")
    return "\n".join(lines)


def _make_r2_outputs(obs_dict: dict, event_text: str):
    return (
        r2_format_timeline(obs_dict),
        r2_format_board(obs_dict),
        r2_format_developers(obs_dict),
        r2_format_instructions(obs_dict),
        r2_format_tech_debt(obs_dict),
        r2_format_metrics(obs_dict),
        r2_make_reward_chart(obs_dict),
        event_text,
        obs_dict,
    )


# ── R2 Gradio handlers ────────────────────────────────────────────────────────

def r2_reset_project(task_name: str):
    global r2_reward_history
    r2_reward_history = []
    obs = r2_env.reset(task_name=task_name, seed=42)
    r2_reward_history.append({"step": 0, "reward": 0.0, "cumulative": 0.0})
    return _make_r2_outputs(obs, "• Project started! 6 sprints · 60 days. Assign tasks to begin.")


def r2_take_action(action_type, task_id, dev_id, new_priority, task_ids_str, current_obs):
    try:
        kwargs = {
            "action_type":  action_type,
            "task_id":      task_id or None,
            "dev_id":       dev_id  or None,
            "new_priority": int(new_priority) if new_priority else None,
        }
        if action_type == "sprint_plan" and task_ids_str:
            kwargs["task_ids"] = [t.strip() for t in task_ids_str.split(",") if t.strip()]
        action = ProjectAction(**kwargs)
        obs, reward, done, info = r2_env.step(action)
        r2_reward_history.append({
            "step": len(r2_reward_history),
            "reward": reward,
            "cumulative": obs.get("cumulative_reward", 0),
        })
        ev = "\n".join(f"• {e}" for e in obs.get("events", []))
        if reward > 0:   ev += f"\n💰 Reward: +{reward:.2f}"
        elif reward < 0: ev += f"\n💸 Reward: {reward:.2f}"
        prev_sprints = len(current_obs.get("sprint_rewards", []))
        curr_sprints = len(obs.get("sprint_rewards", []))
        if curr_sprints > prev_sprints:
            sc = obs["sprint_rewards"][-1]
            ev += f"\n\n🏅 Sprint {curr_sprints} complete! Score: {sc:.3f}"
        if done:
            ev += f"\n\n🏁 PROJECT COMPLETE! Cumulative: {obs.get('cumulative_reward', 0):.2f}"
        return _make_r2_outputs(obs, ev)
    except Exception as e:
        return _make_r2_outputs(current_obs, f"❌ Error: {e}")


def r2_auto_sprint(current_obs: dict):
    """Auto-assign current sprint's backlog tasks, then advance one day."""
    if not current_obs or "tasks" not in current_obs:
        return _make_r2_outputs({}, "⚠️ Reset the project first!")
    obs_dict       = current_obs
    events_log     = []
    current_sprint = obs_dict.get("current_sprint", 1)
    backlog = sorted(
        [t for t in obs_dict["tasks"]
         if t["status"] == "backlog"
         and t.get("metadata", {}).get("sprint") == current_sprint],
        key=lambda t: (t["priority"], t["deadline"])
    )
    if not backlog:
        obs, reward, done, _ = r2_env.step(ProjectAction(action_type="skip"))
        r2_reward_history.append({
            "step": len(r2_reward_history), "reward": reward,
            "cumulative": obs.get("cumulative_reward", 0),
        })
        return _make_r2_outputs(obs, f"⏩ Day advanced — no backlog. reward={reward:+.2f}")
    devs = obs_dict.get("developers", [])
    for task in backlog:
        available   = [d for d in devs
                       if d["is_available"] and d["current_load"] < d["capacity"] * 2]
        skill_match = [d for d in available
                       if d["skill"] == task["required_skill"] or d["skill"] == "fullstack"]
        chosen      = skill_match[0] if skill_match else (available[0] if available else None)
        if chosen:
            action = ProjectAction(action_type="assign", task_id=task["id"], dev_id=chosen["id"])
            obs, reward, done, _ = r2_env.step(action)
            obs_dict = obs
            devs     = obs_dict.get("developers", [])
            r2_reward_history.append({
                "step": len(r2_reward_history), "reward": reward,
                "cumulative": obs_dict.get("cumulative_reward", 0),
            })
            events_log.append(f"✅ {task['id']} → {chosen['name']} (reward {reward:+.2f})")
            if done: break
        else:
            events_log.append(f"⚠️ No dev for {task['id']}")
    return _make_r2_outputs(obs_dict, "\n".join(events_log) or "No actions taken.")


def r2_advance_day(current_obs: dict):
    """Skip one day — lets scheduled instructions release."""
    if not current_obs or "tasks" not in current_obs:
        return _make_r2_outputs({}, "⚠️ Reset the project first!")
    obs, reward, done, _ = r2_env.step(ProjectAction(action_type="skip"))
    r2_reward_history.append({
        "step": len(r2_reward_history), "reward": reward,
        "cumulative": obs.get("cumulative_reward", 0),
    })
    events = "\n".join(f"• {e}" for e in obs.get("events", []))
    if done:
        events += f"\n\n🏁 PROJECT COMPLETE! Cumulative: {obs.get('cumulative_reward', 0):.2f}"
    return _make_r2_outputs(obs, events or f"⏩ Day advanced. reward={reward:+.2f}")


def r2_run_trained_agent(task_name: str):
    """
    Run the trained LLM on a full 60-day R2 project episode.
    Shows every step action + reward + instruction following score in the agent log.
    Falls back to rule-based if no HF_TOKEN is set.
    """
    import requests as _req

    api_key  = os.getenv("HF_TOKEN") or os.getenv("API_KEY", "")
    api_base = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
    model    = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-1.5B-Instruct")
    use_llm  = bool(api_key and api_key != "dummy")

    SYSTEM = (
        "You are an Engineering Manager on day {day}/60, sprint {sprint}/6. "
        "Output ONLY a JSON action: {\"action_type\":\"<assign|reassign|reprioritize|unblock|skip>\","
        "\"task_id\":\"<id or null>\",\"dev_id\":\"<id or null>\",\"new_priority\":<1-5 or null>}. "
        "ALWAYS act on active instructions first. Only assign tasks whose deps are done. "
        "Match developer skill to task required_skill."
    )

    def llm_r2_action(obs: dict) -> dict:
        active = [i for i in obs.get("instruction_queue", []) if not i.get("followed", False)]
        inst_s = "\n".join(f"[{i['id']}] {i['text'][:60]}" for i in active[:3]) or "None"
        debt   = obs.get("tech_debt", [])
        tasks  = obs.get("tasks", [])
        done_ids = {t["id"] for t in tasks if t["status"] == "done"}
        backlog  = sorted([t for t in tasks if t["status"] == "backlog"],
                          key=lambda t: (t["priority"], t["deadline"]))
        tasks_s  = "\n".join(
            f"[{t['id']}] {t['name']} P{t['priority']} skill={t['required_skill']} "
            f"deps_ok={all(d in done_ids for d in t.get('metadata',{}).get('depends_on',[]))}"
            for t in backlog[:8]
        )
        devs_s = "\n".join(
            f"[{d['id']}] {d['name']} {d['skill']} load={d['current_load']}/{d['capacity']} avail={'Y' if d['is_available'] else 'N'}"
            for d in obs.get("developers", [])
        )
        user_msg = (
            f"Day {obs['current_day']}/60 Sprint {obs.get('current_sprint',1)}/6\n"
            f"Instructions to follow:\n{inst_s}\n"
            f"Tech debt: {', '.join(debt) if debt else 'none'}\n"
            f"Backlog:\n{tasks_s}\nDevs:\n{devs_s}\nOutput JSON:"
        )
        sys_msg = SYSTEM.format(day=obs["current_day"], sprint=obs.get("current_sprint", 1))
        try:
            resp = _req.post(
                f"{api_base}/chat/completions",
                headers={"Authorization": f"Bearer {api_key}"},
                json={"model": model, "messages": [
                    {"role": "system", "content": sys_msg},
                    {"role": "user",   "content": user_msg},
                ], "max_tokens": 80, "temperature": 0.1},
                timeout=20,
            )
            text = resp.json()["choices"][0]["message"]["content"].strip()
            if "```" in text:
                text = "\n".join(l for l in text.split("\n") if not l.strip().startswith("```"))
            s, e = text.find("{"), text.rfind("}") + 1
            return json.loads(text[s:e]) if s >= 0 and e > s else {"action_type": "skip"}
        except Exception:
            return _rule_based_r2_dict(obs)

    def _rule_based_r2_dict(obs: dict) -> dict:
        tasks    = obs.get("tasks", [])
        devs     = obs.get("developers", [])
        done_ids = {t["id"] for t in tasks if t["status"] == "done"}
        avail    = [d for d in devs if d["is_available"] and d["current_load"] < d["capacity"] * 2]
        def best(task):
            m = [d for d in avail if d["skill"] == task.get("required_skill") or d["skill"] == "fullstack"]
            return m[0] if m else (avail[0] if avail else None)
        for inst in [i for i in obs.get("instruction_queue", []) if not i.get("followed", False)]:
            for tid in inst.get("affects_tasks", []):
                t = next((t for t in tasks if t["id"] == tid and t["status"] == "backlog"), None)
                if t and all(d in done_ids for d in t.get("metadata", {}).get("depends_on", [])):
                    dev = best(t)
                    if dev:
                        return {"action_type": "assign", "task_id": t["id"], "dev_id": dev["id"], "new_priority": None}
        backlog = sorted([t for t in tasks if t["status"] == "backlog"], key=lambda t: (t["priority"], t["deadline"]))
        for t in backlog:
            if all(d in done_ids for d in t.get("metadata", {}).get("depends_on", [])):
                dev = best(t)
                if dev:
                    return {"action_type": "assign", "task_id": t["id"], "dev_id": dev["id"], "new_priority": None}
        return {"action_type": "skip", "task_id": None, "dev_id": None, "new_priority": None}

    global r2_reward_history
    r2_reward_history = []
    obs = r2_env.reset(task_name=task_name, seed=42)
    r2_reward_history.append({"step": 0, "reward": 0.0, "cumulative": 0.0})

    mode = f"🤖 LLM ({model})" if use_llm else "🔧 Rule-based fallback (set HF_TOKEN for LLM)"
    logs = [f"{mode} — {task_name} — 60 steps", "─" * 45]

    for step in range(60):
        if obs.get("done", False):
            break
        action_dict = llm_r2_action(obs) if use_llm else _rule_based_r2_dict(obs)
        try:
            action = ProjectAction(**action_dict)
        except Exception:
            action = ProjectAction(action_type="skip")
        obs, reward, done, info = r2_env.step(action)
        r2_reward_history.append({
            "step": step + 1, "reward": reward,
            "cumulative": obs.get("cumulative_reward", 0),
        })
        inst_s = f"{obs.get('instruction_following_score', 0):.2f}"
        logs.append(
            f"D{obs['current_day']-1:02d}|S{obs.get('current_sprint',1)}: "
            f"{action.action_type:<11} {action.task_id or '':>4} "
            f"r={reward:+.2f} inst={inst_s} debt={len(obs.get('tech_debt',[]))}"
        )
        if done:
            logs.append(f"\n🏁 Project complete! Cumul: {obs.get('cumulative_reward', 0):.2f}")
            break

    return _make_r2_outputs(obs, "\n".join(logs))


# ═══════════════════════════════════════════════════════════════════════════════
# BUILD GRADIO UI — two tabs, single mount
# ═══════════════════════════════════════════════════════════════════════════════

CSS = """
.gradio-container { max-width: 1400px; margin: auto; }
footer { display: none !important; }
"""

with gr.Blocks(title="🤖 AI Sprint Manager", css=CSS) as demo:

    gr.Markdown("""
    # 🤖 AI Sprint Manager — OpenEnv
    **Round 1:** Single-sprint RL · (10 days · Max 12 tasks ) &nbsp;|&nbsp;
    **Round 2:** Long-horizon 6-sprint project management (60 days · 50+ tasks · adaptive instructions)
    """)

    with gr.Tabs():

        # ══════════════════════════════════════════════════════════════════════
        # TAB 1 — ROUND 1  (all controls & wiring identical to uploaded ui.py)
        # ══════════════════════════════════════════════════════════════════════
        with gr.TabItem("🏃 Round 1 — Sprint Manager"):

            r1_obs_state = gr.State({})

            gr.Markdown("### Single-Sprint RL Environment")
            with gr.Row():
                r1_task_sel  = gr.Dropdown(choices=SCENARIO_NAMES, value=SCENARIO_NAMES[0],
                                            label="🎯 Sprint Scenario", scale=2)
                r1_reset_btn = gr.Button("🔄 Reset Sprint",    variant="primary",  scale=1)
                r1_auto_btn  = gr.Button("🤖 Auto-Assign All", variant="secondary", scale=1)

            with gr.Row():
                with gr.Column(scale=3):
                    r1_board = gr.Textbox(label="📋 Sprint Board", lines=26, interactive=False,
                                          value="👆 Select a scenario and click Reset Sprint to begin!")
                with gr.Column(scale=2):
                    r1_dev   = gr.Textbox(label="👥 Team Workload",     lines=9,  interactive=False)
                    r1_skill = gr.Textbox(label="🎯 Skill → Dev Guide", lines=9,  interactive=False)
                    r1_metr  = gr.Textbox(label="📊 Sprint Metrics",    lines=8,  interactive=False)

            with gr.Row():
                r1_rchart = gr.Textbox(label="📈 Reward History", lines=14, interactive=False)
                r1_tchart = gr.Textbox(label="📊 Task Status",    lines=14, interactive=False)

            gr.Markdown("### 🤖 Run Trained LLM Agent")
            with gr.Row():
                r1_agent_btn = gr.Button("▶️ Run LLM Agent (Qwen2.5-1.5B)", variant="primary", scale=1)
                r1_agent_log = gr.Textbox(
                    label="🤖 Agent Log — step-by-step actions and rewards",
                    lines=14, interactive=False, scale=3,
                    value="Click ▶️ Run LLM Agent to watch the model manage the sprint step by step.\n"
                          "Each line shows: Day | action | task→dev | reward | cumulative reward\n"
                          "(Set HF_TOKEN env var to use the actual LLM; otherwise rule-based fallback runs.)"
                )

            gr.Markdown("### 🎮 Manual Action")
            with gr.Row():
                r1_at  = gr.Dropdown(choices=["assign","reassign","reprioritize","unblock","skip"],
                                      value="assign", label="Action", scale=1)
                r1_tid = gr.Textbox(label="Task ID",  placeholder="e.g. T1",   scale=1)
                r1_did = gr.Textbox(label="Dev ID",   placeholder="e.g. dev1", scale=1)
                r1_pri = gr.Dropdown(choices=["","1","2","3","4","5"], value="",
                                      label="Priority (reprioritize only)", scale=1)
                r1_act = gr.Button("▶️ Take Action", variant="primary", scale=1)

            r1_elog = gr.Textbox(label="📜 Event Log", lines=4, interactive=False)

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

            R1_OUT = [r1_board, r1_dev, r1_skill, r1_elog,
                      r1_metr, r1_rchart, r1_tchart, r1_obs_state]
            # Agent button returns same R1_OUT but uses r1_elog as the log display
            R1_AGENT_OUT = [r1_board, r1_dev, r1_skill, r1_agent_log,
                            r1_metr, r1_rchart, r1_tchart, r1_obs_state]

            r1_reset_btn.click(fn=r1_reset_env,         inputs=[r1_task_sel],                         outputs=R1_OUT)
            r1_auto_btn.click( fn=r1_auto_assign,       inputs=[r1_obs_state],                        outputs=R1_OUT)
            r1_act.click(      fn=r1_take_action,
                               inputs=[r1_at, r1_tid, r1_did, r1_pri, r1_obs_state],                 outputs=R1_OUT)
            r1_agent_btn.click(fn=r1_run_trained_agent, inputs=[r1_task_sel],                         outputs=R1_AGENT_OUT)

        # ══════════════════════════════════════════════════════════════════════
        # TAB 2 — ROUND 2  (new multi-sprint project environment)
        # ══════════════════════════════════════════════════════════════════════
        with gr.TabItem("🚀 Round 2 — Project Manager"):

            r2_obs_state = gr.State({})

            gr.Markdown("""
            ### Long-Horizon Sprint Planning — 6 Sprints · 60 Days · Adaptive Instructions
            Instructions drip-feed over time. Missed tasks become **tech debt** that slows the team.
            Cascade failures cross sprint boundaries. Score = delivery × instruction-following × team health.
            """)

            with gr.Row():
                r2_task_sel   = gr.Dropdown(choices=VALID_PROJECT_TASK_NAMES, value="project_easy",
                                             label="🎯 Project Scenario", scale=2)
                r2_reset_btn  = gr.Button("🔄 Reset Project",      variant="primary",   scale=1)
                r2_auto_btn   = gr.Button("🤖 Auto-Assign Sprint",  variant="secondary", scale=1)
                r2_adv_btn    = gr.Button("⏩ Advance Day",          variant="secondary", scale=1)

            with gr.Row():
                with gr.Column(scale=2):
                    r2_timeline = gr.Textbox(
                        label="🗓️ Sprint Timeline", lines=16, interactive=False,
                        value="👆 Select a project scenario and click Reset Project to begin!"
                    )
                with gr.Column(scale=3):
                    r2_board = gr.Textbox(label="📋 Current Sprint Board", lines=16, interactive=False)
                with gr.Column(scale=2):
                    r2_devs  = gr.Textbox(label="👥 Team Workload",        lines=16, interactive=False)

            with gr.Row():
                r2_inst  = gr.Textbox(label="📋 Instruction Queue",  lines=12, interactive=False, scale=2)
                r2_debt  = gr.Textbox(label="🔴 Tech Debt Tracker",  lines=12, interactive=False, scale=1)
                r2_metr  = gr.Textbox(label="📊 Project Metrics",    lines=12, interactive=False, scale=1)

            r2_rchart = gr.Textbox(label="📈 Cross-Sprint Reward Chart", lines=12, interactive=False)

            gr.Markdown("### 🤖 Run Trained LLM Agent (Round 2)")
            with gr.Row():
                r2_agent_btn = gr.Button("▶️ Run LLM Agent (60-day project)", variant="primary", scale=1)
                r2_agent_log = gr.Textbox(
                    label="🤖 R2 Agent Log — Day|Sprint | action | reward | inst_score | debt",
                    lines=14, interactive=False, scale=3,
                    value="Click ▶️ Run LLM Agent to watch the model manage the full 60-day project.\n"
                          "Format: D{day}|S{sprint}: {action} {task} r={reward} inst={score} debt={n}\n"
                          "(Set HF_TOKEN env var to use the actual LLM; otherwise rule-based fallback runs.)"
                )

            gr.Markdown("### 🎮 Manual Action")
            with gr.Row():
                r2_at   = gr.Dropdown(
                    choices=["assign","reassign","reprioritize","unblock","skip","sprint_plan"],
                    value="assign", label="Action", scale=1)
                r2_tid  = gr.Textbox(label="Task ID",    placeholder="e.g. T01",      scale=1)
                r2_did  = gr.Textbox(label="Dev ID",     placeholder="e.g. dev1",     scale=1)
                r2_pri  = gr.Dropdown(choices=["","1","2","3","4","5"], value="",
                                       label="Priority (reprioritize)", scale=1)
                r2_tids = gr.Textbox(label="Task IDs (sprint_plan, comma-sep)",
                                      placeholder="T01,T02,T03", scale=2)
                r2_act  = gr.Button("▶️ Take Action", variant="primary", scale=1)

            r2_elog = gr.Textbox(label="📜 Event Log", lines=5, interactive=False)

            gr.Markdown("""
            ---
            | Action | When | Example |
            |--------|------|---------|
            | `assign` | Assign backlog task to a dev | Task=T01, Dev=dev1 |
            | `reassign` | Move task to another dev | Task=T05, Dev=dev3 |
            | `reprioritize` | Change task priority | Task=T08, Priority=1 |
            | `unblock` | Clear a blocked task | Task=T03 |
            | `skip` | Advance 1 day (releases instructions) | — |
            | `sprint_plan` | **R2 new** — batch plan for sprint | Task IDs=T09,T10,T11 |

            **Tip:** Check the Instruction Queue and act on flagged tasks for bonus rewards.
            Tech debt from missed tasks reduces team productivity in future sprints.
            """)

            R2_OUT = [
                r2_timeline, r2_board, r2_devs,
                r2_inst, r2_debt, r2_metr,
                r2_rchart, r2_elog, r2_obs_state,
            ]
            R2_AGENT_OUT = [
                r2_timeline, r2_board, r2_devs,
                r2_inst, r2_debt, r2_metr,
                r2_rchart, r2_agent_log, r2_obs_state,
            ]

            r2_reset_btn.click(fn=r2_reset_project,     inputs=[r2_task_sel],               outputs=R2_OUT)
            r2_auto_btn.click( fn=r2_auto_sprint,       inputs=[r2_obs_state],              outputs=R2_OUT)
            r2_adv_btn.click(  fn=r2_advance_day,       inputs=[r2_obs_state],              outputs=R2_OUT)
            r2_act.click(      fn=r2_take_action,
                               inputs=[r2_at, r2_tid, r2_did, r2_pri, r2_tids, r2_obs_state],
                               outputs=R2_OUT)
            r2_agent_btn.click(fn=r2_run_trained_agent, inputs=[r2_task_sel],               outputs=R2_AGENT_OUT)

# ── Mount into FastAPI — single port 7860 ─────────────────────────────────────
app = gr.mount_gradio_app(api, demo, path="/")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)