"""
AI Sprint Manager — Gradio UI + FastAPI
Round 1: single-sprint environment
Round 2: long-horizon multi-sprint project environment

  [SESSION]  Per-user isolation. Every browser session gets its own
             SprintManagerEnv / ProjectManagerEnv + reward history stored
             inside gr.State. No shared global env.

  [API]      FastAPI /reset + /step are stateless per-call with
             optional episode_id for multi-step clients. No shared env.

  [MODEL]    The fine-tuned agent (sejal-k/multi-sprint-model) is not
             included in this UI — HF Spaces has no GPU and dedicated
             inference endpoints require paid hosting. The demo supports
             manual actions and rule-based auto-assign. See blog post
             "Future Improvements" for the integration plan.

BUGS FIXED:
  1. R1 sprint complete shows "Score: 0.00/1.0" — info key was 'final_score'
     but env likely returns 'score'; now tries both keys with fallback to
     cumulative_reward normalised.
  2. R2 timeline sprint icons always show ❌ — icon was derived from
     task-completion % (0/5 = 0%) instead of the sprint reward score.
     Fixed to use spr_rews[s-1] directly for completed sprints.
  3. R2 reset_project didn't call .model_dump() on the observation —
     causing downstream .get() calls to fail when env returns a Pydantic
     model. Fixed to mirror R1 pattern.
  4. R2 take/auto/advance handlers had a no-op isinstance guard
     `obs if isinstance(obs, dict) else obs` that never converted Pydantic
     models. Replaced with a proper _to_dict() helper.
  5. _SKILL_EMOJI had 'devops' mapped to 🚀 but R1 easy_sprint labels
     Carol as devops — the emoji was correct, but the Skills reference
     table in the UI said "devops → Carol" while the code comment said
     "devops → Carol" too, so no fix needed there.  The real mismatch was
     that R1 format_developers used _SKILL_EMOJI which has no 'devops'
     key producing the fallback 👤 — added 'devops' to _SKILL_EMOJI.
"""

from __future__ import annotations

import json
import os
import threading
import uuid
import warnings

import gradio as gr
import uvicorn
from fastapi import FastAPI

warnings.filterwarnings("ignore", message=".*max_new_tokens.*")
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

# ── Environment imports ───────────────────────────────────────────────────────
from sprint_env.environment         import SprintManagerEnv
from sprint_env.models              import SprintAction
from sprint_env.data_loader         import load_sprint_data, get_scenario_names
from sprint_env.project_environment import ProjectManagerEnv, VALID_PROJECT_TASK_NAMES
from sprint_env.project_models      import ProjectAction
from server.project_app             import project_router

SCENARIO_NAMES = get_scenario_names()
_sprint_data   = load_sprint_data()


def _to_dict(obs) -> dict:
    """Safely convert a Pydantic model or plain dict observation to dict."""
    if isinstance(obs, dict):
        return obs
    if hasattr(obs, "model_dump"):
        return obs.model_dump()
    if hasattr(obs, "dict"):          # Pydantic v1 compat
        return obs.dict()
    return dict(obs)


def _parse_json(raw: str) -> dict:
    """Parse JSON action from model output. Returns skip dict on any failure."""
    _skip = {"action_type": "skip", "task_id": None, "dev_id": None, "new_priority": None}
    if not raw:
        return _skip
    if "```" in raw:
        raw = "\n".join(l for l in raw.split("\n") if not l.strip().startswith("```"))
    s, e = raw.find("{"), raw.rfind("}") + 1
    if s < 0 or e <= s:
        return _skip
    try:
        d  = json.loads(raw[s:e])
        at = str(d.get("action_type", "skip")).lower()
        if at not in ("assign", "reassign", "reprioritize", "unblock", "skip"):
            at = "skip"
        return {"action_type": at, "task_id": d.get("task_id"),
                "dev_id": d.get("dev_id"), "new_priority": d.get("new_priority")}
    except Exception:
        return _skip


# ══════════════════════════════════════════════════════════════════════════════
# FastAPI — stateless per-call with optional episode cache
# ══════════════════════════════════════════════════════════════════════════════

_api_episodes: dict[str, SprintManagerEnv]    = {}
_api_r2_episodes: dict[str, ProjectManagerEnv] = {}
_api_lock = threading.Lock()

api = FastAPI(title="AI Sprint Manager — OpenEnv", version="2.0.0")


@api.post("/reset")
def api_reset(req: dict = {}):
    eid = req.get("episode_id") or str(uuid.uuid4())
    env = SprintManagerEnv()
    with _api_lock:
        _api_episodes[eid] = env
    obs = env.reset(task_name=req.get("task_name", "easy_sprint"), seed=req.get("seed"))
    r   = _to_dict(obs)
    r["episode_id"] = eid
    return r


@api.post("/step")
def api_step(req: dict):
    eid = req.get("episode_id")
    with _api_lock:
        env = _api_episodes.get(eid)
    if env is None:
        return {"error": "episode_id not found — call /reset first"}
    action = SprintAction(**req.get("action", {}))
    obs, reward, done, info = env.step(action)
    return {"observation": _to_dict(obs), "reward": reward, "done": done, "info": info}


@api.get("/state")
def api_state():
    with _api_lock:
        env = (list(_api_episodes.values()) or [SprintManagerEnv()])[-1]
    return env.state.model_dump()


@api.get("/health")
def api_health():
    return {"status": "ok", "env": "ai-sprint-manager"}


@api.get("/tasks")
def api_tasks():
    return {"tasks": [{"id": k, "description": v.get("description", ""),
                        "difficulty": v.get("difficulty", "")}
                       for k, v in _sprint_data["scenarios"].items()]}


api.include_router(project_router)


# ══════════════════════════════════════════════════════════════════════════════
# Shared UI formatters  (stateless — take obs dict, return string)
# ══════════════════════════════════════════════════════════════════════════════

_TYPE_EMOJI  = {"feature": "🔧", "bug": "🐛", "urgent_bug": "🚨", "tech_debt": "🔩"}
_PRIO_LABEL  = ["", "🔴P1", "🟠P2", "🟡P3", "🟢P4", "⚪P5"]
# FIX 5: added 'devops' so Carol's skill renders 🚀 instead of fallback 👤
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
        f"📈 REWARD HISTORY (Step 0 → {len(history)-1})", "─" * 45,
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
        if s in counts: counts[s] += 1
    config = [("done","✅ Done       ","#"),("in_progress","🔄 In Progress","="),
              ("backlog","📋 Backlog    ","·"),("missed","❌ Missed     ","!"),("blocked","🚫 Blocked    ","?")]
    lines = [f"📊 TASK STATUS ({total} total)", "─" * 40]
    for key, label, char in config:
        count   = counts[key]
        bar_len = int((count / total) * 24) if total else 0
        pct     = count / total * 100         if total else 0
        bar     = char * bar_len + "·" * (24 - bar_len)
        lines.append(f"{label}: [{bar}] {count} ({pct:.0f}%)")
    lines += ["", f"Sprint completion: {counts['done']}/{total} tasks done"]
    if total:
        cp = int(counts["done"] / total * 20)
        lines.append(f"[{'█'*cp}{'░'*(20-cp)}] {counts['done']/total*100:.0f}%")
    return "\n".join(lines)


def format_sprint_board(obs: dict) -> str:
    if not obs or "tasks" not in obs:
        return "👆 Select a scenario and click Reset Sprint to begin!"
    sections: dict[str, list[str]] = {k: [] for k in ("in_progress","backlog","done","missed","blocked")}
    for t in obs["tasks"]:
        s = t["status"] if t["status"] in sections else "backlog"
        filled = int(t["progress"] * 10)
        bar = "█" * filled + "░" * (10 - filled)
        te  = _TYPE_EMOJI.get(t["task_type"], "📌")
        pl  = _PRIO_LABEL[t["priority"]] if t["priority"] <= 5 else ""
        sections[s].append(
            f"  {te} [{t['id']}] {t['name']}\n"
            f"     {pl} | Effort:{t['effort']}sp | Due:Day{t['deadline']} | {t['required_skill']}\n"
            f"     Dev:{t['assigned_to'] or '—'} | [{bar}] {t['progress']:.0%}"
        )
    day  = int(obs.get("current_day", 1))
    slen = int(obs.get("sprint_length", 10))
    lines = [
        f"📅 Day {day}/{slen}  [{'▓'*day}{'░'*(slen-day)}]",
        f"✅{obs['tasks_completed']} 🔄{obs['tasks_in_progress']} 📋{obs['tasks_backlog']} ❌{obs['tasks_missed']}",
        "─" * 50,
    ]
    for key, label in [("in_progress","🔄 IN PROGRESS"),("backlog","📋 BACKLOG"),
                        ("done","✅ DONE"),("missed","❌ MISSED"),("blocked","🚫 BLOCKED")]:
        if sections[key]:
            lines.append(f"\n{label} ({len(sections[key])})")
            lines.extend(sections[key])
    return "\n".join(lines)


def format_developers(obs: dict) -> str:
    if not obs or "developers" not in obs:
        return ""
    lines = ["👥 TEAM WORKLOAD", "─" * 38, ""]
    for d in obs["developers"]:
        load, cap = d["current_load"], d["capacity"]
        pct    = load / cap if cap else 0
        filled = min(int(pct * 10), 10)
        bar    = "█" * filled + "░" * (10 - filled)
        status = "✅" if d["is_available"] else "🤒"
        load_s = "🔴FULL" if pct >= 1.0 else ("🟡BUSY" if pct >= 0.6 else "🟢FREE")
        se     = _SKILL_EMOJI.get(d["skill"], "👤")
        tasks  = ", ".join(d["assigned_tasks"]) if d["assigned_tasks"] else "—"
        lines += [f"{status} {d['name']} {se} ({d['skill']})",
                  f"  [{bar}] {load}/{cap}sp {load_s}",
                  f"  Tasks: {tasks}", ""]
    return "\n".join(lines)


def format_skill_table(obs: dict) -> str:
    if not obs or "developers" not in obs:
        return ""
    lines = ["🎯 SKILL → DEV GUIDE", "─" * 38, ""]
    skill_groups: dict[str, list[str]] = {}
    for d in obs["developers"]:
        avail = "✅" if d["is_available"] and d["current_load"] < d["capacity"] else "❌"
        skill_groups.setdefault(d["skill"], []).append(
            f"  {avail} {d['name']} ({d['id']}) {d['current_load']}/{d['capacity']}sp")
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
    bar    = "█" * int(bal * 10) + "░" * (10 - int(bal * 10))
    return (f"📊 Cumulative Reward : {obs.get('cumulative_reward', 0):+.2f}\n"
            f"⚖️  Balance           : [{bar}] {bal:.2f}\n"
            f"✅ Done              : {obs.get('tasks_completed', 0)}\n"
            f"❌ Missed            : {obs.get('tasks_missed', 0)}\n"
            f"🔄 In Progress       : {obs.get('tasks_in_progress', 0)}\n"
            f"📋 Backlog           : {obs.get('tasks_backlog', 0)}")


# ══════════════════════════════════════════════════════════════════════════════
# Per-session state factories
# ══════════════════════════════════════════════════════════════════════════════

def _new_r1_session() -> dict:
    """Each browser session gets its own isolated R1 env + history."""
    return {"obs": {}, "reward_history": [], "env": SprintManagerEnv()}


def _new_r2_session() -> dict:
    return {"obs": {}, "reward_history": [], "env": ProjectManagerEnv()}


def _make_r1_outputs(sess: dict, event_text: str) -> tuple:
    obs = sess["obs"]
    return (format_sprint_board(obs), format_developers(obs), format_skill_table(obs),
            event_text, format_metrics(obs),
            make_reward_chart(sess["reward_history"]), make_task_chart(obs), sess)


# ══════════════════════════════════════════════════════════════════════════════
# R1 — Gradio handlers (all take + return session dict)
# ══════════════════════════════════════════════════════════════════════════════

def r1_reset_env(task_name: str, sess: dict) -> tuple:
    if not sess or "env" not in sess:
        sess = _new_r1_session()
    sess["reward_history"] = []
    sess["done"] = False          # clear completed flag on reset
    obs = sess["env"].reset(task_name=task_name, seed=42)
    sess["obs"] = _to_dict(obs)
    sess["reward_history"].append({"step": 0, "reward": 0.0, "cumulative": 0.0})
    return _make_r1_outputs(sess, "• Sprint started! Assign tasks to begin.")


def r1_take_action(action_type, task_id, dev_id, new_priority, sess: dict) -> tuple:
    if not sess or "env" not in sess:
        return _make_r1_outputs(_new_r1_session(), "⚠️ Reset the sprint first!")
    # FIX: guard against stepping on a completed episode — env freezes and
    # day counter stops advancing, giving the illusion nothing happened.
    if sess.get("done"):
        return _make_r1_outputs(sess, "🏁 Sprint is complete — click Reset Sprint to start a new one.")
    try:
        action = SprintAction(action_type=action_type, task_id=task_id or None,
                              dev_id=dev_id or None,
                              new_priority=int(new_priority) if new_priority else None)
        obs, reward, done, info = sess["env"].step(action)
        sess["obs"] = _to_dict(obs)
        sess["reward_history"].append({"step": len(sess["reward_history"]),
                                       "reward": reward, "cumulative": sess["obs"]["cumulative_reward"]})
        ev = format_events(sess["obs"])
        ev += f"\n{'💰' if reward >= 0 else '💸'} Reward: {reward:+.2f}"
        if done:
            sess["done"] = True   # mark episode finished so future actions are blocked
            # FIX 1: env may return 'score', 'final_score', or neither.
            # Try multiple keys; fall back to normalising cumulative reward.
            final_score = (
                info.get("final_score")
                or info.get("score")
                or info.get("episode_score")
                or round(max(0.0, sess["obs"].get("cumulative_reward", 0)) / 100.0, 2)
            )
            ev += f"\n\n🏁 SPRINT COMPLETE! Score: {final_score:.2f}/1.0"
        return _make_r1_outputs(sess, ev)
    except Exception as e:
        return _make_r1_outputs(sess, f"❌ Error: {e}")


def r1_auto_assign(sess: dict) -> tuple:
    if not sess or "env" not in sess or not sess.get("obs"):
        return _make_r1_outputs(_new_r1_session(), "⚠️ Reset the sprint first!")
    obs_dict = sess["obs"]
    devs     = list(obs_dict.get("developers", []))
    backlog  = sorted([t for t in obs_dict.get("tasks", []) if t["status"] == "backlog"],
                      key=lambda t: (t["priority"], t["deadline"]))
    if not backlog:
        return _make_r1_outputs(sess, "✅ No backlog tasks to assign!")
    events_log = []
    for task in backlog:
        available   = [d for d in devs if d["is_available"] and d["current_load"] < d["capacity"]]
        skill_match = [d for d in available if d["skill"] == task["required_skill"] or d["skill"] == "fullstack"]
        chosen = skill_match[0] if skill_match else (available[0] if available else None)
        if chosen:
            obs, reward, done, _ = sess["env"].step(
                SprintAction(action_type="assign", task_id=task["id"], dev_id=chosen["id"]))
            sess["obs"] = _to_dict(obs)
            devs = sess["obs"]["developers"]
            sess["reward_history"].append({"step": len(sess["reward_history"]),
                                           "reward": reward, "cumulative": sess["obs"]["cumulative_reward"]})
            events_log.append(f"✅ {task['id']} → {chosen['name']} (r={reward:+.2f})")
        else:
            events_log.append(f"⚠️ No available dev for {task['id']}")
    return _make_r1_outputs(sess, "\n".join(events_log))


# ── R1 rule-based fallback ────────────────────────────────────────────────────

def _r1_rule_based(obs_dict: dict, assigned_ids: set) -> SprintAction:
    tasks   = obs_dict.get("tasks", [])
    devs    = obs_dict.get("developers", [])
    avail   = [d for d in devs if d["is_available"] and d["current_load"] < d["capacity"]]
    backlog = sorted([t for t in tasks if t["status"] == "backlog" and t["id"] not in assigned_ids],
                     key=lambda t: (t["priority"], t["deadline"]))
    for task in backlog:
        match = [d for d in avail if d["skill"] == task["required_skill"] or d["skill"] == "fullstack"]
        dev   = match[0] if match else (avail[0] if avail else None)
        if dev:
            return SprintAction(action_type="assign", task_id=task["id"], dev_id=dev["id"])
    return SprintAction(action_type="skip")


# ══════════════════════════════════════════════════════════════════════════════
# R2 — formatters
# ══════════════════════════════════════════════════════════════════════════════

def r2_format_timeline(obs: dict) -> str:
    if not obs or "tasks" not in obs:
        return "👆 Select a project scenario and click Reset Project to begin!"
    cur_sprint = obs.get("current_sprint", 1)
    cur_day    = obs.get("current_day", 1)
    spr_rews   = obs.get("sprint_rewards", [])
    tasks      = obs.get("tasks", [])
    project_done = cur_day > 60 or (cur_sprint > 6)

    # Clamp display values — env emits day 61 on final step; cap for display
    display_day    = min(cur_day, 60)
    display_sprint = min(cur_sprint, 6)

    lines = [f"🗓️  PROJECT TIMELINE  —  Day {display_day}/60  |  Sprint {display_sprint}/6",
             "─" * 56, ""]
    for s in range(1, 7):
        st    = [t for t in tasks if t.get("metadata", {}).get("sprint") == s]
        total = len(st)
        actually_done   = sum(1 for t in st if t["status"] == "done")
        actually_missed = sum(1 for t in st if t["status"] == "missed")
        actually_blocked= sum(1 for t in st if t["status"] == "blocked")
        # Bar reflects true completion (done only) — honest progress indicator
        pct = actually_done / total * 100 if total else 0
        bar = "█" * int(pct / 10) + "░" * (10 - int(pct / 10))

        # Build a compact status suffix: e.g. "2✅ 1❌ 2🚫" when there are misses
        def _sprint_label(d, m, b, tot):
            if d == tot:
                return f"{d}/{tot} ✅"
            parts = [f"{d}✅"] if d else []
            if m: parts.append(f"{m}❌")
            if b: parts.append(f"{b}🚫")
            remaining = tot - d - m - b
            if remaining: parts.append(f"{remaining}📋")
            return f"{d}/{tot} (" + " ".join(parts) + ")" if parts else f"{d}/{tot}"

        # A sprint is "past" if we have a reward recorded for it
        sprint_is_past    = (s-1) < len(spr_rews)
        sprint_is_current = (s == display_sprint) and not project_done

        if sprint_is_past and (s < display_sprint or project_done):
            rew  = spr_rews[s-1]
            icon = "✅" if rew >= 0.65 else ("⚠️" if rew >= 0.40 else "❌")
            lbl  = _sprint_label(actually_done, actually_missed, actually_blocked, total)
            lines.append(f"  {icon} Sprint {s} (D{(s-1)*10+1}-{s*10}): [{bar}] {lbl}  score={rew:.2f}")
        elif sprint_is_current:
            di  = ((cur_day - 1) % 10) + 1
            pb  = "▓" * di + "░" * (10 - di)
            lbl = _sprint_label(actually_done, actually_missed, actually_blocked, total)
            lines.append(f"  🏃 Sprint {s} (D{(s-1)*10+1}-{s*10}): [{bar}] {lbl}  day {di}/10 [{pb}]")
        else:
            lines.append(f"  ⏳ Sprint {s} (D{(s-1)*10+1}-{s*10}): {'·'*10}  {total} tasks queued")
    lines.append("")
    od  = sum(1 for t in tasks if t["status"] == "done")
    om  = sum(1 for t in tasks if t["status"] == "missed")
    ob  = sum(1 for t in tasks if t["status"] == "blocked")
    ot  = len(tasks)
    pp  = od / ot * 100 if ot else 0
    pf  = int(pp / 5)
    summary = f"📦 Project: [{'█'*pf}{'░'*(20-pf)}] {od}/{ot} done"
    if om or ob:
        summary += f"  ({om}❌ missed, {ob}🚫 blocked)"
    lines.append(summary)
    if project_done:
        lines.append("🏁 PROJECT COMPLETE")
    return "\n".join(lines)


def r2_format_board(obs: dict) -> str:
    if not obs or "tasks" not in obs:
        return "Reset the project to see the sprint board."
    cur_sprint   = obs.get("current_sprint", 1)
    cur_day      = obs.get("current_day", 1)
    project_done = cur_day > 60 or cur_sprint > 6

    # Clamp to final sprint/day for display purposes
    display_sprint = min(cur_sprint, 6)
    display_day    = min(cur_day, 60)

    s_tasks = [t for t in obs["tasks"] if t.get("metadata", {}).get("sprint") == display_sprint]
    sections: dict[str, list[str]] = {k: [] for k in ("in_progress","backlog","done","missed","blocked")}
    for t in s_tasks:
        s = t["status"] if t["status"] in sections else "backlog"
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
    if project_done:
        # Sprint 6 is complete — show final day 10/10 and COMPLETE banner
        di    = 10
        d_bar = "▓" * 10
    else:
        di    = ((display_day - 1) % 10) + 1
        d_bar = "▓" * di + "░" * (10 - di)
    dc   = sum(1 for t in s_tasks if t["status"] == "done")
    header = f"📋 SPRINT {display_sprint} BOARD  {'🏁 COMPLETE' if project_done else f'Day {di}/10 [{d_bar}]'}"
    lines = [header, f"  {dc}/{len(s_tasks)} tasks done", "─" * 50]
    for key, label in [("in_progress","🔄 IN PROGRESS"),("backlog","📋 BACKLOG"),
                        ("done","✅ DONE"),("missed","❌ MISSED"),("blocked","🚫 BLOCKED")]:
        if sections[key]:
            lines.append(f"\n{label} ({len(sections[key])})")
            lines.extend(sections[key])
    return "\n".join(lines)


def r2_format_developers(obs: dict) -> str:
    if not obs or "developers" not in obs:
        return ""
    lines = ["👥 TEAM WORKLOAD", "─" * 38, ""]
    for d in obs["developers"]:
        load, cap = d.get("current_load", 0), d.get("capacity", 5)
        prod  = d.get("productivity", 1.0)
        pct   = load / cap if cap else 0
        bar   = "█" * min(int(pct * 10), 10) + "░" * max(10 - int(pct * 10), 0)
        status = "✅" if d["is_available"] else "🤒"
        load_s = "🔴FULL" if pct >= 1.0 else ("🟡BUSY" if pct >= 0.6 else "🟢FREE")
        prod_s = f" ⚠️prod={prod:.1f}" if prod < 0.95 else ""
        se     = _SKILL_EMOJI.get(d["skill"], "👤")
        lines += [f"{status} {d['name']} {se} ({d['skill']}){prod_s}",
                  f"  [{bar}] {load}/{cap}sp {load_s}", ""]
    return "\n".join(lines)


def r2_format_instructions(obs: dict) -> str:
    queue = obs.get("instruction_queue", []) if obs else []
    lines = [f"📋 INSTRUCTION QUEUE  ({len(queue)} total)", "─" * 38, ""]
    if not queue:
        lines.append("  (none yet — instructions drip-feed by day)")
    else:
        for inst in queue[-12:]:
            icon = "✅" if inst.get("followed", False) else "⚠️ "
            txt  = inst.get("text", "")[:55] + ("…" if len(inst.get("text","")) > 55 else "")
            lines += [f"  {icon} [{inst['id']}] Day {inst['release_day']} → Sprint {inst['target_sprint']}",
                      f"      {txt}", ""]
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
            t = tasks.get(tid, {})
            lines.append(f"  🔴 {tid} — {t.get('name', tid)}  (was Sprint {t.get('metadata',{}).get('sprint','?')})")
        lines += ["", f"  ⚠️ {len(debt)} missed tasks dragging productivity"]
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
    return (f"📊 Cumulative Reward  : {obs.get('cumulative_reward', 0):+.2f}\n"
            f"⚖️  Team Balance       : [{b_bar}] {bal:.2f}\n"
            f"📋 Inst Following     : [{i_bar}] {inst_s:.2f}\n"
            f"🔴 Tech Debt          : {len(debt)} tasks\n"
            f"🏅 Avg Sprint Score   : {avg_sr:.3f}\n"
            f"✅ Done               : {obs.get('tasks_completed', 0)}\n"
            f"❌ Missed             : {obs.get('tasks_missed', 0)}\n"
            f"🔄 In Progress        : {obs.get('tasks_in_progress', 0)}\n"
            f"📋 Backlog            : {obs.get('tasks_backlog', 0)}")


def r2_make_reward_chart(obs: dict, history: list) -> str:
    spr_rews = obs.get("sprint_rewards", []) if obs else []
    lines = ["📈 PROJECT REWARD CHART", "─" * 48, ""]
    if spr_rews:
        lines.append("Sprint Scores:")
        for i, sc in enumerate(spr_rews):
            bar  = "█" * int(sc * 20) + "░" * (20 - int(sc * 20))
            icon = "✅" if sc >= 0.65 else ("⚠️" if sc >= 0.40 else "❌")
            lines.append(f"  {icon} S{i+1}: [{bar}] {sc:.3f}")
        lines.append("")
    if len(history) >= 2:
        cumulative = [r["cumulative"] for r in history]
        lines += [f"Cumulative: {_sparkline(cumulative)}",
                  f"  min={min(cumulative):+.2f}  max={max(cumulative):+.2f}  current={cumulative[-1]:+.2f}"]
    else:
        lines.append("Cumulative: (take actions to see chart)")
    return "\n".join(lines)


def _make_r2_outputs(sess: dict, event_text: str) -> tuple:
    obs = sess["obs"]
    return (r2_format_timeline(obs), r2_format_board(obs), r2_format_developers(obs),
            r2_format_instructions(obs), r2_format_tech_debt(obs), r2_format_metrics(obs),
            r2_make_reward_chart(obs, sess["reward_history"]), event_text, sess)


# ── R2 Gradio handlers ────────────────────────────────────────────────────────

def r2_reset_project(task_name: str, sess: dict) -> tuple:
    if not sess or "env" not in sess:
        sess = _new_r2_session()
    sess["reward_history"] = []
    sess["done"] = False          # clear completed flag on reset
    obs = sess["env"].reset(task_name=task_name, seed=42)
    # FIX 3: always store as plain dict so downstream .get() calls work
    sess["obs"] = _to_dict(obs)
    sess["reward_history"].append({"step": 0, "reward": 0.0, "cumulative": 0.0})
    return _make_r2_outputs(sess, "• Project started! 6 sprints · 60 days. Click Auto-Assign Sprint to begin.")


def r2_take_action(action_type, task_id, dev_id, new_priority, task_ids_str, sess: dict) -> tuple:
    if not sess or "env" not in sess:
        return _make_r2_outputs(_new_r2_session(), "⚠️ Reset the project first!")
    if sess.get("done"):
        return _make_r2_outputs(sess, "🏁 Project is complete — click Reset Project to start a new one.")
    try:
        kwargs = {"action_type": action_type, "task_id": task_id or None,
                  "dev_id": dev_id or None,
                  "new_priority": int(new_priority) if new_priority else None}
        if action_type == "sprint_plan" and task_ids_str:
            kwargs["task_ids"] = [t.strip() for t in task_ids_str.split(",") if t.strip()]
        obs, reward, done, info = sess["env"].step(ProjectAction(**kwargs))
        # FIX 4: properly convert Pydantic model to dict
        sess["obs"] = _to_dict(obs)
        sess["reward_history"].append({"step": len(sess["reward_history"]), "reward": reward,
                                       "cumulative": sess["obs"].get("cumulative_reward", 0)})
        ev = "\n".join(f"• {e}" for e in sess["obs"].get("events", []))
        ev += f"\n{'💰' if reward >= 0 else '💸'} Reward: {reward:+.2f}"
        if done:
            sess["done"] = True
            ev += f"\n\n🏁 PROJECT COMPLETE! Cumulative: {sess['obs'].get('cumulative_reward', 0):.2f}"
        return _make_r2_outputs(sess, ev)
    except Exception as e:
        return _make_r2_outputs(sess, f"❌ Error: {e}")


def _r2_do_auto_assign(sess: dict) -> str:
    """
    Assign ONE backlog task per call (the highest-priority assignable one).
    Each env.step() advances the day, so bulk-assigning wastes days.
    Returns a string event log. Mutates sess in place.
    """
    obs_dict   = sess["obs"]
    cur_sprint = obs_dict.get("current_sprint", 1)
    cur_day    = obs_dict.get("current_day", 1)
    all_tasks  = obs_dict.get("tasks", [])
    done_ids   = {t["id"] for t in all_tasks if t["status"] == "done"}

    # Assignable: backlog, current sprint, dependencies met
    backlog = sorted(
        [t for t in all_tasks
         if t["status"] == "backlog"
         and t.get("metadata", {}).get("sprint") == cur_sprint
         and all(d in done_ids for d in t.get("metadata", {}).get("depends_on", []))],
        key=lambda t: (t["priority"], t["deadline"])
    )

    # Count tasks waiting on deps (informational)
    waiting = [t for t in all_tasks
               if t["status"] == "backlog"
               and t.get("metadata", {}).get("sprint") == cur_sprint
               and not all(d in done_ids for d in t.get("metadata", {}).get("depends_on", []))]

    if not backlog:
        if waiting:
            ids = ", ".join(t["id"] for t in waiting)
            return f"⏳ All remaining tasks blocked by unmet deps: {ids}\n   Advance days to let in-progress tasks complete."
        return "✅ No backlog tasks to assign for this sprint."

    # Pick the single highest-priority task
    task = backlog[0]
    devs = obs_dict.get("developers", [])
    available   = [d for d in devs if d["is_available"] and d["current_load"] < d["capacity"]]
    skill_match = [d for d in available if d["skill"] == task["required_skill"] or d["skill"] == "fullstack"]
    chosen = skill_match[0] if skill_match else (available[0] if available else None)

    if not chosen:
        return f"⚠️ No available dev for {task['id']} ({task['required_skill']}) — all at capacity. Advance a day."

    obs, reward, done, _ = sess["env"].step(
        ProjectAction(action_type="assign", task_id=task["id"], dev_id=chosen["id"]))
    sess["obs"] = _to_dict(obs)
    sess["reward_history"].append({"step": len(sess["reward_history"]), "reward": reward,
                                   "cumulative": sess["obs"].get("cumulative_reward", 0)})
    if done:
        sess["done"] = True

    # How many more are assignable after this step
    new_obs     = sess["obs"]
    new_tasks   = new_obs.get("tasks", [])
    new_done    = {t["id"] for t in new_tasks if t["status"] == "done"}
    more = [t for t in new_tasks
            if t["status"] == "backlog"
            and t.get("metadata", {}).get("sprint") == new_obs.get("current_sprint", 1)
            and all(d in new_done for d in t.get("metadata", {}).get("depends_on", []))]

    result = f"✅ Assigned {task['id']} ({task['name'][:28]}) → {chosen['name']} (r={reward:+.2f})"
    if more:
        result += f"\n   {len(more)} more task(s) ready — click Auto-Assign again or advance days."
    elif waiting:
        result += f"\n   ⏳ {len(waiting)} task(s) waiting on deps — advance days to unblock."
    return result


def r2_auto_sprint(sess: dict) -> tuple:
    if not sess or "env" not in sess or not sess.get("obs"):
        return _make_r2_outputs(_new_r2_session(), "⚠️ Reset the project first!")
    if sess.get("done"):
        return _make_r2_outputs(sess, "🏁 Project is complete — click Reset Project to start a new one.")
    result = _r2_do_auto_assign(sess)
    return _make_r2_outputs(sess, result)


def r2_advance_day(sess: dict) -> tuple:
    if not sess or "env" not in sess or not sess.get("obs"):
        return _make_r2_outputs(_new_r2_session(), "⚠️ Reset the project first!")
    if sess.get("done"):
        return _make_r2_outputs(sess, "🏁 Project is complete — click Reset Project to start a new one.")

    sprint_before = sess["obs"].get("current_sprint", 1)
    obs, reward, done, _ = sess["env"].step(ProjectAction(action_type="skip"))
    sess["obs"] = _to_dict(obs)
    sess["reward_history"].append({"step": len(sess["reward_history"]), "reward": reward,
                                   "cumulative": sess["obs"].get("cumulative_reward", 0)})
    ev = "\n".join(f"• {e}" for e in sess["obs"].get("events", []))
    if done:
        sess["done"] = True
        ev += f"\n\n🏁 PROJECT COMPLETE! Cumulative: {sess['obs'].get('cumulative_reward', 0):.2f}"
        return _make_r2_outputs(sess, ev or f"⏩ Day advanced. r={reward:+.2f}")

    # Notify on sprint transition so user knows to re-run Auto-Assign
    sprint_after = sess["obs"].get("current_sprint", 1)
    if sprint_after > sprint_before:
        ev += f"\n\n🔄 Sprint {sprint_after} started! Click Auto-Assign Sprint to assign the new backlog."

    return _make_r2_outputs(sess, ev or f"⏩ Day advanced. r={reward:+.2f}")


def _r2_rule_based(obs: dict, assigned: set) -> dict:
    tasks    = obs.get("tasks", [])
    devs     = obs.get("developers", [])
    done_ids = {t["id"] for t in tasks if t["status"] == "done"}
    avail    = [d for d in devs if d["is_available"] and d["current_load"] < d["capacity"] * 2]
    def best(task):
        m = [d for d in avail if d["skill"] == task.get("required_skill") or d["skill"] == "fullstack"]
        return m[0] if m else (avail[0] if avail else None)
    for inst in [i for i in obs.get("instruction_queue", []) if not i.get("followed", False)]:
        for tid in inst.get("affects_tasks", []):
            t = next((t for t in tasks if t["id"] == tid and t["status"] == "backlog"
                      and tid not in assigned), None)
            if t and all(d in done_ids for d in t.get("metadata", {}).get("depends_on", [])):
                dev = best(t)
                if dev:
                    return {"action_type": "assign", "task_id": t["id"], "dev_id": dev["id"], "new_priority": None}
    for t in sorted([t for t in tasks if t["status"] == "backlog" and t["id"] not in assigned],
                    key=lambda t: (t["priority"], t["deadline"])):
        if all(d in done_ids for d in t.get("metadata", {}).get("depends_on", [])):
            dev = best(t)
            if dev:
                return {"action_type": "assign", "task_id": t["id"], "dev_id": dev["id"], "new_priority": None}
    return {"action_type": "skip", "task_id": None, "dev_id": None, "new_priority": None}


# ══════════════════════════════════════════════════════════════════════════════
# BUILD GRADIO UI
# ══════════════════════════════════════════════════════════════════════════════

CSS = """
.gradio-container { max-width: 1400px; margin: auto; }
footer { display: none !important; }
"""

with gr.Blocks(title="🤖 AI Sprint Manager", css=CSS) as demo:

    gr.Markdown("""
    # 🤖 AI Sprint Manager — OpenEnv
    **Round 1:** Single-sprint RL · 10 days · up to 12 tasks
    """)

    r1_sess = gr.State(_new_r1_session)   # per-user, isolated

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

    gr.Markdown("### 🎮 Manual Action")
    with gr.Row():
        r1_at  = gr.Dropdown(choices=["assign","reassign","reprioritize","skip"],
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

    R1_OUT = [r1_board, r1_dev, r1_skill, r1_elog, r1_metr, r1_rchart, r1_tchart, r1_sess]

    r1_reset_btn.click(fn=r1_reset_env,   inputs=[r1_task_sel, r1_sess],                   outputs=R1_OUT)
    r1_auto_btn.click( fn=r1_auto_assign,  inputs=[r1_sess],                                outputs=R1_OUT)
    r1_act.click(      fn=r1_take_action,  inputs=[r1_at, r1_tid, r1_did, r1_pri, r1_sess], outputs=R1_OUT)

app = gr.mount_gradio_app(api, demo, path="/")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)