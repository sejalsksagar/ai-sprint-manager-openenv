"""
AI Sprint Manager — Gradio UI + FastAPI
Round 1: single-sprint environment
Round 2: long-horizon multi-sprint project environment

KEY CHANGES:
  [SESSION]  Per-user isolation. Every browser session gets its own
             SprintManagerEnv / ProjectManagerEnv + reward history stored
             inside gr.State. No shared global env. One user's reset or
             agent run cannot corrupt another user's board.

  [MODEL]    Trained model (sejal-k/multi-sprint-model) loaded ONCE at
             startup via Unsloth 4-bit. Falls back to HF router API if
             Unsloth unavailable, then to rule-based if no token.

  [ATTRIBUTION] Every agent step is tagged [LLM] or [FB] (fallback) in
             the log. Guard fires, loop fires, and fallback triggers are
             counted separately. The final summary line shows:
               LLM actions: N/total  |  LLM valid rate: X%
               Fallback actions: M/total  (guard/loop/parse)
             This lets you audit exactly how much of the score came from
             the trained model vs the rule-based fallback.

  [API]      FastAPI /reset + /step are now stateless per-call with
             optional episode_id for multi-step clients. No shared env.
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

LOCAL_MODEL_PATH = os.getenv("LOCAL_MODEL_PATH", "sejal-k/multi-sprint-model")
HF_TOKEN         = os.getenv("HF_TOKEN", "")

# ══════════════════════════════════════════════════════════════════════════════
# Trained model — loaded once, lazily, in a thread-safe way
# ══════════════════════════════════════════════════════════════════════════════

# ── Local model disabled: always use HF Router ───────────────────────────────
# Unsloth and HF Transformers loading are skipped intentionally.
# All LLM calls go through the HF Inference Router (OpenAI-compatible API).

_local_model     = None   # always None — local loading disabled
_local_tokenizer = None
_model_source    = "hf_router"

HF_ROUTER_MODEL  = os.getenv("HF_ROUTER_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
HF_ROUTER_BASE   = os.getenv("API_BASE_URL",     "https://router.huggingface.co/v1")


def _load_trained_model() -> bool:
    """Local model loading disabled. Always delegate to HF Router."""
    print("[MODEL] Local model loading skipped — using HF Router.", flush=True)
    return False


def _hf_router_call(system_prompt: str, user_prompt: str,
                    temperature: float = 0.3, max_tokens: int = 96) -> str:
    """Call HF Router (OpenAI-compatible). Returns raw text or empty string."""
    if not HF_TOKEN:
        return ""
    try:
        import requests as _req
        resp = _req.post(
            f"{HF_ROUTER_BASE}/chat/completions",
            headers={"Authorization": f"Bearer {HF_TOKEN}"},
            json={
                "model": HF_ROUTER_MODEL,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_prompt},
                ],
                "max_tokens": max_tokens,
                "temperature": temperature,
            },
            timeout=20,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(f"[MODEL] HF Router error: {e}", flush=True)
        return ""


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
    r   = obs.model_dump()
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
    return {"observation": obs.model_dump(), "reward": reward, "done": done, "info": info}


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
    obs = sess["env"].reset(task_name=task_name, seed=42)
    sess["obs"] = obs.model_dump()
    sess["reward_history"].append({"step": 0, "reward": 0.0, "cumulative": 0.0})
    return _make_r1_outputs(sess, "• Sprint started! Assign tasks to begin.")


def r1_take_action(action_type, task_id, dev_id, new_priority, sess: dict) -> tuple:
    if not sess or "env" not in sess:
        return _make_r1_outputs(_new_r1_session(), "⚠️ Reset the sprint first!")
    try:
        action = SprintAction(action_type=action_type, task_id=task_id or None,
                              dev_id=dev_id or None,
                              new_priority=int(new_priority) if new_priority else None)
        obs, reward, done, info = sess["env"].step(action)
        sess["obs"] = obs.model_dump()
        sess["reward_history"].append({"step": len(sess["reward_history"]),
                                       "reward": reward, "cumulative": sess["obs"]["cumulative_reward"]})
        ev = format_events(sess["obs"])
        ev += f"\n{'💰' if reward >= 0 else '💸'} Reward: {reward:+.2f}"
        if done:
            ev += f"\n\n🏁 SPRINT COMPLETE! Score: {info.get('final_score', 0):.2f}/1.0"
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
            sess["obs"] = obs.model_dump()
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


R1_SYSTEM = (
    "You are an expert Tech Lead. Output ONLY a single JSON action:\n"
    '{"action_type":"<assign|reassign|reprioritize|unblock|skip>",'
    '"task_id":"<id or null>","dev_id":"<id or null>","new_priority":<1-5 or null>}\n'
    "Rules: assign only BACKLOG tasks to AVAILABLE developers with matching skill. "
    "Never assign a task already assigned this episode. No markdown, no explanation."
)


def _r1_llm_action(obs_dict: dict, assigned_ids: set) -> tuple[SprintAction, str]:
    """
    Returns (action, source) where source is one of:
      "llm_valid"   — model output parsed and passed guard checks
      "llm_guard"   — model output parsed but failed guard → fallback used
      "llm_parse"   — model output could not be parsed → fallback used
      "llm_api_err" — API/network error → fallback used
      "rule_based"  — no model available, rule-based used directly
    """
    if not HF_TOKEN:
        return _r1_rule_based(obs_dict, assigned_ids), "rule_based"

    tasks_s = "\n".join(
        f"[{t['id']}] {t['name']} P{t['priority']} {t['status']} "
        f"skill={t['required_skill']} assigned={t['assigned_to'] or '-'}"
        for t in obs_dict["tasks"])
    devs_s = "\n".join(
        f"[{d['id']}] {d['name']} skill={d['skill']} "
        f"load={d['current_load']}/{d['capacity']} avail={d['is_available']}"
        for d in obs_dict["developers"])
    user_msg = (
        f"Day {obs_dict['current_day']}/{obs_dict['sprint_length']} "
        f"done={obs_dict['tasks_completed']} missed={obs_dict['tasks_missed']}\n"
        f"Already assigned this episode (DO NOT assign again): "
        f"{', '.join(assigned_ids) or 'none'}\n"
        f"TASKS:\n{tasks_s}\nDEVS:\n{devs_s}\nOutput JSON action:"
    )

    raw = _hf_router_call(R1_SYSTEM, user_msg, temperature=0.3, max_tokens=80)
    if not raw:
        return _r1_rule_based(obs_dict, assigned_ids), "llm_api_err"

    d = _parse_json(raw)
    if d["action_type"] == "skip" and not raw:
        return _r1_rule_based(obs_dict, assigned_ids), "llm_parse"

    # Guard: don't re-assign same task this episode
    if d["action_type"] == "assign" and d.get("task_id") in assigned_ids:
        return _r1_rule_based(obs_dict, assigned_ids), "llm_guard"

    try:
        action = SprintAction(**d)
        return action, "llm_valid"
    except Exception:
        return _r1_rule_based(obs_dict, assigned_ids), "llm_guard"


def r1_run_trained_agent(task_name: str, sess: dict) -> tuple:
    """
    Run the trained model for a full R1 episode. Each step is tagged [LLM] or [FB].
    Final summary shows LLM valid rate so you can see actual model contribution.
    """
    if not sess or "env" not in sess:
        sess = _new_r1_session()

    if HF_TOKEN:
        mode_label = f"🌐 HF Router ({HF_ROUTER_MODEL}) via {HF_ROUTER_BASE}"
    else:
        mode_label = "🔧 Rule-based only (set HF_TOKEN to enable HF Router)"

    sess["reward_history"] = []
    obs = sess["env"].reset(task_name=task_name, seed=42)
    sess["obs"] = obs.model_dump()
    sess["reward_history"].append({"step": 0, "reward": 0.0, "cumulative": 0.0})

    step_logs      = [mode_label, f"Scenario: {task_name}", "─" * 52]
    assigned_ids: set = set()
    last_key          = None
    loop_count        = 0

    # Attribution counters
    counts = {"llm_valid": 0, "llm_guard": 0, "llm_parse": 0, "llm_api_err": 0, "rule_based": 0}

    for step in range(12):
        if sess["obs"].get("done"):
            break

        # Loop detection: same (action_type, task_id) twice in a row → force fallback
        action, source = _r1_llm_action(sess["obs"], assigned_ids)
        key = (action.action_type, action.task_id)
        if key == last_key and source == "llm_valid":
            # Model is looping — override
            action = _r1_rule_based(sess["obs"], assigned_ids)
            source = "llm_guard"
            loop_count += 1
        else:
            loop_count = 0
        last_key = key

        counts[source] += 1
        if action.action_type == "assign" and action.task_id:
            assigned_ids.add(action.task_id)

        obs, reward, done, info = sess["env"].step(action)
        sess["obs"] = obs.model_dump()
        sess["reward_history"].append({"step": step + 1, "reward": reward,
                                       "cumulative": sess["obs"]["cumulative_reward"]})

        tag       = "[LLM]" if source == "llm_valid" else "[FB] "
        guard_tag = {"llm_guard": "guard", "llm_parse": "parse",
                     "llm_api_err": "api_err", "rule_based": "rule"}.get(source, "")
        fb_note   = f" ← {guard_tag}" if guard_tag else ""
        t_part    = f"→ {action.task_id}" if action.task_id else "      "
        d_part    = f"/ {action.dev_id}" if action.dev_id else ""
        step_logs.append(
            f"{tag} Day {sess['obs']['current_day']:02d}: "
            f"{action.action_type:<11} {t_part} {d_part} "
            f"r={reward:+.2f} cumul={sess['obs']['cumulative_reward']:+.2f}{fb_note}"
        )
        if done:
            score = info.get("final_score", 0.01)
            step_logs.append(f"\n🏁 Sprint done! Score: {score:.4f}/0.99")
            break

    # Attribution summary
    total = sum(counts.values())
    llm_v = counts["llm_valid"]
    fb_total = total - llm_v
    llm_pct  = llm_v / total * 100 if total else 0
    step_logs += [
        "─" * 52,
        "📊 ATTRIBUTION SUMMARY",
        f"  [LLM] Valid model actions : {llm_v}/{total}  ({llm_pct:.0f}% of steps)",
        f"  [FB]  Fallback actions     : {fb_total}/{total}  ({100-llm_pct:.0f}% of steps)",
        f"        breakdown → guard:{counts['llm_guard']}  parse:{counts['llm_parse']}  "
        f"api_err:{counts['llm_api_err']}  pure_rule:{counts['rule_based']}",
        "",
        "  If LLM% is low, the score is mostly from the rule-based fallback,",
        "  not the trained model. Raise temperature or fix guard issues.",
    ]
    return _make_r1_outputs(sess, "\n".join(step_logs))


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
    lines = [f"🗓️  PROJECT TIMELINE  —  Day {cur_day}/60  |  Sprint {cur_sprint}/6",
             "─" * 56, ""]
    for s in range(1, 7):
        st    = [t for t in tasks if t.get("metadata", {}).get("sprint") == s]
        done  = sum(1 for t in st if t["status"] == "done")
        total = len(st)
        pct   = done / total * 100 if total else 0
        bar   = "█" * int(pct / 10) + "░" * (10 - int(pct / 10))
        if s < cur_sprint:
            rew  = spr_rews[s-1] if (s-1) < len(spr_rews) else 0.0
            icon = "✅" if pct >= 70 else ("⚠️" if pct >= 40 else "❌")
            lines.append(f"  {icon} Sprint {s} (D{(s-1)*10+1}-{s*10}): [{bar}] {done}/{total}  score={rew:.2f}")
        elif s == cur_sprint:
            di  = ((cur_day - 1) % 10) + 1
            pb  = "▓" * di + "░" * (10 - di)
            lines.append(f"  🏃 Sprint {s} (D{(s-1)*10+1}-{s*10}): [{bar}] {done}/{total}  day {di}/10 [{pb}]")
        else:
            lines.append(f"  ⏳ Sprint {s} (D{(s-1)*10+1}-{s*10}): {'·'*10}  {total} tasks queued")
    lines.append("")
    od = sum(1 for t in tasks if t["status"] == "done")
    ot = len(tasks)
    pp = od / ot * 100 if ot else 0
    pf = int(pp / 5)
    lines.append(f"📦 Project: [{'█'*pf}{'░'*(20-pf)}] {od}/{ot} ({pp:.0f}%)")
    return "\n".join(lines)


def r2_format_board(obs: dict) -> str:
    if not obs or "tasks" not in obs:
        return "Reset the project to see the sprint board."
    cur_sprint = obs.get("current_sprint", 1)
    cur_day    = obs.get("current_day", 1)
    s_tasks = [t for t in obs["tasks"] if t.get("metadata", {}).get("sprint") == cur_sprint]
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
    di   = ((cur_day - 1) % 10) + 1
    d_bar = "▓" * di + "░" * (10 - di)
    dc   = sum(1 for t in s_tasks if t["status"] == "done")
    lines = [f"📋 SPRINT {cur_sprint} BOARD  Day {di}/10 [{d_bar}]",
             f"  {dc}/{len(s_tasks)} tasks done", "─" * 50]
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
    obs = sess["env"].reset(task_name=task_name, seed=42)
    sess["obs"] = obs if isinstance(obs, dict) else obs
    sess["reward_history"].append({"step": 0, "reward": 0.0, "cumulative": 0.0})
    return _make_r2_outputs(sess, "• Project started! 6 sprints · 60 days. Assign tasks to begin.")


def r2_take_action(action_type, task_id, dev_id, new_priority, task_ids_str, sess: dict) -> tuple:
    if not sess or "env" not in sess:
        return _make_r2_outputs(_new_r2_session(), "⚠️ Reset the project first!")
    try:
        kwargs = {"action_type": action_type, "task_id": task_id or None,
                  "dev_id": dev_id or None,
                  "new_priority": int(new_priority) if new_priority else None}
        if action_type == "sprint_plan" and task_ids_str:
            kwargs["task_ids"] = [t.strip() for t in task_ids_str.split(",") if t.strip()]
        obs, reward, done, info = sess["env"].step(ProjectAction(**kwargs))
        sess["obs"] = obs if isinstance(obs, dict) else obs
        sess["reward_history"].append({"step": len(sess["reward_history"]), "reward": reward,
                                       "cumulative": sess["obs"].get("cumulative_reward", 0)})
        ev = "\n".join(f"• {e}" for e in sess["obs"].get("events", []))
        ev += f"\n{'💰' if reward >= 0 else '💸'} Reward: {reward:+.2f}"
        if done:
            ev += f"\n\n🏁 PROJECT COMPLETE! Cumulative: {sess['obs'].get('cumulative_reward', 0):.2f}"
        return _make_r2_outputs(sess, ev)
    except Exception as e:
        return _make_r2_outputs(sess, f"❌ Error: {e}")


def r2_auto_sprint(sess: dict) -> tuple:
    if not sess or "env" not in sess or not sess.get("obs"):
        return _make_r2_outputs(_new_r2_session(), "⚠️ Reset the project first!")
    obs_dict = sess["obs"]
    cur_sprint = obs_dict.get("current_sprint", 1)
    backlog = sorted([t for t in obs_dict["tasks"]
                      if t["status"] == "backlog" and t.get("metadata", {}).get("sprint") == cur_sprint],
                     key=lambda t: (t["priority"], t["deadline"]))
    if not backlog:
        obs, reward, done, _ = sess["env"].step(ProjectAction(action_type="skip"))
        sess["obs"] = obs if isinstance(obs, dict) else obs
        sess["reward_history"].append({"step": len(sess["reward_history"]), "reward": reward,
                                       "cumulative": sess["obs"].get("cumulative_reward", 0)})
        return _make_r2_outputs(sess, f"⏩ Day advanced — no backlog. r={reward:+.2f}")
    devs = obs_dict.get("developers", [])
    events_log = []
    for task in backlog:
        available   = [d for d in devs if d["is_available"] and d["current_load"] < d["capacity"] * 2]
        skill_match = [d for d in available if d["skill"] == task["required_skill"] or d["skill"] == "fullstack"]
        chosen = skill_match[0] if skill_match else (available[0] if available else None)
        if chosen:
            obs, reward, done, _ = sess["env"].step(
                ProjectAction(action_type="assign", task_id=task["id"], dev_id=chosen["id"]))
            sess["obs"] = obs if isinstance(obs, dict) else obs
            devs = sess["obs"].get("developers", [])
            sess["reward_history"].append({"step": len(sess["reward_history"]), "reward": reward,
                                           "cumulative": sess["obs"].get("cumulative_reward", 0)})
            events_log.append(f"✅ {task['id']} → {chosen['name']} (r={reward:+.2f})")
            if done: break
        else:
            events_log.append(f"⚠️ No dev for {task['id']}")
    return _make_r2_outputs(sess, "\n".join(events_log) or "No actions taken.")


def r2_advance_day(sess: dict) -> tuple:
    if not sess or "env" not in sess or not sess.get("obs"):
        return _make_r2_outputs(_new_r2_session(), "⚠️ Reset the project first!")
    obs, reward, done, _ = sess["env"].step(ProjectAction(action_type="skip"))
    sess["obs"] = obs if isinstance(obs, dict) else obs
    sess["reward_history"].append({"step": len(sess["reward_history"]), "reward": reward,
                                   "cumulative": sess["obs"].get("cumulative_reward", 0)})
    ev = "\n".join(f"• {e}" for e in sess["obs"].get("events", []))
    if done:
        ev += f"\n\n🏁 PROJECT COMPLETE! Cumulative: {sess['obs'].get('cumulative_reward', 0):.2f}"
    return _make_r2_outputs(sess, ev or f"⏩ Day advanced. r={reward:+.2f}")


R2_SYSTEM = (
    "You are an Engineering Manager. Output ONLY a JSON action each step.\n"
    '{"action_type":"<assign|reassign|reprioritize|unblock|skip>",'
    '"task_id":"<id or null>","dev_id":"<id or null>","new_priority":<1-5 or null>}\n'
    "Rules:\n"
    "- Follow ACTIVE INSTRUCTIONS first — assign their named tasks immediately\n"
    "- Only assign tasks in BACKLOG status (not in_progress, not done)\n"
    "- Only assign if all dependencies are done and developer is available\n"
    "- Never assign a task already assigned this episode\n"
    "- skip only as last resort\n"
    "Output ONLY the JSON. No explanation."
)


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


def _r2_llm_action(obs: dict, assigned: set) -> tuple[dict, str]:
    """Same attribution tagging as R1. Returns (action_dict, source)."""
    if not HF_TOKEN:
        return _r2_rule_based(obs, assigned), "rule_based"

    active  = [i for i in obs.get("instruction_queue", []) if not i.get("followed", False)]
    inst_s  = "\n".join(f"[{i['id']}] {i['text'][:60]}" for i in active[:3]) or "None"
    done_ids = {t["id"] for t in obs.get("tasks", []) if t["status"] == "done"}
    backlog  = sorted([t for t in obs.get("tasks", []) if t["status"] == "backlog" and t["id"] not in assigned],
                      key=lambda t: (t["priority"], t["deadline"]))
    tasks_s  = "\n".join(
        f"[{t['id']}] {t['name']} P{t['priority']} skill={t['required_skill']} "
        f"deps_ok={all(d in done_ids for d in t.get('metadata',{}).get('depends_on',[]))}"
        for t in backlog[:8])
    devs_s = "\n".join(
        f"[{d['id']}] {d['name']} {d['skill']} load={d['current_load']}/{d['capacity']} "
        f"avail={'Y' if d['is_available'] else 'N'}"
        for d in obs.get("developers", []))
    user_msg = (
        f"Day {obs['current_day']}/60 Sprint {obs.get('current_sprint',1)}/6\n"
        f"Active instructions:\n{inst_s}\n"
        f"Already assigned this episode (skip): {', '.join(assigned) or 'none'}\n"
        f"Backlog:\n{tasks_s}\nDevs:\n{devs_s}\nOutput JSON:"
    )

    raw = _hf_router_call(R2_SYSTEM, user_msg, temperature=0.5, max_tokens=80)
    if not raw:
        return _r2_rule_based(obs, assigned), "llm_api_err"

    d = _parse_json(raw)
    if not raw:
        return _r2_rule_based(obs, assigned), "llm_parse"
    if d["action_type"] == "assign" and d.get("task_id") in assigned:
        return _r2_rule_based(obs, assigned), "llm_guard"
    return d, "llm_valid"


def r2_run_trained_agent(task_name: str, sess: dict) -> tuple:
    if not sess or "env" not in sess:
        sess = _new_r2_session()

    if HF_TOKEN:
        mode_label = f"🌐 HF Router ({HF_ROUTER_MODEL}) via {HF_ROUTER_BASE}"
    else:
        mode_label = "🔧 Rule-based only (set HF_TOKEN to enable HF Router)"

    sess["reward_history"] = []
    obs = sess["env"].reset(task_name=task_name, seed=42)
    sess["obs"] = obs if isinstance(obs, dict) else obs
    sess["reward_history"].append({"step": 0, "reward": 0.0, "cumulative": 0.0})

    logs = [mode_label, f"Scenario: {task_name}  |  60 steps max", "─" * 56]
    assigned: set = set()
    last_key      = None
    loop_count    = 0
    counts = {"llm_valid": 0, "llm_guard": 0, "llm_parse": 0, "llm_api_err": 0, "rule_based": 0}

    for step in range(60):
        obs_dict = sess["obs"]
        if obs_dict.get("done", False):
            break

        action_dict, source = _r2_llm_action(obs_dict, assigned)

        # Loop detection
        key = (action_dict.get("action_type"), action_dict.get("task_id"))
        if key == last_key and source == "llm_valid":
            loop_count += 1
            if loop_count >= 2:
                action_dict = _r2_rule_based(obs_dict, assigned)
                source = "llm_guard"
                loop_count = 0
        else:
            loop_count = 0
        last_key = key

        counts[source] += 1
        try:
            action = ProjectAction(**action_dict)
        except Exception:
            action = ProjectAction(action_type="skip")
            source = "llm_guard"
            counts["llm_guard"] += 1

        if action.action_type == "assign" and action.task_id:
            assigned.add(action.task_id)

        obs, reward, done, info = sess["env"].step(action)
        sess["obs"] = obs if isinstance(obs, dict) else obs
        obs_dict    = sess["obs"]
        sess["reward_history"].append({"step": step + 1, "reward": reward,
                                       "cumulative": obs_dict.get("cumulative_reward", 0)})

        tag      = "[LLM]" if source == "llm_valid" else "[FB] "
        fb_note  = {"llm_guard": "guard", "llm_parse": "parse",
                    "llm_api_err": "api_err", "rule_based": "rule"}.get(source, "")
        fb_annot = f" ←{fb_note}" if fb_note else ""
        logs.append(
            f"{tag} D{obs_dict.get('current_day',0)-1:02d}|S{obs_dict.get('current_sprint',1)}: "
            f"{action.action_type:<11} {action.task_id or '':>4} "
            f"r={reward:+.2f} inst={obs_dict.get('instruction_following_score',0):.2f} "
            f"debt={len(obs_dict.get('tech_debt',[]))}{fb_annot}"
        )
        if done:
            logs.append(f"\n🏁 Project complete! Cumul: {obs_dict.get('cumulative_reward', 0):.2f}")
            break

    total   = sum(counts.values())
    llm_v   = counts["llm_valid"]
    fb_total = total - llm_v
    llm_pct  = llm_v / total * 100 if total else 0
    logs += [
        "─" * 56,
        "📊 ATTRIBUTION SUMMARY",
        f"  [LLM] Valid model actions : {llm_v}/{total}  ({llm_pct:.0f}% of steps)",
        f"  [FB]  Fallback actions     : {fb_total}/{total}  ({100-llm_pct:.0f}% of steps)",
        f"        breakdown → guard:{counts['llm_guard']}  parse:{counts['llm_parse']}  "
        f"api_err:{counts['llm_api_err']}  pure_rule:{counts['rule_based']}",
        "",
        "  Score is trustworthy when LLM% > 70%. Below that, fallback is",
        "  doing most of the work — check guard fire reasons above.",
    ]
    return _make_r2_outputs(sess, "\n".join(logs))


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
    **Round 1:** Single-sprint RL · 10 days · up to 12 tasks &nbsp;|&nbsp;
    **Round 2:** Long-horizon 6-sprint project · 60 days · 50+ tasks · adaptive instructions
    """)

    with gr.Tabs():

        # ── TAB 1 — ROUND 1 ───────────────────────────────────────────────────
        with gr.TabItem("🏃 Round 1 — Sprint Manager"):

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

            gr.Markdown("### 🤖 Run Trained LLM Agent")
            with gr.Row():
                r1_agent_btn = gr.Button("▶️ Run LLM Agent (sejal-k/multi-sprint-model)",
                                          variant="primary", scale=1)
                r1_agent_log = gr.Textbox(
                    label="🤖 Agent Log  —  [LLM] = trained model acted  |  [FB] = fallback fired",
                    lines=20, interactive=False, scale=3,
                    value=(
                        "Click ▶️ to run the fine-tuned Qwen2.5-1.5B model.\n\n"
                        "Each line is tagged [LLM] (trained model action) or [FB] (fallback).\n"
                        "The ATTRIBUTION SUMMARY at the end shows what % of steps the model\n"
                        "actually drove vs the rule-based fallback — so you can judge the score.\n\n"
                        "Set LOCAL_MODEL_PATH env var to override the model path.\n"
                        "Requires HF_TOKEN to download from Hub on first run."
                    )
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

            R1_OUT       = [r1_board, r1_dev, r1_skill, r1_elog,      r1_metr, r1_rchart, r1_tchart, r1_sess]
            R1_AGENT_OUT = [r1_board, r1_dev, r1_skill, r1_agent_log, r1_metr, r1_rchart, r1_tchart, r1_sess]

            r1_reset_btn.click(fn=r1_reset_env,         inputs=[r1_task_sel, r1_sess],                           outputs=R1_OUT)
            r1_auto_btn.click( fn=r1_auto_assign,       inputs=[r1_sess],                                        outputs=R1_OUT)
            r1_act.click(      fn=r1_take_action,       inputs=[r1_at, r1_tid, r1_did, r1_pri, r1_sess],         outputs=R1_OUT)
            r1_agent_btn.click(fn=r1_run_trained_agent, inputs=[r1_task_sel, r1_sess],                           outputs=R1_AGENT_OUT)

        # ── TAB 2 — ROUND 2 ───────────────────────────────────────────────────
        with gr.TabItem("🚀 Round 2 — Project Manager"):

            r2_sess = gr.State(_new_r2_session)   # per-user, isolated

            gr.Markdown("""
            ### Long-Horizon Sprint Planning — 6 Sprints · 60 Days · Adaptive Instructions
            Instructions drip-feed over time. Missed tasks become **tech debt** that slows the team.
            Cascade failures cross sprint boundaries. Score = delivery × instruction-following × team health.
            """)

            with gr.Row():
                r2_task_sel   = gr.Dropdown(choices=VALID_PROJECT_TASK_NAMES, value="project_easy",
                                             label="🎯 Project Scenario", scale=2)
                r2_reset_btn  = gr.Button("🔄 Reset Project",     variant="primary",   scale=1)
                r2_auto_btn   = gr.Button("🤖 Auto-Assign Sprint", variant="secondary", scale=1)
                r2_adv_btn    = gr.Button("⏩ Advance Day",         variant="secondary", scale=1)

            with gr.Row():
                with gr.Column(scale=2):
                    r2_timeline = gr.Textbox(label="🗓️ Sprint Timeline", lines=16, interactive=False,
                                              value="👆 Select a project scenario and click Reset Project to begin!")
                with gr.Column(scale=3):
                    r2_board = gr.Textbox(label="📋 Current Sprint Board", lines=16, interactive=False)
                with gr.Column(scale=2):
                    r2_devs  = gr.Textbox(label="👥 Team Workload",        lines=16, interactive=False)

            with gr.Row():
                r2_inst  = gr.Textbox(label="📋 Instruction Queue", lines=12, interactive=False, scale=2)
                r2_debt  = gr.Textbox(label="🔴 Tech Debt Tracker", lines=12, interactive=False, scale=1)
                r2_metr  = gr.Textbox(label="📊 Project Metrics",   lines=12, interactive=False, scale=1)

            r2_rchart = gr.Textbox(label="📈 Cross-Sprint Reward Chart", lines=12, interactive=False)

            gr.Markdown("### 🤖 Run Trained LLM Agent (Round 2)")
            with gr.Row():
                r2_agent_btn = gr.Button("▶️ Run LLM Agent (60-day project)", variant="primary", scale=1)
                r2_agent_log = gr.Textbox(
                    label="🤖 R2 Agent Log  —  [LLM] = model acted  |  [FB] = fallback  |  attribution at end",
                    lines=20, interactive=False, scale=3,
                    value=(
                        "Click ▶️ to run the trained model on the full 60-day project.\n\n"
                        "Format: [LLM/FB] D{day}|S{sprint}: {action} {task}  r={reward}  inst={score}  debt={n}\n"
                        "Fallback tags: ←guard (invalid action), ←parse (bad JSON), ←rule (no model)\n\n"
                        "ATTRIBUTION SUMMARY at the end shows the LLM valid rate —\n"
                        "the only reliable way to know if the score came from the model."
                    )
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
            | `assign` | Assign backlog task to dev | Task=T01, Dev=dev1 |
            | `reassign` | Move task to another dev | Task=T05, Dev=dev3 |
            | `reprioritize` | Change task priority | Task=T08, Priority=1 |
            | `unblock` | Clear a blocked task | Task=T03 |
            | `skip` | Advance 1 day (releases instructions) | — |
            | `sprint_plan` | **R2 new** — batch plan for sprint | Task IDs=T09,T10,T11 |

            **Tip:** Check the Instruction Queue and act on flagged tasks for bonus rewards.
            Tech debt from missed tasks reduces team productivity in future sprints.
            """)

            R2_OUT       = [r2_timeline, r2_board, r2_devs, r2_inst, r2_debt, r2_metr, r2_rchart, r2_elog,      r2_sess]
            R2_AGENT_OUT = [r2_timeline, r2_board, r2_devs, r2_inst, r2_debt, r2_metr, r2_rchart, r2_agent_log, r2_sess]

            r2_reset_btn.click(fn=r2_reset_project,     inputs=[r2_task_sel, r2_sess],                                    outputs=R2_OUT)
            r2_auto_btn.click( fn=r2_auto_sprint,       inputs=[r2_sess],                                                 outputs=R2_OUT)
            r2_adv_btn.click(  fn=r2_advance_day,       inputs=[r2_sess],                                                 outputs=R2_OUT)
            r2_act.click(      fn=r2_take_action,       inputs=[r2_at, r2_tid, r2_did, r2_pri, r2_tids, r2_sess],         outputs=R2_OUT)
            r2_agent_btn.click(fn=r2_run_trained_agent, inputs=[r2_task_sel, r2_sess],                                    outputs=R2_AGENT_OUT)

app = gr.mount_gradio_app(api, demo, path="/")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)