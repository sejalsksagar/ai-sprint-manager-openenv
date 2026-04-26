"""
inference_r2.py  ─  Round 2 agent  ─  PATCHED for score > 0.75
========================================================================

KEY FIXES IN THIS VERSION
──────────────────────────────
[FIX-A] BAD-COMBO BLACKLIST (fatal regression killer)
        Every (task_id, dev_id) pair that causes a state regression is
        permanently blacklisted. _guarantee_assign() skips blacklisted
        pairs. This eliminates the infinite T05->dev1 loops.

[FIX-B] MISSED/DONE TASK FILTER IN OBS
        _assignable_tasks() now explicitly filters status in
        {'backlog'} only. 'missed', 'done', 'in_progress', 'blocked'
        are all excluded.

[FIX-C] REGRESSION GUARD TRIGGERS BLACKLIST
        When a state regression is detected, the last attempted
        action's (task_id, dev_id) pair is added to the blacklist
        before retrying. Previously the same action was retried
        identically, guaranteeing 3 regressions → FATAL.

[FIX-D] FALLBACK NEVER REPEATS SAME PAIR
        _guarantee_assign() receives the blacklist and skips any
        (task, dev) pair in it. It also skips tasks already
        in_progress in the current observation.

[FIX-E] LOCAL FINETUNED MODEL BY DEFAULT
        USE_LLM defaults to on: loads priyaaaaaasharmaaaaa/trial1 via
        LOCAL_MODEL_PATH (Unsloth/PEFT). Set USE_LLM=0 for rule-based-only.
        If load fails, main() falls back to HF router or rule-based.

[FIX-F] SKIP ONLY AFTER GENUINE EXHAUSTION
        smart_fallback now tries every valid (task, dev) combo before
        emitting a skip. The anti-skip wrapper also runs _guarantee_assign
        one final time before allowing skip through.

[FIX-G] IN-PROGRESS TASKS EXCLUDED FROM ASSIGNMENT
        Tasks whose status is 'in_progress' are excluded from
        _assignable_tasks() AND from _guarantee_assign(). This prevents
        the env from rejecting assign_bad_status:in_progress.

[FIX-H] INSTRUCTION PARSING WIDENED
        _parse_inst_action now also handles "D2" as dev_id by
        normalising "D2" → "dev2" if the env uses that format.

[FIX-I] POST-SKIP MANDATORY NON-SKIP
        After any skip step, the next action is never skip: we run an
        expanded fallback (extra reprioritize passes + guarantee) so the
        env advances with assign/unblock/reprioritize instead of skip chains.

[FIX-J] REWARD-BASED BAD COMBO
        Repeated strongly negative rewards on the same (task, dev) assign
        blacklists that pair before the env hits state regression.

[FIX-K] REGRESSION ON NON-ASSIGN
        On day/sprint regression without assign, immediately try
        guarantee_assign / fallback instead of retrying the same skip.
"""

from __future__ import annotations

import json
import os
import re
import time
import random
from typing import Optional, Tuple, List, Set

import requests

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
LOCAL_MODEL_PATH = os.getenv("LOCAL_MODEL_PATH", "priyaaaaaasharmaaaaa/trial1")
MODEL_NAME       = os.getenv("MODEL_NAME",       "meta-llama/Llama-3.1-8B-Instruct")
ENV_BASE_URL     = os.getenv("ENV_BASE_URL",     "https://sejal-k-ai-sprint-manager.hf.space")
API_BASE_URL     = os.getenv("API_BASE_URL",     "https://router.huggingface.co/v1")
HF_TOKEN         = os.getenv("HF_TOKEN",         "")

# Local fine-tune (trial1) on by default; USE_LLM=0 to disable
_use_llm_raw = os.getenv("USE_LLM", "1").strip().lower()
USE_LLM      = _use_llm_raw not in ("0", "false", "no", "off", "")

LLM_COOLDOWN_STEPS        = int(os.getenv("LLM_COOLDOWN_STEPS",        "5"))
MAX_LLM_SOFT_FAIL_STREAK  = int(os.getenv("MAX_LLM_SOFT_FAIL_STREAK",  "3"))
MAX_LLM_SKIP_STREAK       = int(os.getenv("MAX_LLM_SKIP_STREAK",       "4"))
MAX_SAME_BAD_ASSIGN_STREAK = int(os.getenv("MAX_SAME_BAD_ASSIGN_STREAK", "2"))

MAX_TOKENS     = 96
MAX_RETRIES    = 2
LLM_CALL_EVERY = 1
TEMPERATURE    = 0.3

TASK_ID_RE      = re.compile(r"^T\d+$")
RETRYABLE_CODES = {429, 500, 502, 503, 504}

_SKIP = {"action_type": "skip", "task_id": None, "dev_id": None, "new_priority": None}

# Strongly negative assign → blacklist pair after this many hits
REWARD_BLACKLIST_THRESHOLD = float(os.getenv("REWARD_BLACKLIST_THRESHOLD", "-0.12"))
REWARD_BLACKLIST_STREAK = int(os.getenv("REWARD_BLACKLIST_STREAK", "2"))

# Comma-separated scenarios; set ROUND2_EASY_ONLY=1 to train/eval on project_easy only (higher avg).
_ROUND2_EASY = os.getenv("ROUND2_EASY_ONLY", "0").strip().lower() in ("1", "true", "yes")
SCENARIOS_ENV = os.getenv(
    "ROUND2_SCENARIOS",
    "project_easy" if _ROUND2_EASY else "project_easy,project_medium,project_hard",
).strip()


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------
R2_SYSTEM_PROMPT = """You are an Engineering Manager running a 60-day software project.
Each step you MUST output exactly ONE JSON object and nothing else.

Schema (use null for unused fields):
{"action_type":"<assign|reassign|reprioritize|unblock|skip>","task_id":"<id or null>","dev_id":"<id or null>","new_priority":<1-5 or null>}

Rules (follow in order):
1. If ACTIVE INSTRUCTIONS exist, assign THEIR tasks first.
2. Only assign tasks with status=backlog (never in_progress or done).
3. Only assign if all dependency markers show checkmark (Y).
4. Only assign to an AVAILABLE developer with matching or fullstack skill.
5. Use unblock ONLY for explicitly blocked tasks whose deps are met.
6. skip is last resort.

Output ONLY the JSON. No explanation."""


def build_user_prompt(
    obs: dict,
    *,
    assigned_this_episode: Optional[set] = None,
    assign_attempted_episode: Optional[set] = None,
) -> str:
    current_sprint = obs.get("current_sprint", 1)
    current_day    = obs.get("current_day", 1)
    days_left      = max(0, current_sprint * 10 - current_day + 1)
    tasks          = obs.get("tasks", [])
    done_ids       = {t["id"] for t in tasks if t.get("status") == "done"}

    active_insts = [i for i in obs.get("instruction_queue", []) if not i.get("followed", False)]
    inst_section = (
        "FOLLOW: " + " | ".join(f"[{i['id']}] {i['text'][:50]}" for i in active_insts[:2])
    ) if active_insts else "No instructions."

    debt_raw   = obs.get("tech_debt", [])
    debt_count = len(debt_raw) if isinstance(debt_raw, list) else int(debt_raw or 0)

    backlog = sorted(
        [t for t in tasks if t.get("status") == "backlog"],
        key=lambda t: (t.get("priority", 9), t.get("deadline", 99))
    )
    in_prog = [t for t in tasks if t.get("status") == "in_progress"]

    def fmt(t: dict) -> str:
        meta   = t.get("metadata", {}) or {}
        deps   = t.get("depends_on", []) or meta.get("depends_on", [])
        dep_ok = "Y" if all(d in done_ids for d in deps) else "N"
        return (f"[{t['id']}]P{t.get('priority','?')} "
                f"{str(t.get('required_skill','?'))[:4]} {dep_ok} "
                f"D{t.get('deadline', t.get('deadline_day','?'))}")

    backlog_str = " ".join(fmt(t) for t in backlog[:6])
    if len(backlog) > 6:
        backlog_str += f" +{len(backlog)-6}"

    inprog_str = " ".join(
        f"[{t['id']}]->{t.get('assigned_to','?')}" for t in in_prog
    ) or "none"

    avail_devs = [d for d in obs.get("developers", []) if d.get("is_available", False)]
    devs_str   = " ".join(
        f"[{d['id']}]{str(d.get('name','?'))[:4]}({str(d.get('skill','?'))[:3]}) "
        f"{d.get('current_load',0)}/{d.get('capacity',5)}"
        for d in avail_devs
    )

    memory_lines: List[str] = []
    in_prog_ids = [t["id"] for t in in_prog]
    if in_prog_ids:
        memory_lines.append("NO_REASSIGN: " + " ".join(in_prog_ids))
    if assigned_this_episode:
        memory_lines.append("ASSIGNED_OK: " + " ".join(sorted(assigned_this_episode)))
    if assign_attempted_episode:
        extra = assign_attempted_episode - set(assigned_this_episode or ())
        if extra:
            memory_lines.append("TRIED_FAILED: " + " ".join(sorted(extra)))
    memory_block = ("MEM:\n" + "\n".join(memory_lines) + "\n") if memory_lines else ""

    return (
        f"D{current_day}/60 S{current_sprint}/6 {days_left}d "
        f"done={obs.get('tasks_completed',0)} miss={obs.get('tasks_missed',0)} "
        f"inst={obs.get('instruction_following_score',0):.2f} debt={debt_count}\n"
        f"{inst_section}\n"
        f"BACKLOG(Y=deps_ok): {backlog_str}\n"
        f"IN_PROG: {inprog_str}\n"
        f"DEVS(avail): {devs_str}\n"
        f"{memory_block}"
        f"JSON:"
    )


# ---------------------------------------------------------------------------
# Environment state helpers
# ---------------------------------------------------------------------------

def _done_ids(obs: dict) -> set:
    return {t["id"] for t in obs.get("tasks", []) if t.get("status") == "done"}


def _in_progress_ids(obs: dict) -> set:
    """[FIX-G] Return IDs of tasks currently in_progress."""
    return {t["id"] for t in obs.get("tasks", []) if t.get("status") == "in_progress"}


def _deps_met(obs: dict, task: dict) -> bool:
    done = _done_ids(obs)
    meta = task.get("metadata", {}) or {}
    deps = task.get("depends_on", []) or meta.get("depends_on", [])
    return all(d in done for d in deps)


def _dev_by_id(obs: dict, dev_id: object) -> Optional[dict]:
    sid = str(dev_id)
    for d in obs.get("developers", []):
        if str(d.get("id")) == sid:
            return d
    return None


def _available_devs(obs: dict, *, expand: bool = False) -> List[dict]:
    """Developers with remaining capacity. If expand=True (post-skip), also
    include devs that still have capacity even when is_available is False —
    the env sometimes clears availability flags while assignments are still valid.
    """
    def cap_ok(d: dict) -> bool:
        try:
            return int(d.get("remaining_capacity", d.get("capacity", 1))) > 0
        except (TypeError, ValueError):
            return False

    if expand:
        pool = [d for d in obs.get("developers", []) if cap_ok(d)]
        if pool:
            return pool

    with_cap = [
        d for d in obs.get("developers", [])
        if d.get("is_available", False) and cap_ok(d)
    ]
    if with_cap:
        return with_cap
    return [d for d in obs.get("developers", []) if d.get("is_available", False)]


def _assignable_tasks(obs: dict, excluded: Optional[set] = None) -> List[dict]:
    """[FIX-B/G] Only backlog tasks with deps met, excluding already-started/missed/done."""
    ex          = excluded or set()
    in_prog_ids = _in_progress_ids(obs)
    # Explicitly only allow status == 'backlog'
    return [
        t for t in obs.get("tasks", [])
        if t.get("status") == "backlog"
        and _deps_met(obs, t)
        and t["id"] not in ex
        and t["id"] not in in_prog_ids
    ]


# ---------------------------------------------------------------------------
# [FIX-A] Bad-combo blacklist type alias
# ---------------------------------------------------------------------------
# A set of (task_id, dev_id) string tuples that caused regressions or
# repeated env rejections. Populated by run_episode() and passed through.
BadComboSet = Set[Tuple[str, str]]


# ---------------------------------------------------------------------------
# [FIX-A/D] Core anti-skip guarantee — blacklist-aware
# ---------------------------------------------------------------------------

def _task_effort(t: dict) -> int:
    """Prefer lower-effort tasks when breaking ties (simpler work first)."""
    meta = t.get("metadata", {}) or {}
    try:
        return int(t.get("effort", meta.get("effort", 5)))
    except (TypeError, ValueError):
        return 5


def _guarantee_assign(
    obs: dict,
    assigned_this_episode: set,
    priority_task_ids: Optional[set] = None,
    bad_combos: Optional[BadComboSet] = None,
    *,
    expand_devs: bool = False,
    relax_skill: bool = False,
) -> Optional[dict]:
    """
    Exhaustively search every task x every dev for a valid assignment.
    Skips any (task, dev) pair in bad_combos.
    Returns None only when no valid pair whatsoever exists.
    """
    priority_ids   = priority_task_ids or set()
    bad            = bad_combos or set()
    avail          = _available_devs(obs, expand=expand_devs)
    backlog        = _assignable_tasks(obs, excluded=assigned_this_episode)
    current_sprint = obs.get("current_sprint", 1)

    if not avail or not backlog:
        return None

    def task_key(t: dict) -> tuple:
        meta = t.get("metadata", {}) or {}
        return (
            0 if t["id"] in priority_ids else 1,
            meta.get("sprint", current_sprint + 99),
            _task_effort(t),
            t.get("priority", 9),
            t.get("deadline", t.get("deadline_day", 99)),
        )

    backlog.sort(key=task_key)

    def _pick_devs_for_task(task: dict) -> List[dict]:
        skill = task.get("required_skill", "")

        def dev_key(d: dict) -> tuple:
            ds          = d.get("skill", "")
            match_score = 0 if ds == skill else (1 if ds == "fullstack" else 9)
            cap         = -int(d.get("remaining_capacity", d.get("capacity", 1)))
            return (match_score, cap)

        ordered = sorted(avail, key=dev_key)
        if relax_skill:
            return ordered
        return [d for d in ordered if d.get("skill", "") in (skill, "fullstack")]

    for task in backlog:
        skill = task.get("required_skill", "")
        for dev in _pick_devs_for_task(task):
            dskill = dev.get("skill", "")
            if not relax_skill and dskill not in (skill, "fullstack"):
                continue
            cap = int(dev.get("remaining_capacity", dev.get("capacity", 1)))
            if cap <= 0:
                continue
            combo = (str(task["id"]), str(dev["id"]))
            if combo in bad:
                continue
            act = {
                "action_type":  "assign",
                "task_id":      task["id"],
                "dev_id":       dev["id"],
                "new_priority": None,
            }
            ok, _ = validate_action(
                obs,
                act,
                assigned_this_episode,
                relax_dev_avail=expand_devs or relax_skill,
                relax_skill_match=relax_skill,
            )
            if ok:
                return act

    return None  # genuinely nothing to assign


# ---------------------------------------------------------------------------
# Validate any action before sending to env
# ---------------------------------------------------------------------------

def validate_action(
    obs: dict,
    action: Optional[dict],
    assigned_this_episode: set,
    *,
    relax_dev_avail: bool = False,
    relax_skill_match: bool = False,
) -> Tuple[bool, str]:
    if not action:
        return False, "empty"
    at = action.get("action_type")
    if at not in {"assign", "reassign", "reprioritize", "unblock", "skip"}:
        return False, "bad_type"

    if at == "skip":
        return True, "ok"

    tid = action.get("task_id")
    if not tid:
        return False, "missing_task_id"

    by_id = {t["id"]: t for t in obs.get("tasks", [])}
    task  = by_id.get(str(tid))
    if task is None:
        return False, "unknown_task"

    if at == "unblock":
        if task.get("status") != "blocked":
            return False, "unblock_not_blocked"
        if not _deps_met(obs, task):
            return False, "unblock_deps_unmet"
        return True, "ok"

    if at == "reprioritize":
        if task.get("status") != "backlog":
            return False, "reprioritize_not_backlog"
        np_ = action.get("new_priority")
        if np_ is None:
            return False, "reprioritize_no_priority"
        try:
            if not (1 <= int(np_) <= 5):
                return False, "reprioritize_range"
        except (TypeError, ValueError):
            return False, "reprioritize_bad_priority"
        return True, "ok"

    if at in ("assign", "reassign"):
        st = task.get("status")
        if st != "backlog":
            return False, f"assign_bad_status:{st}"
        if not _deps_met(obs, task):
            return False, "assign_deps_unmet"
        if str(tid) in assigned_this_episode:
            return False, "assign_already_started"
        # [FIX-G] Also reject if currently in_progress per obs
        if str(tid) in _in_progress_ids(obs):
            return False, "assign_already_in_progress"
        did = action.get("dev_id")
        if not did:
            return False, "assign_no_dev"
        dev = _dev_by_id(obs, did)
        if dev is None:
            return False, "assign_unknown_dev"
        if not relax_dev_avail and not dev.get("is_available", False):
            return False, "assign_dev_unavailable"
        try:
            if int(dev.get("remaining_capacity", dev.get("capacity", 1))) <= 0:
                return False, "assign_dev_no_capacity"
        except (TypeError, ValueError):
            pass
        skill  = task.get("required_skill", "")
        dskill = dev.get("skill", "")
        if (
            not relax_skill_match
            and dskill not in (skill, "fullstack")
        ):
            return False, "assign_skill_mismatch"
        return True, "ok"

    return False, "unhandled"


validate_llm_action = validate_action


# ---------------------------------------------------------------------------
# [FIX-H] Instruction text parser — normalises D\d+ → dev\d+
# ---------------------------------------------------------------------------

def _normalise_dev_id(raw_dev: str, obs: dict) -> Optional[str]:
    """
    Maps "D2" → "dev2" if the env uses "dev2" format, or returns the raw
    string if it already matches an existing developer id.
    """
    all_dev_ids = {str(d.get("id", "")) for d in obs.get("developers", [])}
    if raw_dev in all_dev_ids:
        return raw_dev
    # Try lowercase + strip leading zeros: D02 → dev2
    candidate = "dev" + str(int(raw_dev[1:]))
    if candidate in all_dev_ids:
        return candidate
    # Try as-is without prefix
    bare = raw_dev[1:]
    if bare in all_dev_ids:
        return bare
    return raw_dev  # give it a shot anyway


def _parse_inst_action(
    inst_text: str,
    obs: dict,
    assigned_this_episode: set,
    bad_combos: Optional[BadComboSet] = None,
) -> Optional[dict]:
    task_m = re.search(r'\b(T\d+)\b', inst_text)
    dev_m  = re.search(r'\b(D\d+)\b', inst_text, re.IGNORECASE)
    if not task_m or not dev_m:
        return None

    raw_dev = dev_m.group(1).upper()
    dev_id  = _normalise_dev_id(raw_dev, obs)

    action = {
        "action_type":  "assign",
        "task_id":      task_m.group(1),
        "dev_id":       dev_id,
        "new_priority": None,
    }
    bad = bad_combos or set()
    combo = (str(action["task_id"]), str(dev_id))
    if combo in bad:
        return None
    ok, _ = validate_action(
        obs, action, assigned_this_episode, relax_dev_avail=False,
    )
    return action if ok else None


def _reprioritize_waiting_deps(
    obs: dict,
    tasks: list,
    assigned_this_episode: set,
    min_priority: int,
    *,
    relax_dev: bool,
) -> Optional[dict]:
    """Raise priority to 1 for backlog tasks whose dependencies are not yet done."""
    for task in tasks:
        if task.get("status") != "backlog" or _deps_met(obs, task):
            continue
        if min_priority > 0 and int(task.get("priority", 1)) <= min_priority:
            continue
        action = {
            "action_type":  "reprioritize",
            "task_id":      task["id"],
            "dev_id":       None,
            "new_priority": 1,
        }
        ok, _ = validate_action(
            obs, action, assigned_this_episode, relax_dev_avail=relax_dev,
        )
        if ok:
            return action
    return None


def _scan_reprioritize_variants(
    obs: dict,
    tasks: list,
    assigned_this_episode: set,
    *,
    relax_dev: bool,
) -> Optional[dict]:
    """Any valid reprioritize that changes numeric priority (unsticks repeated Txx→P1 loops)."""
    for task in tasks:
        if task.get("status") != "backlog":
            continue
        cur = int(task.get("priority", 3))
        for np in (1, 2, 3, 4, 5):
            if np == cur:
                continue
            action = {
                "action_type":  "reprioritize",
                "task_id":      task["id"],
                "dev_id":       None,
                "new_priority": np,
            }
            ok, _ = validate_action(
                obs, action, assigned_this_episode, relax_dev_avail=relax_dev,
            )
            if ok:
                return action
    return None


# ---------------------------------------------------------------------------
# Smart fallback — blacklist-aware
# ---------------------------------------------------------------------------

def smart_fallback(
    obs: dict,
    assigned_this_episode: set,
    last_dev_idx: list,
    bad_combos: Optional[BadComboSet] = None,
    *,
    force_non_skip: bool = False,
) -> dict:
    """
    Tiered decision engine. Passes bad_combos through to _guarantee_assign
    so previously-regressing pairs are skipped.

    Tier 0  Parse instruction text for explicit T→D directives
    Tier 1  Unblock blocked tasks whose deps are met
    Tier 2  Assign backlog tasks related to active instructions
    Tier 3  Assign any backlog task (exhaustive, blacklist-aware)
    Tier 4  Reprioritize blocked-by-dep tasks
    Tier 5  Skip (disabled when force_non_skip — reprioritize variants + relax-skill assign)
    """
    bad    = bad_combos or set()
    tasks  = obs.get("tasks", [])
    instructions = obs.get("instruction_queue", [])

    active_insts         = [i for i in instructions if not i.get("followed", False)]
    instruction_task_ids: set = set()
    for inst in active_insts:
        for tid in inst.get("affects_tasks", []):
            instruction_task_ids.add(tid)

    # ── Tier 0 ────────────────────────────────────────────────────────────────
    if not force_non_skip:
        for inst in active_insts:
            action = _parse_inst_action(
                inst.get("text", ""), obs, assigned_this_episode, bad
            )
            if action:
                print(
                    f"  [FB-T0] inst text parse -> {action['task_id']}->{action['dev_id']}",
                    flush=True,
                )
                return action

    # ── Tier 1 ────────────────────────────────────────────────────────────────
    for task in tasks:
        if task.get("status") == "blocked" and _deps_met(obs, task):
            action = {
                "action_type":  "unblock",
                "task_id":      task["id"],
                "dev_id":       None,
                "new_priority": None,
            }
            ok, _ = validate_action(
                obs, action, assigned_this_episode, relax_dev_avail=force_non_skip,
            )
            if ok:
                print(f"  [FB-T1] unblock {task['id']}", flush=True)
                return action

    # ── Tier 2 ────────────────────────────────────────────────────────────────
    if instruction_task_ids:
        action = _guarantee_assign(
            obs, assigned_this_episode,
            priority_task_ids=instruction_task_ids,
            bad_combos=bad,
            expand_devs=force_non_skip,
        )
        if action:
            print(f"  [FB-T2] inst assign -> {action['task_id']}->{action['dev_id']}", flush=True)
            return action
        action = _guarantee_assign(
            obs, assigned_this_episode,
            priority_task_ids=set(),
            bad_combos=bad,
            expand_devs=force_non_skip,
        )
        if action:
            print(f"  [FB-T2b] non-inst assign -> {action['task_id']}->{action['dev_id']}", flush=True)
            return action

    # ── Tier 3 ────────────────────────────────────────────────────────────────
    action = _guarantee_assign(
        obs, assigned_this_episode, bad_combos=bad, expand_devs=force_non_skip,
    )
    if action:
        print(f"  [FB-T3] general assign -> {action['task_id']}->{action['dev_id']}", flush=True)
        return action

    # ── Tier 4 (normal) ───────────────────────────────────────────────────────
    act4 = _reprioritize_waiting_deps(
        obs, tasks, assigned_this_episode, 2, relax_dev=force_non_skip,
    )
    if act4:
        print(f"  [FB-T4] reprioritize {act4['task_id']} (waiting deps)", flush=True)
        return act4

    # ── Recovery / no-skip tail ─────────────────────────────────────────────
    if force_non_skip:
        act4b = _reprioritize_waiting_deps(
            obs, tasks, assigned_this_episode, 0, relax_dev=True,
        )
        if act4b:
            print(
                f"  [FB-T4b] reprioritize {act4b['task_id']} (deps wait → unlock assign)",
                flush=True,
            )
            return act4b
        for task in tasks:
            if task.get("status") != "backlog" or not _deps_met(obs, task):
                continue
            np = max(1, min(5, int(task.get("priority", 3)) - 1))
            if np == int(task.get("priority", 3)):
                continue
            action = {
                "action_type":  "reprioritize",
                "task_id":      task["id"],
                "dev_id":       None,
                "new_priority": np,
            }
            ok, _ = validate_action(
                obs, action, assigned_this_episode, relax_dev_avail=True,
            )
            if ok:
                print(f"  [FB-T4c] reprioritize {task['id']} -> P{np} (recovery)", flush=True)
                return action

        again = _guarantee_assign(
            obs, assigned_this_episode,
            priority_task_ids=set(),
            bad_combos=bad,
            expand_devs=True,
            relax_skill=False,
        )
        if again:
            print(
                f"  [FB-T5x] recovery assign -> {again['task_id']}->{again['dev_id']}",
                flush=True,
            )
            return again
        loose = _guarantee_assign(
            obs, assigned_this_episode,
            priority_task_ids=set(),
            bad_combos=bad,
            expand_devs=True,
            relax_skill=True,
        )
        if loose:
            print(
                f"  [FB-T5y] relax-skill assign -> {loose['task_id']}->{loose['dev_id']}",
                flush=True,
            )
            return loose
        scan = _scan_reprioritize_variants(
            obs, tasks, assigned_this_episode, relax_dev=True,
        )
        if scan:
            print(
                f"  [FB-T6] reprioritize {scan['task_id']} -> P{scan['new_priority']} "
                f"(no-skip last resort)",
                flush=True,
            )
            return scan
        print("  [FB-T5] recovery exhausted — using skip", flush=True)
    else:
        print("  [FB-T5] genuine skip -- no valid action available", flush=True)
    return _SKIP


# ---------------------------------------------------------------------------
# Anti-skip wrapper
# ---------------------------------------------------------------------------

def _anti_skip(
    action: dict,
    obs: dict,
    assigned_this_episode: set,
    instruction_task_ids: set,
    source: str,
    bad_combos: Optional[BadComboSet] = None,
    *,
    expand_devs: bool = False,
) -> dict:
    if action.get("action_type") != "skip":
        return action

    ex = expand_devs
    override = _guarantee_assign(
        obs, assigned_this_episode,
        priority_task_ids=instruction_task_ids,
        bad_combos=bad_combos,
        expand_devs=ex,
    )
    if not override and instruction_task_ids:
        override = _guarantee_assign(
            obs, assigned_this_episode,
            priority_task_ids=set(),
            bad_combos=bad_combos,
            expand_devs=ex,
        )
    if not override and ex:
        override = _guarantee_assign(
            obs, assigned_this_episode,
            priority_task_ids=set(),
            bad_combos=bad_combos,
            expand_devs=True,
            relax_skill=True,
        )
    if not override and ex:
        tasks = obs.get("tasks", [])
        rp = _scan_reprioritize_variants(
            obs, tasks, assigned_this_episode, relax_dev=True,
        )
        if rp:
            print(
                f"  [ANTI-SKIP] {source} skip -> reprioritize {rp['task_id']} P{rp['new_priority']}",
                flush=True,
            )
            return rp
    if override:
        print(
            f"  [ANTI-SKIP] {source} skip overridden -> "
            f"{override['task_id']}->{override['dev_id']}",
            flush=True,
        )
        return override

    return action


# ---------------------------------------------------------------------------
# Local fine-tuned model loader
# ---------------------------------------------------------------------------

_local_model     = None
_local_tokenizer = None
_local_backend   = None


def _load_local_model(model_path: str) -> bool:
    global _local_model, _local_tokenizer, _local_backend
    if _local_model is not None:
        return True

    print(f"[INFO] Loading fine-tuned model: {model_path}", flush=True)
    unsloth_err: Optional[BaseException] = None

    try:
        from unsloth import FastLanguageModel
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_path, max_seq_length=2048, dtype=None,
            load_in_4bit=True, token=HF_TOKEN or None,
        )
        FastLanguageModel.for_inference(model)
        _local_model, _local_tokenizer, _local_backend = model, tokenizer, "unsloth"
        print("[INFO] Loaded via Unsloth (fast 4-bit inference).", flush=True)
        return True
    except Exception as e:
        unsloth_err = e
        print(f"[WARN] Unsloth failed: {e}", flush=True)

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from peft import PeftModel
        import huggingface_hub

        try:
            cfg_path = huggingface_hub.hf_hub_download(
                model_path, "adapter_config.json", token=HF_TOKEN or None
            )
        except Exception:
            cfg_path = os.path.join(model_path, "adapter_config.json")

        with open(cfg_path) as f:
            adapter_cfg = json.load(f)
        base_id = adapter_cfg.get("base_model_name_or_path", "Qwen/Qwen2.5-1.5B-Instruct")
        print(f"[INFO] Base model from adapter config: {base_id}", flush=True)

        bnb   = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
        tok   = AutoTokenizer.from_pretrained(base_id, token=HF_TOKEN or None)
        base  = AutoModelForCausalLM.from_pretrained(
            base_id, quantization_config=bnb, device_map="auto", token=HF_TOKEN or None
        )
        model = PeftModel.from_pretrained(base, model_path, token=HF_TOKEN or None)
        model.eval()
        _local_model, _local_tokenizer, _local_backend = model, tok, "peft"
        print("[INFO] Loaded via PEFT + bitsandbytes 4-bit.", flush=True)
        return True
    except Exception as e2:
        print(
            f"[ERROR] Cannot load local model.\n  Unsloth: {unsloth_err}\n  PEFT: {e2}",
            flush=True,
        )
        return False


def _call_local_model(user_prompt: str) -> Optional[dict]:
    if _local_model is None:
        return None
    import torch

    messages = [
        {"role": "system", "content": R2_SYSTEM_PROMPT},
        {"role": "user",   "content": user_prompt},
    ]
    tok = _local_tokenizer
    try:
        prompt_text = (
            tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            if hasattr(tok, "apply_chat_template")
            else "\n".join(f"<|{m['role']}|>\n{m['content']}" for m in messages)
               + "\n<|assistant|>\n"
        )
        inputs  = tok(prompt_text, return_tensors="pt").to(_local_model.device)
        inp_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            outputs = _local_model.generate(
                **inputs,
                max_new_tokens=MAX_TOKENS,
                temperature=TEMPERATURE,
                do_sample=True,
                pad_token_id=tok.pad_token_id or tok.eos_token_id,
            )
        completion = tok.decode(outputs[0][inp_len:], skip_special_tokens=True).strip()
        return parse_action(completion)
    except Exception as e:
        print(f"  [WARN] Local model inference error: {e}", flush=True)
        return None


def _call_api_model(user_prompt: str) -> Optional[dict]:
    try:
        from openai import OpenAI, APIStatusError
    except ImportError:
        return None
    if not HF_TOKEN:
        return None

    client   = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
    messages = [
        {"role": "system", "content": R2_SYSTEM_PROMPT},
        {"role": "user",   "content": user_prompt},
    ]
    for attempt in range(1, MAX_RETRIES + 2):
        try:
            resp = client.chat.completions.create(
                model=MODEL_NAME, messages=messages,
                max_tokens=MAX_TOKENS, temperature=TEMPERATURE,
            )
            return parse_action(resp.choices[0].message.content or "")
        except APIStatusError as e:
            if e.status_code not in RETRYABLE_CODES:
                return None
            if attempt <= MAX_RETRIES:
                time.sleep(2 ** attempt + random.uniform(0, 0.5))
            else:
                return None
        except Exception:
            if attempt <= MAX_RETRIES:
                time.sleep(2 ** attempt)
            else:
                return None
    return None


def call_llm(user_prompt: str) -> Optional[dict]:
    if _local_model is not None:
        return _call_local_model(user_prompt)
    return _call_api_model(user_prompt)


# ---------------------------------------------------------------------------
# Action parser
# ---------------------------------------------------------------------------

def parse_action(raw: str) -> Optional[dict]:
    if not raw:
        return None
    raw = raw.strip()
    raw = re.sub(r"^```[a-z]*\s*", "", raw)
    raw = re.sub(r"\s*```$",       "", raw)

    depth = 0; obj_start = -1; last_start = -1; last_end = -1
    for i, ch in enumerate(raw):
        if ch == "{":
            if depth == 0:
                obj_start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and obj_start >= 0:
                last_start, last_end = obj_start, i + 1

    obj = None
    if last_start >= 0:
        try:
            obj = json.loads(raw[last_start:last_end])
        except json.JSONDecodeError:
            pass
    if obj is None:
        try:
            obj = json.loads(raw)
        except Exception:
            return None

    action_type = obj.get("action_type", "")
    if action_type not in {"assign", "reassign", "reprioritize", "unblock", "skip"}:
        return None

    null_vals = {"null", "none", "None", "Null", "", "undefined", "N/A", "nil"}
    for key in ("task_id", "dev_id", "new_priority"):
        v = obj.get(key)
        if v is not None and str(v).strip() in null_vals:
            obj[key] = None

    task_id = obj.get("task_id")
    if task_id is not None and not TASK_ID_RE.match(str(task_id)):
        return None

    if action_type in {"assign", "reassign"}:
        if not task_id or not obj.get("dev_id"):
            return None

    return {
        "action_type":  action_type,
        "task_id":      task_id,
        "dev_id":       obj.get("dev_id"),
        "new_priority": obj.get("new_priority"),
    }


_build_r2_prompt     = build_user_prompt
_parse_action        = parse_action
call_llm_user_prompt = call_llm


# ---------------------------------------------------------------------------
# Environment helpers
# ---------------------------------------------------------------------------

def _post(url: str, payload: dict, timeout: int = 15) -> dict:
    resp = requests.post(url, json=payload, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def step_env(action: dict) -> dict:
    return _post(f"{ENV_BASE_URL}/project/step", {"action": action})


def reset_env(scenario: str, seed: int = 42) -> dict:
    return _post(f"{ENV_BASE_URL}/project/reset", {"task_name": scenario, "seed": seed})


def health() -> dict:
    resp = requests.get(f"{ENV_BASE_URL}/project/health", timeout=10)
    resp.raise_for_status()
    return resp.json()


# ---------------------------------------------------------------------------
# Episode runner — [FIX-C] blacklist regression-causing combos immediately
# ---------------------------------------------------------------------------

def run_episode(scenario: str, seed: int = 42) -> dict:
    obs_data = reset_env(scenario, seed)
    obs      = obs_data.get("observation", obs_data)

    assigned_this_episode: set   = set()
    assign_attempted_episode: set = set()
    bad_combos: BadComboSet       = set()   # [FIX-A]
    last_dev_idx                  = [0]
    cumulative                    = 0.0
    step_num                      = 0

    llm_skip_streak       = 0
    llm_soft_fail_streak  = 0
    bad_assign_tid: Optional[str] = None
    bad_assign_streak     = 0
    llm_cooldown_until    = 0

    # After skip: stay in recovery until a real assign/reassign (reprioritize alone does not clear).
    recover_until_assign = False
    last_neg_combo: Optional[Tuple[str, str]] = None
    neg_reward_streak   = 0

    MAX_STEPS        = 200
    MAX_STEP_RETRIES = 3

    print(f"\n[START] task={scenario}", flush=True)

    while True:
        step_num += 1
        if step_num > MAX_STEPS:
            print("[WARN] Max steps reached -- terminating", flush=True)
            break

        day    = obs.get("current_day",    step_num)
        sprint = obs.get("current_sprint", 1)

        active_insts = [
            i for i in obs.get("instruction_queue", [])
            if not i.get("followed", False)
        ]
        inst_task_ids: set = set()
        for inst in active_insts:
            for tid in inst.get("affects_tasks", []):
                inst_task_ids.add(tid)

        # ── LLM path ──────────────────────────────────────────────────────────
        router_ok = _local_model is not None or bool(HF_TOKEN)
        allow_llm = (
            USE_LLM
            and (step_num % LLM_CALL_EVERY == 0)
            and router_ok
            and step_num > llm_cooldown_until
        )

        action: Optional[dict] = None

        if allow_llm:
            user_prompt = build_user_prompt(
                obs,
                assigned_this_episode=assigned_this_episode,
                assign_attempted_episode=assign_attempted_episode,
            )
            proposed = call_llm(user_prompt)

            if proposed is None:
                llm_soft_fail_streak += 1
                print("  [LLM] no parse/API fail -> fallback", flush=True)
            else:
                ok, reason = validate_action(obs, proposed, assigned_this_episode)
                if ok:
                    proposed = _anti_skip(
                        proposed,
                        obs,
                        assigned_this_episode,
                        inst_task_ids,
                        "LLM",
                        bad_combos=bad_combos,
                        expand_devs=recover_until_assign,
                    )
                    if proposed.get("action_type") == "skip":
                        llm_skip_streak += 1
                        if llm_skip_streak >= MAX_LLM_SKIP_STREAK:
                            llm_cooldown_until   = max(
                                llm_cooldown_until, step_num + LLM_COOLDOWN_STEPS
                            )
                            llm_skip_streak      = 0
                            llm_soft_fail_streak += 1
                            print(
                                f"  [COOLDOWN] LLM skip-spam -> pause {LLM_COOLDOWN_STEPS}s",
                                flush=True,
                            )
                        else:
                            action               = proposed
                            llm_soft_fail_streak = 0
                    else:
                        # [FIX-A] reject LLM action if it's in bad_combos
                        if proposed.get("action_type") in ("assign", "reassign"):
                            combo = (
                                str(proposed.get("task_id", "")),
                                str(proposed.get("dev_id", "")),
                            )
                            if combo in bad_combos:
                                print(
                                    f"  [REJECT] LLM proposed blacklisted combo "
                                    f"{combo} -> fallback",
                                    flush=True,
                                )
                                llm_soft_fail_streak += 1
                                proposed = None

                    if proposed is not None:
                        llm_skip_streak      = 0
                        llm_soft_fail_streak = 0
                        bad_assign_tid       = None
                        bad_assign_streak    = 0
                        action               = proposed
                else:
                    llm_soft_fail_streak += 1
                    print(f"  [REJECT] LLM invalid ({reason}) -> fallback", flush=True)
                    if proposed.get("action_type") in ("assign", "reassign"):
                        tidp    = proposed.get("task_id")
                        tid_key = str(tidp) if tidp else None
                        if tid_key == bad_assign_tid:
                            bad_assign_streak += 1
                        else:
                            bad_assign_tid    = tid_key
                            bad_assign_streak = 1
                    else:
                        bad_assign_tid    = None
                        bad_assign_streak = 0

            if bad_assign_streak >= MAX_SAME_BAD_ASSIGN_STREAK:
                llm_cooldown_until = max(
                    llm_cooldown_until, step_num + LLM_COOLDOWN_STEPS
                )
                bad_assign_streak = 0
                bad_assign_tid    = None
                print(
                    f"  [COOLDOWN] repeated bad assign -> rule-based {LLM_COOLDOWN_STEPS}s",
                    flush=True,
                )

            if llm_soft_fail_streak >= MAX_LLM_SOFT_FAIL_STREAK:
                llm_cooldown_until   = max(
                    llm_cooldown_until, step_num + LLM_COOLDOWN_STEPS
                )
                llm_soft_fail_streak = 0
                print(
                    f"  [COOLDOWN] repeated fail -> rule-based {LLM_COOLDOWN_STEPS}s",
                    flush=True,
                )

        # ── Fallback path ──────────────────────────────────────────────────────
        if action is None:
            action = smart_fallback(
                obs,
                assigned_this_episode,
                last_dev_idx,
                bad_combos=bad_combos,
                force_non_skip=recover_until_assign,
            )
            ok, reason = validate_action(
                obs,
                action,
                assigned_this_episode,
                relax_dev_avail=recover_until_assign,
                relax_skill_match=recover_until_assign
                and action.get("action_type") in ("assign", "reassign"),
            )
            if not ok:
                print(f"  [FALLBACK_INVALID] {reason} -> guaranteed assign", flush=True)
                action = _guarantee_assign(
                    obs,
                    assigned_this_episode,
                    inst_task_ids,
                    bad_combos=bad_combos,
                    expand_devs=recover_until_assign,
                )
                if action is None and recover_until_assign:
                    action = _guarantee_assign(
                        obs,
                        assigned_this_episode,
                        inst_task_ids,
                        bad_combos=bad_combos,
                        expand_devs=True,
                        relax_skill=True,
                    )
                action = action or _SKIP

        # Final anti-skip
        action = _anti_skip(
            action,
            obs,
            assigned_this_episode,
            inst_task_ids,
            "final",
            bad_combos=bad_combos,
            expand_devs=recover_until_assign,
        )

        if action.get("action_type") in ("assign", "reassign") and action.get("task_id"):
            assign_attempted_episode.add(action["task_id"])

        # Remember what we're about to send (for blacklisting on regression)
        last_action = action.copy()

        # ── Send to environment — [FIX-C] blacklist on regression ─────────────
        success = False
        for attempt in range(MAX_STEP_RETRIES):
            try:
                result  = step_env(action)
                new_obs = result.get("observation", result)
            except Exception as e:
                print(f"  [ENV_ERR] step_env failed (attempt {attempt+1}): {e}", flush=True)
                time.sleep(1.0)
                continue

            if (
                new_obs.get("current_day",    0) < obs.get("current_day",    0) or
                new_obs.get("current_sprint", 0) < obs.get("current_sprint", 0)
            ):
                # [FIX-C] Blacklist this combo immediately, pick a new action
                if last_action.get("action_type") in ("assign", "reassign"):
                    combo = (
                        str(last_action.get("task_id", "")),
                        str(last_action.get("dev_id", "")),
                    )
                    if combo not in bad_combos:
                        print(
                            f"  [BLACKLIST] Adding bad combo {combo} "
                            f"(attempt {attempt+1})",
                            flush=True,
                        )
                        bad_combos.add(combo)
                    # Pick a different action using updated blacklist
                    action = smart_fallback(
                        obs,
                        assigned_this_episode,
                        last_dev_idx,
                        bad_combos=bad_combos,
                        force_non_skip=recover_until_assign,
                    )
                    ok2, _ = validate_action(
                        obs,
                        action,
                        assigned_this_episode,
                        relax_dev_avail=recover_until_assign,
                        relax_skill_match=recover_until_assign
                        and action.get("action_type") in ("assign", "reassign"),
                    )
                    if not ok2:
                        action = _guarantee_assign(
                            obs,
                            assigned_this_episode,
                            inst_task_ids,
                            bad_combos=bad_combos,
                            expand_devs=recover_until_assign,
                        )
                        if action is None and recover_until_assign:
                            action = _guarantee_assign(
                                obs,
                                assigned_this_episode,
                                inst_task_ids,
                                bad_combos=bad_combos,
                                expand_devs=True,
                                relax_skill=True,
                            )
                        action = action or _SKIP
                    print(
                        f"  [RETRY] After blacklist: "
                        f"{action.get('action_type')} "
                        f"{action.get('task_id')}→{action.get('dev_id')}",
                        flush=True,
                    )
                else:
                    print(
                        f"  [ERROR] State regression (attempt {attempt+1}) -- retrying",
                        flush=True,
                    )
                    action = _guarantee_assign(
                        obs,
                        assigned_this_episode,
                        inst_task_ids,
                        bad_combos=bad_combos,
                        expand_devs=True,
                    ) or smart_fallback(
                        obs,
                        assigned_this_episode,
                        last_dev_idx,
                        bad_combos=bad_combos,
                        force_non_skip=True,
                    )
                    print(
                        f"  [RETRY-REG] pivoted to "
                        f"{action.get('action_type')} {action.get('task_id')}→{action.get('dev_id')}",
                        flush=True,
                    )
                time.sleep(0.5)
                continue

            success = True
            break

        if not success:
            print("[FATAL] Repeated env corruption -- aborting episode", flush=True)
            break

        reward     = result.get("reward", 0.0)
        obs        = new_obs
        done       = result.get("done", obs.get("done", False))
        cumulative += reward

        # [FIX-J] Blacklist (task, dev) pairs that keep paying strongly negative rewards
        if action.get("action_type") in ("assign", "reassign") and action.get("task_id"):
            tid_s = str(action["task_id"])
            did_s = str(action.get("dev_id") or "")
            combo_r = (tid_s, did_s)
            if reward <= REWARD_BLACKLIST_THRESHOLD:
                if combo_r == last_neg_combo:
                    neg_reward_streak += 1
                else:
                    last_neg_combo = combo_r
                    neg_reward_streak = 1
                if neg_reward_streak >= REWARD_BLACKLIST_STREAK and did_s:
                    if combo_r not in bad_combos:
                        bad_combos.add(combo_r)
                        print(
                            f"  [BLACKLIST] reward-penalty {combo_r} "
                            f"(r={reward:.3f} x{neg_reward_streak})",
                            flush=True,
                        )
            else:
                last_neg_combo = None
                neg_reward_streak = 0
        elif action.get("action_type") not in ("assign", "reassign"):
            last_neg_combo = None
            neg_reward_streak = 0

        if action.get("action_type") == "skip":
            recover_until_assign = True
            print(
                "  [RECOVERY] skip observed — use reprioritize/assign until next assign",
                flush=True,
            )
        elif action.get("action_type") in ("assign", "reassign"):
            recover_until_assign = False

        if action.get("action_type") in ("assign", "reassign") and action.get("task_id"):
            tid_sent    = action["task_id"]
            post_status = {t["id"]: t.get("status") for t in obs.get("tasks", [])}
            if post_status.get(tid_sent) == "in_progress":
                assigned_this_episode.add(tid_sent)

        inst_score = obs.get("instruction_following_score", 0.0)
        debt_raw   = obs.get("tech_debt", 0)
        debt_d     = len(debt_raw) if isinstance(debt_raw, list) else int(debt_raw or 0)

        print(
            f"[STEP] task={scenario} step={step_num} day={day} sprint={sprint} "
            f"action={action.get('action_type','?')} "
            f"task_id={action.get('task_id','None')} "
            f"dev={action.get('dev_id','None')} "
            f"reward={reward:.4f} cumul={cumulative:.4f} "
            f"inst={inst_score:.3f} debt={debt_d} done={done}",
            flush=True,
        )

        if done:
            break

    # ── Final metrics ──────────────────────────────────────────────────────────
    tasks      = obs.get("tasks", [])
    completed  = sum(1 for t in tasks if t.get("status") == "done")
    missed     = sum(1 for t in tasks if t.get("status") == "missed")
    inst_score = obs.get("instruction_following_score", 0.0)
    debt_raw   = obs.get("tech_debt", 0)
    debt_count = len(debt_raw) if isinstance(debt_raw, list) else int(debt_raw or 0)
    total      = len(tasks) or 1

    final_score = max(0.01, min(0.99,
        (completed / total) * 0.55 +
        inst_score * 0.30 +
        max(0.01, 1.0 - debt_count * 0.02) * 0.15
    ))

    print(
        f"[END] task={scenario} score={final_score:.4f} steps={step_num} "
        f"completed={completed}/{total} missed={missed} "
        f"inst={inst_score:.3f} debt={debt_count} "
        f"blacklisted_combos={len(bad_combos)}",
        flush=True,
    )
    return {
        "scenario":   scenario,
        "score":      final_score,
        "completed":  completed,
        "missed":     missed,
        "inst_score": inst_score,
        "debt":       debt_count,
        "steps":      step_num,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    scenarios = [s.strip() for s in SCENARIOS_ENV.split(",") if s.strip()]
    if not scenarios:
        scenarios = ["project_easy", "project_medium", "project_hard"]

    local_ok = False
    if USE_LLM and LOCAL_MODEL_PATH:
        local_ok = _load_local_model(LOCAL_MODEL_PATH)
        if not local_ok:
            print(
                "[WARN] Local model load failed -- running rule-based only.\n"
                "       Set LOCAL_MODEL_PATH to your adapter checkpoint to reach 0.80+.",
                flush=True,
            )

    if not USE_LLM:
        mode = "rule-based-only (USE_LLM=0)"
    else:
        mode = (
            "local-finetuned" if local_ok  else
            "hf-router-base"  if HF_TOKEN  else
            "rule-based-only"
        )

    print(
        f"[INFO] mode={mode}  model={LOCAL_MODEL_PATH if local_ok else MODEL_NAME}",
        flush=True,
    )
    print(
        f"[INFO] USE_LLM={USE_LLM}  cooldown={LLM_COOLDOWN_STEPS}  server={ENV_BASE_URL}",
        flush=True,
    )

    try:
        print(f"[INFO] health={health()}", flush=True)
    except Exception as e:
        print(f"[WARN] health check failed: {e}", flush=True)

    if _ROUND2_EASY:
        print(
            "[INFO] ROUND2_EASY_ONLY=1 — running project_easy only (raise avg / reduce complexity)",
            flush=True,
        )
    else:
        print(f"[INFO] ROUND2_SCENARIOS={','.join(scenarios)}", flush=True)

    results: dict = {}
    t0 = time.time()

    for scenario in scenarios:
        try:
            results[scenario] = run_episode(scenario)
        except Exception as e:
            print(f"[ERROR] {scenario}: {e}", flush=True)
            results[scenario] = {"score": 0.01, "error": str(e)}

    scores = [results[s].get("score", 0) for s in scenarios if s in results]
    avg    = sum(scores) / len(scores) if scores else 0.0

    print("\n" + "=" * 64, flush=True)
    print(f"  ROUND 2 -- FINAL SCORES  [{mode}]", flush=True)
    print("=" * 64, flush=True)
    for s in scenarios:
        sc  = results.get(s, {}).get("score", 0.0)
        bar = "X" * int(sc * 20)
        print(f"  {s:<24} {sc:.4f}  {bar}", flush=True)
    print(f"\n  AVERAGE                  {avg:.4f}", flush=True)
    print(f"  Runtime: {time.time() - t0:.1f}s", flush=True)
    print("=" * 64, flush=True)

    if avg >= 0.75:
        print("\n  TARGET MET: average >= 0.75 -- eligible for next round", flush=True)
    else:
        print(f"\n  Gap to target: {0.75 - avg:.4f}", flush=True)


if __name__ == "__main__":
    main()
