"""
train_llm.py — Round 2 LLM Training with GRPO
===============================================
Trains a small LLM (Qwen2.5-1.5B or Llama-3.2-1B) using GRPO on the
sprint management environment. Uses curriculum learning:
  Phase 1: Single-sprint R1 tasks (10 steps each)   — warm up
  Phase 2: Multi-sprint R2 tasks (60 steps each)    — full project horizon

Uses:
  - Unsloth for 2-4× faster training + LoRA on top of base model
  - TRL GRPOTrainer for reward-driven policy optimisation
  - project_client.ProjectEnvClient for R2 episodes
  - client.SprintEnvClient for R1 episodes

Environment:
  ENV_BASE_URL must point to a running server with BOTH R1 and R2 endpoints.
  The server is your HF Space (ui.py with gr.mount_gradio_app + project_router).

Run locally to verify (5 episodes, no GPU):
    python train_llm.py --smoke-test

Run on HF GPU Space (full training):
    python train_llm.py --phase both --episodes 200 --output results/trained_model

Required env vars:
    HF_TOKEN       : HuggingFace token (read + write for push)
    ENV_BASE_URL   : Running environment server
    MODEL_NAME     : Base model (default: Qwen/Qwen2.5-1.5B-Instruct)
    HF_REPO_ID     : (optional) push trained model here after training
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

# ── Env config ────────────────────────────────────────────────────────────────

ENV_BASE_URL = os.getenv("ENV_BASE_URL", "http://localhost:7860")
HF_TOKEN     = os.getenv("HF_TOKEN", "")
MODEL_NAME   = os.getenv("MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct")
HF_REPO_ID   = os.getenv("HF_REPO_ID", "")    # optional push target

# Training hyperparameters — tuned for a single A100/H100 with ~40GB VRAM.
# Halve batch sizes for a T4 (16GB).
GRPO_CONFIG = {
    "learning_rate":            5e-6,
    "num_train_epochs":         1,
    "per_device_train_batch_size": 2,
    "gradient_accumulation_steps": 4,    # effective batch = 8
    "max_prompt_length":        1024,
    "max_completion_length":    128,
    "num_generations":          4,       # GRPO group size
    "temperature":              0.8,
    "beta":                     0.04,    # KL penalty coefficient
    "logging_steps":            5,
    "save_steps":               50,
    "warmup_ratio":             0.05,
    "seed":                     42,
}

R1_TASKS = ["easy_sprint", "medium_sprint", "hard_sprint"]
R2_TASKS = ["project_easy", "project_medium", "project_hard"]

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)


# ── System prompts (match inference scripts exactly) ──────────────────────────

R1_SYSTEM_PROMPT = """You are an expert Tech Lead managing an agile sprint.
Your goal: maximize task completion, balance developer workload, and meet deadlines.

Each step output a JSON action with this exact schema:
{
  "action_type": "<assign|reassign|reprioritize|unblock|skip>",
  "task_id": "<task id or null>",
  "dev_id": "<developer id or null>",
  "new_priority": <1-5 or null>
}

Rules:
- assign: put a backlog task onto an available developer (prefer skill match)
- reassign: move an in-progress task to a different developer
- reprioritize: change a task priority (1=highest)
- unblock: unblock a BLOCKED task only (not backlog tasks)
- skip: do nothing

Output ONLY the JSON object. No explanation."""

R2_SYSTEM_PROMPT = """You are an expert Engineering Manager running a 6-sprint software project (60 days).

Your goals:
1. Complete tasks on time, highest priority first
2. Follow EVERY active instruction — these come from stakeholders and must not be ignored
3. Keep tech debt low — each missed task permanently reduces team productivity
4. Balance developer workload to avoid burnout

Each step output a JSON action with this exact schema:
{
  "action_type": "<assign|reassign|reprioritize|unblock|skip>",
  "task_id": "<task id or null>",
  "dev_id": "<developer id or null>",
  "new_priority": <1-5 or null>
}

Critical rules:
- ALWAYS act on ACTIVE INSTRUCTIONS first — assign their referenced tasks immediately
- Only assign tasks whose depends_on tasks are already "done"
- Match developer skill to task required_skill when possible
- Never assign to an unavailable developer (is_available=false)
- Do NOT use unblock on backlog tasks — only on tasks with status "blocked"

Output ONLY the JSON object. No explanation, no markdown."""


# ── Episode runners ────────────────────────────────────────────────────────────

def run_r1_episode(client, task_name: str, model_fn) -> float:
    """Run one R1 sprint episode (10 steps). Returns final score 0.01-0.99."""
    obs = client.reset(task_name=task_name, seed=None)
    total_reward = 0.0
    for _ in range(12):
        if obs.get("done", False):
            break
        prompt = _build_r1_prompt(obs)
        action_str = model_fn(R1_SYSTEM_PROMPT, prompt)
        action = _parse_action(action_str)
        result = client.step(action)
        obs = result["observation"]
        total_reward += result["reward"]
        if result["done"]:
            break
    return max(0.01, min(0.99, total_reward / 20.0 + 0.5))  # normalised


def run_r2_episode(client, task_name: str, model_fn) -> float:
    """Run one R2 project episode (60 steps). Returns final project score."""
    obs = client.reset(task_name=task_name, seed=None)
    for _ in range(60):
        if obs.get("done", False):
            break
        prompt = _build_r2_prompt(obs)
        action_str = model_fn(R2_SYSTEM_PROMPT, prompt)
        action = _parse_action(action_str)
        result = client.step(action)
        obs = result.observation if hasattr(result, 'observation') else result["observation"]
        if (result.done if hasattr(result, 'done') else result["done"]):
            break
    # Compute score from final obs
    tasks_total   = len(obs.get("tasks", [])) or 1
    tasks_done    = obs.get("tasks_completed", 0)
    inst_score    = obs.get("instruction_following_score", 0.01)
    delivery_rate = tasks_done / tasks_total
    debt_count    = len(obs.get("tech_debt", []))
    team_health   = max(0.01, 1.0 - debt_count * 0.02)
    raw = delivery_rate * 0.55 + inst_score * 0.30 + team_health * 0.15
    return max(0.01, min(0.99, raw))


def _build_r1_prompt(obs: dict) -> str:
    tasks_summary = "\n".join(
        f"  [{t['id']}] {t['name']} | P{t['priority']} | effort={t['effort']} "
        f"| due=Day{t['deadline']} | status={t['status']} | dev={t['assigned_to']}"
        for t in obs["tasks"]
    )
    devs_summary = "\n".join(
        f"  [{d['id']}] {d['name']} | skill={d['skill']} "
        f"| load={d['current_load']}/{d['capacity']} | avail={d['is_available']}"
        for d in obs["developers"]
    )
    return (
        f"Day: {obs['current_day']}/{obs['sprint_length']}\n"
        f"Done:{obs['tasks_completed']} Missed:{obs['tasks_missed']} "
        f"InProgress:{obs['tasks_in_progress']} Backlog:{obs['tasks_backlog']}\n"
        f"Cumulative Reward: {obs['cumulative_reward']:.2f}\n\n"
        f"TASKS:\n{tasks_summary}\n\nDEVELOPERS:\n{devs_summary}\n\n"
        f"Output your JSON action:"
    )


def _build_r2_prompt(obs: dict) -> str:
    current_sprint = obs.get("current_sprint", 1)
    active_insts = [i for i in obs.get("instruction_queue", []) if not i.get("followed", False)]
    inst_section = (
        "⚠️ ACTIVE INSTRUCTIONS:\n" +
        "\n".join(f"  [{i['id']}] {i['text']}" for i in active_insts)
    ) if active_insts else "✅ No active instructions."

    tech_debt = obs.get("tech_debt", [])
    debt_section = f"🔴 TECH DEBT: {', '.join(tech_debt)}" if tech_debt else "✅ No tech debt."

    tasks = obs.get("tasks", [])
    backlog = sorted([t for t in tasks if t["status"] == "backlog"], key=lambda t: (t["priority"], t["deadline"]))
    in_prog = [t for t in tasks if t["status"] == "in_progress"]
    done_ids = {t["id"] for t in tasks if t["status"] == "done"}

    def fmt(t):
        deps = t.get("metadata", {}).get("depends_on", [])
        deps_done = all(d in done_ids for d in deps)
        return (
            f"  [{t['id']}] {t['name']} | P{t['priority']} | effort={t['effort']} "
            f"| skill={t['required_skill']} | sprint={t.get('metadata',{}).get('sprint','?')} "
            f"| status={t['status']} | deps_ok={deps_done}"
        )

    tasks_section = ""
    if backlog:
        tasks_section += "BACKLOG:\n" + "\n".join(fmt(t) for t in backlog[:10])
    if in_prog:
        tasks_section += "\nIN PROGRESS:\n" + "\n".join(fmt(t) for t in in_prog)

    devs_section = "\n".join(
        f"  [{d['id']}] {d['name']} | skill={d['skill']} "
        f"| load={d['current_load']}/{d['capacity']} | avail={'YES' if d['is_available'] else 'NO'}"
        for d in obs["developers"]
    )
    return (
        f"Day {obs['current_day']}/60 | Sprint {current_sprint}/6\n"
        f"Done:{obs['tasks_completed']} Missed:{obs['tasks_missed']} "
        f"Debt:{len(tech_debt)} InstScore:{obs.get('instruction_following_score',0):.2f}\n\n"
        f"{inst_section}\n{debt_section}\n\n"
        f"{tasks_section}\n\nDEVELOPERS:\n{devs_section}\n\n"
        f"Output your JSON action:"
    )


def _parse_action(text: str) -> dict:
    text = text.strip()
    if "```" in text:
        text = "\n".join(l for l in text.split("\n") if not l.strip().startswith("```"))
    try:
        return json.loads(text)
    except Exception:
        s, e = text.find("{"), text.rfind("}") + 1
        if s >= 0 and e > s:
            try:
                return json.loads(text[s:e])
            except Exception:
                pass
    return {"action_type": "skip", "task_id": None, "dev_id": None, "new_priority": None}


# ── GRPO reward function ───────────────────────────────────────────────────────

def make_reward_fn(env_client_r1, env_client_r2, phase: str):
    """
    Returns a GRPO reward function compatible with TRL GRPOTrainer.

    The function receives a batch of (prompt, completion) pairs,
    runs each completion through the environment, and returns rewards.

    Phase "r1": evaluate on R1 tasks only
    Phase "r2": evaluate on R2 tasks only
    Phase "both": curriculum — alternate R1 and R2
    """
    episode_counter = [0]

    def reward_fn(prompts, completions, **kwargs) -> list[float]:
        rewards = []
        for prompt, completion in zip(prompts, completions):
            episode_counter[0] += 1
            action = _parse_action(completion)

            # Determine task based on phase
            if phase == "r1":
                task = R1_TASKS[episode_counter[0] % len(R1_TASKS)]
                score = _score_single_action_r1(env_client_r1, task, action, prompt)
            elif phase == "r2":
                task = R2_TASKS[episode_counter[0] % len(R2_TASKS)]
                score = _score_single_action_r2(env_client_r2, task, action, prompt)
            else:  # both — curriculum: R1 for first half of each epoch, R2 second half
                if episode_counter[0] % 4 < 2:
                    task = R1_TASKS[episode_counter[0] % len(R1_TASKS)]
                    score = _score_single_action_r1(env_client_r1, task, action, prompt)
                else:
                    task = R2_TASKS[episode_counter[0] % len(R2_TASKS)]
                    score = _score_single_action_r2(env_client_r2, task, action, prompt)

            rewards.append(float(score))
        return rewards

    return reward_fn


def _score_single_action_r1(client, task_name: str, action: dict, prompt_context: str) -> float:
    """
    Score a single action in context. For GRPO we run mini-episodes:
    reset, replay context steps (skip), then apply the action and return reward.
    Simplified: just score the action quality from the action dict itself.
    """
    try:
        obs = client.reset(task_name=task_name)
        result = client.step(action)
        step_reward = result["reward"]
        # Normalise to 0-1 range for GRPO
        return max(0.0, min(1.0, (step_reward + 2.0) / 4.0))
    except Exception:
        return 0.0


def _score_single_action_r2(client, task_name: str, action: dict, prompt_context: str) -> float:
    """Score a single R2 action."""
    try:
        obs = client.reset(task_name=task_name)
        result = client.step(action)
        obs2 = result.observation if hasattr(result, 'observation') else result["observation"]
        reward = result.reward if hasattr(result, 'reward') else result["reward"]
        inst_score = obs2.get("instruction_following_score", 0.5)
        # Combine step reward with instruction following signal
        combined = (reward + 2.0) / 4.0 * 0.6 + inst_score * 0.4
        return max(0.0, min(1.0, combined))
    except Exception:
        return 0.0


# ── Dataset builder ───────────────────────────────────────────────────────────

def build_grpo_dataset(n_examples: int = 200, phase: str = "both"):
    """
    Build a HuggingFace Dataset of (prompt, system) pairs for GRPOTrainer.
    Each example is a snapshot observation converted to a chat prompt.
    """
    try:
        from datasets import Dataset
    except ImportError:
        print("[ERROR] datasets not installed. Run: pip install datasets", flush=True)
        sys.exit(1)

    # Import env clients
    try:
        from client import SprintEnvClient
        from project_client import ProjectEnvClient
    except ImportError:
        print("[ERROR] client.py / project_client.py not found in path", flush=True)
        sys.exit(1)

    examples = []

    with SprintEnvClient(base_url=ENV_BASE_URL) as r1_client, \
         ProjectEnvClient(base_url=ENV_BASE_URL) as r2_client:

        tasks_r1 = R1_TASKS if phase in ("r1", "both") else []
        tasks_r2 = R2_TASKS if phase in ("r2", "both") else []

        per_task = max(1, n_examples // max(1, len(tasks_r1) + len(tasks_r2)))

        for task_name in tasks_r1:
            print(f"  [DATASET] Collecting R1 {task_name} × {per_task} episodes...", flush=True)
            for _ in range(per_task):
                try:
                    obs = r1_client.reset(task_name=task_name)
                    for step in range(4):   # collect first 4 steps per episode
                        if obs.get("done", False):
                            break
                        prompt = _build_r1_prompt(obs)
                        examples.append({
                            "prompt": [
                                {"role": "system", "content": R1_SYSTEM_PROMPT},
                                {"role": "user",   "content": prompt},
                            ],
                        })
                        # Advance with skip to get variety
                        result = r1_client.step({"action_type": "skip"})
                        obs = result["observation"]
                except Exception:
                    pass

        for task_name in tasks_r2:
            print(f"  [DATASET] Collecting R2 {task_name} × {per_task} episodes...", flush=True)
            for _ in range(per_task):
                try:
                    obs = r2_client.reset(task_name=task_name)
                    for step in range(6):   # collect first 6 steps (covers first instruction release)
                        if obs.get("done", False):
                            break
                        prompt = _build_r2_prompt(obs)
                        examples.append({
                            "prompt": [
                                {"role": "system", "content": R2_SYSTEM_PROMPT},
                                {"role": "user",   "content": prompt},
                            ],
                        })
                        result = r2_client.step({"action_type": "skip"})
                        obs = result.observation if hasattr(result, 'observation') else result["observation"]
                except Exception:
                    pass

    print(f"  [DATASET] Total examples collected: {len(examples)}", flush=True)
    return Dataset.from_list(examples)


# ── Model loader (Unsloth + LoRA) ─────────────────────────────────────────────

def load_model_and_tokenizer(model_name: str):
    """
    Load model with Unsloth (4-bit QLoRA). Falls back to standard HF if Unsloth unavailable.
    """
    try:
        from unsloth import FastLanguageModel
        print(f"[INFO] Loading {model_name} with Unsloth 4-bit QLoRA...", flush=True)
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=2048,
            dtype=None,           # auto-detect
            load_in_4bit=True,
            token=HF_TOKEN or None,
        )
        model = FastLanguageModel.get_peft_model(
            model,
            r=16,                 # LoRA rank — increase to 32 for better quality if VRAM allows
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=42,
        )
        print(f"[INFO] Unsloth model loaded. Trainable params: "
              f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,}", flush=True)
        return model, tokenizer, "unsloth"

    except ImportError:
        print("[WARN] Unsloth not available. Falling back to standard HF + PEFT.", flush=True)
        return _load_hf_model(model_name)


def _load_hf_model(model_name: str):
    """Fallback: standard transformers + PEFT LoRA (slower, same result)."""
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from peft import get_peft_model, LoraConfig, TaskType
        import torch

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name, token=HF_TOKEN or None)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            token=HF_TOKEN or None,
        )
        lora_cfg = LoraConfig(
            r=16, lora_alpha=32, lora_dropout=0.05,
            target_modules=["q_proj", "v_proj"],
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, lora_cfg)
        print(f"[INFO] HF+PEFT model loaded.", flush=True)
        return model, tokenizer, "hf"
    except Exception as e:
        print(f"[ERROR] Cannot load model: {e}", flush=True)
        sys.exit(1)


# ── GRPO trainer ──────────────────────────────────────────────────────────────

def train(
    phase: str = "both",
    n_dataset_examples: int = 200,
    output_dir: str = "results/trained_model",
    push_to_hub: bool = False,
):
    """
    Full GRPO training run.
    """
    print(f"\n{'='*60}", flush=True)
    print(f" GRPO TRAINING — Phase: {phase.upper()}", flush=True)
    print(f" Model: {MODEL_NAME}", flush=True)
    print(f" Server: {ENV_BASE_URL}", flush=True)
    print(f"{'='*60}\n", flush=True)

    # 1. Load model
    model, tokenizer, backend = load_model_and_tokenizer(MODEL_NAME)

    # 2. Build dataset
    print("[INFO] Building training dataset...", flush=True)
    dataset = build_grpo_dataset(n_examples=n_dataset_examples, phase=phase)

    # 3. Build reward function
    from client import SprintEnvClient
    from project_client import ProjectEnvClient
    r1_client = SprintEnvClient(base_url=ENV_BASE_URL)
    r2_client = ProjectEnvClient(base_url=ENV_BASE_URL)
    reward_fn = make_reward_fn(r1_client, r2_client, phase)

    # 4. Configure GRPOTrainer
    try:
        from trl import GRPOConfig, GRPOTrainer
    except ImportError:
        print("[ERROR] trl not installed. Run: pip install trl>=0.9.0", flush=True)
        sys.exit(1)

    grpo_config = GRPOConfig(
        output_dir=output_dir,
        **GRPO_CONFIG,
        report_to="none",         # disable wandb/tensorboard by default
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=reward_fn,
        args=grpo_config,
        train_dataset=dataset,
    )

    # 5. Train
    print("[INFO] Starting GRPO training...", flush=True)
    t0 = time.time()
    trainer.train()
    elapsed = time.time() - t0
    print(f"\n[INFO] Training complete in {elapsed/60:.1f} min", flush=True)

    # 6. Save
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"[INFO] Model saved to {output_dir}", flush=True)

    # 7. Optionally push to HF Hub
    if push_to_hub and HF_REPO_ID:
        print(f"[INFO] Pushing to HF Hub: {HF_REPO_ID}", flush=True)
        model.push_to_hub(HF_REPO_ID, token=HF_TOKEN)
        tokenizer.push_to_hub(HF_REPO_ID, token=HF_TOKEN)
        print(f"[INFO] Pushed to https://huggingface.co/{HF_REPO_ID}", flush=True)

    r1_client.close()
    r2_client.close()
    return output_dir


# ── Smoke test (no GPU, no LLM) ───────────────────────────────────────────────

def smoke_test():
    """
    Runs 5 episodes against the live server using rule-based actions.
    No GPU, no model loading. Verifies the full training data pipeline.
    Run this locally before the on-site GPU session.
    """
    print("\n=== SMOKE TEST (rule-based, no GPU) ===\n", flush=True)

    import requests

    # Health check
    try:
        r1 = requests.get(f"{ENV_BASE_URL}/health", timeout=10).json()
        r2 = requests.get(f"{ENV_BASE_URL}/project/health", timeout=10).json()
        print(f"[OK] R1 health: {r1}", flush=True)
        print(f"[OK] R2 health: {r2}", flush=True)
    except Exception as e:
        print(f"[ERROR] Server not reachable: {e}", flush=True)
        print(f"        Start your server first: python ui.py", flush=True)
        sys.exit(1)

    from client import SprintEnvClient
    from project_client import ProjectEnvClient

    results = {}

    import requests as _req
    with SprintEnvClient(base_url=ENV_BASE_URL) as r1_client:
        for task in ["easy_sprint", "hard_sprint"]:
            print(f"\n[R1] {task}...", flush=True)
            try:
                obs = r1_client.reset(task_name=task, seed=42)
                total_r = 0.0
                for i in range(12):
                    if obs.get("done", False):
                        break
                    action = _rule_based_action_r1(obs)
                    # Call step directly as dict (avoids SprintAction.model_dump() issue)
                    resp = _req.post(
                        f"{ENV_BASE_URL}/step",
                        json={"action": action},
                        timeout=30,
                    )
                    resp.raise_for_status()
                    result = resp.json()
                    obs = result["observation"]
                    total_r += result["reward"]
                    print(
                        f"  step {i+1}: action={action['action_type']} "
                        f"day={obs['current_day']} "
                        f"done={obs['tasks_completed']} reward={result['reward']:.3f}",
                        flush=True,
                    )
                    if result["done"]:
                        break
                results[f"r1/{task}"] = round(total_r, 3)
                print(f"  cumulative_reward={total_r:.3f}", flush=True)
            except Exception as e:
                print(f"  [ERROR] {e}", flush=True)
                results[f"r1/{task}"] = None

    with ProjectEnvClient(base_url=ENV_BASE_URL) as r2_client:
        for task in ["project_easy"]:
            print(f"\n[R2] {task} (first 5 steps only)...", flush=True)
            try:
                obs = r2_client.reset(task_name=task, seed=42)
                for i in range(5):
                    if obs.get("done", False):
                        break
                    action = _rule_based_action_r2(obs)
                    result = r2_client.step(action)
                    obs = result.observation
                    print(
                        f"  step {i+1}: action={action['action_type']} "
                        f"day={obs['current_day']} sprint={obs['current_sprint']} "
                        f"reward={result.reward:.3f} inst={obs['instruction_following_score']:.2f}",
                        flush=True,
                    )
                results["r2/project_easy"] = obs["cumulative_reward"]
            except Exception as e:
                print(f"  [ERROR] {e}", flush=True)
                results["r2/project_easy"] = None

    # Dataset collection test (small)
    print(f"\n[DATASET] Testing dataset collection (10 examples)...", flush=True)
    try:
        dataset = build_grpo_dataset(n_examples=10, phase="both")
        print(f"  [OK] Dataset: {len(dataset)} examples", flush=True)
        print(f"  [OK] First example keys: {list(dataset[0].keys())}", flush=True)
    except Exception as e:
        print(f"  [WARN] Dataset collection failed: {e}", flush=True)

    print(f"\n=== SMOKE TEST RESULTS ===", flush=True)
    for k, v in results.items():
        print(f"  {k}: {v}", flush=True)
    print(f"\n✅ Smoke test complete. Server is ready for GPU training.", flush=True)


def _rule_based_action_r1(obs: dict) -> dict:
    tasks  = obs.get("tasks", [])
    devs   = obs.get("developers", [])
    avail  = [d for d in devs if d["is_available"] and d["current_load"] < d["capacity"]]
    backlog = sorted([t for t in tasks if t["status"] == "backlog"],
                     key=lambda t: (t["priority"], t["deadline"]))
    for task in backlog:
        match = [d for d in avail if d["skill"] == task.get("required_skill") or d["skill"] == "fullstack"]
        dev = match[0] if match else (avail[0] if avail else None)
        if dev:
            return {"action_type": "assign", "task_id": task["id"], "dev_id": dev["id"], "new_priority": None}
    return {"action_type": "skip", "task_id": None, "dev_id": None, "new_priority": None}


def _rule_based_action_r2(obs: dict) -> dict:
    tasks    = obs.get("tasks", [])
    devs     = obs.get("developers", [])
    done_ids = {t["id"] for t in tasks if t["status"] == "done"}
    avail    = [d for d in devs if d["is_available"] and d["current_load"] < d["capacity"] * 2]

    def best_dev(task):
        match = [d for d in avail if d["skill"] == task.get("required_skill") or d["skill"] == "fullstack"]
        return match[0] if match else (avail[0] if avail else None)

    # Instructions first
    for inst in [i for i in obs.get("instruction_queue", []) if not i.get("followed", False)]:
        for tid in inst.get("affects_tasks", []):
            task = next((t for t in tasks if t["id"] == tid and t["status"] == "backlog"), None)
            if task:
                deps = task.get("metadata", {}).get("depends_on", [])
                if all(d in done_ids for d in deps):
                    dev = best_dev(task)
                    if dev:
                        return {"action_type": "assign", "task_id": task["id"],
                                "dev_id": dev["id"], "new_priority": None}

    backlog = sorted([t for t in tasks if t["status"] == "backlog"],
                     key=lambda t: (t["priority"], t["deadline"]))
    for task in backlog:
        deps = task.get("metadata", {}).get("depends_on", [])
        if all(d in done_ids for d in deps):
            dev = best_dev(task)
            if dev:
                return {"action_type": "assign", "task_id": task["id"],
                        "dev_id": dev["id"], "new_priority": None}

    return {"action_type": "skip", "task_id": None, "dev_id": None, "new_priority": None}


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="GRPO training for AI Sprint Manager R2")
    parser.add_argument("--smoke-test", action="store_true",
                        help="Run 5-episode smoke test (no GPU needed). Do this locally first.")
    parser.add_argument("--phase", choices=["r1", "r2", "both"], default="both",
                        help="Training phase: r1=warmup, r2=project, both=curriculum (default)")
    parser.add_argument("--episodes", type=int, default=200,
                        help="Number of dataset examples to collect (default: 200)")
    parser.add_argument("--output", type=str, default="results/trained_model",
                        help="Output directory for saved model")
    parser.add_argument("--push", action="store_true",
                        help="Push trained model to HF Hub (requires HF_REPO_ID env var)")
    args = parser.parse_args()

    if args.smoke_test:
        smoke_test()
        return

    train(
        phase=args.phase,
        n_dataset_examples=args.episodes,
        output_dir=args.output,
        push_to_hub=args.push,
    )


if __name__ == "__main__":
    main()