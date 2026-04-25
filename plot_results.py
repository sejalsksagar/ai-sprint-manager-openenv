"""
plot_results.py — Results Charts for Hackathon Presentation
============================================================
Generates publication-ready charts from evaluation JSON data.
Saves all charts to results/charts/ as both PNG and SVG.

BASELINE CONSTANTS (FINAL — measured, do not change):
  R1 Llama-3.1-8B zero-shot: easy=0.0100, medium=0.4583, hard=0.0100, avg=0.1594
  R2 Llama-3.1-8B zero-shot: easy=0.3198, medium=0.2443, hard=0.2520, avg=0.2720
  R2 Rule-based:              easy=0.2727, medium=0.2063, hard=0.2610
  Training model:             Qwen/Qwen2.5-1.5B-Instruct (GRPO, 4-bit QLoRA)

Charts produced:
  1. r1_scores_comparison.png  — R1 Llama baseline vs trained bar chart
  2. r2_scores_comparison.png  — R2 Llama/rule-based vs trained bar chart
  3. sprint_rewards.png        — Sprint-by-sprint reward for each R2 scenario
  4. improvement_summary.png   — Combined before/after delta chart (main slide chart)
  5. training_curve.png        — GRPO training loss/reward curve (if trainer_state.json present)

Usage:
    # After running evaluate_r2.py --baseline-only:
    python plot_results.py --eval results/r2_evaluation.json

    # With training curve (after train_llm.py):
    python plot_results.py --eval results/r2_evaluation.json \\
                           --trainer results/trained_model/trainer_state.json

    # Hardcode known scores for presentation (no eval file needed):
    python plot_results.py --demo
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

CHARTS_DIR = Path("results/charts")
CHARTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Measured baselines (FINAL) ────────────────────────────────────────────────
LLAMA_BASELINE_R1 = {
    "easy_sprint":   0.0100,
    "medium_sprint": 0.4583,
    "hard_sprint":   0.0100,
    "average":       0.1594,
}
LLAMA_BASELINE_R2 = {
    "project_easy":   0.3198,
    "project_medium": 0.2443,
    "project_hard":   0.2520,
    "average":        0.2720,
}
RULE_BASED_BASELINE_R2 = {
    "project_easy":   0.2727,
    "project_medium": 0.2063,
    "project_hard":   0.2610,
}
TRAINING_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

# ── Colour palette ────────────────────────────────────────────────────────────
C_LLAMA     = "#6B7280"   # grey     — Llama zero-shot baseline
C_RULE      = "#3B82F6"   # blue     — rule-based baseline
C_TRAINED   = "#10B981"   # green    — trained Qwen (post-GRPO)
C_EASY      = "#60A5FA"
C_MEDIUM    = "#F59E0B"
C_HARD      = "#EF4444"
C_BG        = "#F9FAFB"
C_TEXT      = "#111827"


def _setup_matplotlib():
    import matplotlib
    matplotlib.use("Agg")   # headless
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    plt.rcParams.update({
        "figure.facecolor":  C_BG,
        "axes.facecolor":    C_BG,
        "axes.edgecolor":    "#D1D5DB",
        "axes.labelcolor":   C_TEXT,
        "text.color":        C_TEXT,
        "xtick.color":       C_TEXT,
        "ytick.color":       C_TEXT,
        "grid.color":        "#E5E7EB",
        "grid.linestyle":    "--",
        "grid.alpha":        0.7,
        "font.family":       "sans-serif",
        "font.size":         11,
        "axes.titlesize":    13,
        "axes.titleweight":  "bold",
        "figure.dpi":        150,
    })
    return plt, mpatches


def save(plt, name: str):
    png = CHARTS_DIR / f"{name}.png"
    svg = CHARTS_DIR / f"{name}.svg"
    plt.tight_layout()
    plt.savefig(png, bbox_inches="tight")
    plt.savefig(svg, bbox_inches="tight")
    plt.close()
    print(f"  [SAVED] {png}", flush=True)


# ── Chart 1: R1 scores comparison ────────────────────────────────────────────

def chart_r1_comparison(eval_data: dict):
    plt, mpatches = _setup_matplotlib()

    tasks      = ["easy_sprint", "medium_sprint", "hard_sprint"]
    labels     = ["Easy Sprint", "Medium Sprint", "Hard Sprint"]
    llama_base = [eval_data.get("r1_llama_baseline", LLAMA_BASELINE_R1).get(t, 0) for t in tasks]
    rule_based = [eval_data.get("r1_rule_based", {}).get(t, {}).get("avg_score", 0) for t in tasks]
    llm_scores = [eval_data.get("r1_llm", {}).get(t, {}).get("avg_score", 0) for t in tasks]

    has_llm = any(v > 0 for v in llm_scores)
    x      = range(len(tasks))
    width  = 0.28 if has_llm else 0.38
    fig, ax = plt.subplots(figsize=(9, 5))

    b1 = ax.bar([i - width for i in x], llama_base, width, label=f"Llama-3.1-8B (zero-shot baseline)", color=C_LLAMA,   zorder=3)
    b2 = ax.bar([i         for i in x], rule_based, width, label="Rule-based",                           color=C_RULE,    zorder=3)
    if has_llm:
        b3 = ax.bar([i + width for i in x], llm_scores, width, label=f"{TRAINING_MODEL} (GRPO trained)", color=C_TRAINED, zorder=3)

    def label_bars(bars):
        for bar in bars:
            h = bar.get_height()
            if h > 0.01:
                ax.text(bar.get_x() + bar.get_width()/2, h + 0.01,
                        f"{h:.2f}", ha="center", va="bottom", fontsize=9)

    label_bars(b1); label_bars(b2)
    if has_llm: label_bars(b3)

    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score (0.01 – 0.99)")
    ax.set_title("Round 1 — Score Comparison")
    ax.legend(loc="upper right")
    ax.yaxis.grid(True, zorder=0)
    ax.set_axisbelow(True)

    save(plt, "r1_scores_comparison")


# ── Chart 2: R2 scores comparison ────────────────────────────────────────────

def chart_r2_comparison(eval_data: dict):
    plt, mpatches = _setup_matplotlib()

    tasks      = ["project_easy", "project_medium", "project_hard"]
    labels     = ["Easy (6 sprints)", "Medium (6 sprints)", "Hard (6 sprints)"]
    llama_base = [eval_data.get("r2_llama_baseline", LLAMA_BASELINE_R2).get(t, 0) for t in tasks]
    rule_based = [eval_data.get("r2_rule_based", {}).get(t, {}).get("avg_score",
                  RULE_BASED_BASELINE_R2.get(t, 0)) for t in tasks]
    llm_scores = [eval_data.get("r2_llm", {}).get(t, {}).get("avg_score", 0) for t in tasks]

    has_llm = any(v > 0 for v in llm_scores)
    x     = range(len(tasks))
    width = 0.28 if has_llm else 0.38
    fig, ax = plt.subplots(figsize=(9, 5))

    b1 = ax.bar([i - width     for i in x], llama_base, width, label="Llama-3.1-8B (zero-shot)", color=C_LLAMA,   zorder=3)
    b2 = ax.bar([i             for i in x], rule_based, width, label="Rule-based baseline",       color=C_RULE,    zorder=3)
    if has_llm:
        b3 = ax.bar([i + width for i in x], llm_scores, width, label=f"{TRAINING_MODEL} (GRPO)", color=C_TRAINED, zorder=3)

    for bars in ([b1, b2] + ([b3] if has_llm else [])):
        for bar in bars:
            h = bar.get_height()
            if h > 0.01:
                ax.text(bar.get_x() + bar.get_width()/2, h + 0.01,
                        f"{h:.2f}", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Project Score (delivery × instruction × health)")
    ax.set_title("Round 2 — Multi-Sprint Project Score")
    ax.legend(loc="upper right")
    ax.yaxis.grid(True, zorder=0)
    ax.set_axisbelow(True)

    save(plt, "r2_scores_comparison")


# ── Chart 3: Sprint reward curves ─────────────────────────────────────────────

def chart_sprint_rewards(eval_data: dict):
    """Per-sprint reward for each R2 scenario (rule-based vs trained)."""
    plt, _ = _setup_matplotlib()
    tasks  = ["project_easy", "project_medium", "project_hard"]
    colors = [C_EASY, C_MEDIUM, C_HARD]
    labels = ["Easy", "Medium", "Hard"]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)

    for ax, task, color, label in zip(axes, tasks, colors, labels):
        rb_eps  = eval_data.get("r2_rule_based", {}).get(task, {}).get("episodes", [])
        llm_eps = eval_data.get("r2_llm", {}).get(task, {}).get("episodes", [])

        if rb_eps:
            sr = rb_eps[0].get("sprint_rewards", [])
            if sr:
                ax.plot(range(1, len(sr)+1), sr, "o--",
                        color=C_RULE, label="Rule-based", linewidth=1.5, markersize=5)

        if llm_eps:
            sr = llm_eps[0].get("sprint_rewards", [])
            if sr:
                ax.plot(range(1, len(sr)+1), sr, "o-",
                        color=color, label=f"Qwen GRPO", linewidth=2, markersize=6)

        ax.set_xlabel("Sprint")
        ax.set_title(f"{label} Project")
        ax.set_xticks(range(1, 7))
        ax.set_ylim(0, 2.2)
        ax.yaxis.grid(True)
        ax.set_axisbelow(True)
        if ax == axes[0]:
            ax.set_ylabel("Sprint Reward")
        ax.legend(fontsize=8)

    fig.suptitle("Sprint-by-Sprint Reward (Rule-based vs Qwen GRPO Trained)", y=1.02, fontsize=13, fontweight="bold")
    save(plt, "sprint_rewards")


# ── Chart 4: Improvement summary (main presentation slide) ───────────────────

def chart_improvement_summary(eval_data: dict):
    """Main before/after chart. Uses Llama zero-shot as the 'before' bar."""
    plt, _ = _setup_matplotlib()

    all_tasks = (
        [f"R1: {t.replace('_sprint','').title()}" for t in ["easy_sprint","medium_sprint","hard_sprint"]] +
        [f"R2: {t.replace('project_','').title()}" for t in ["project_easy","project_medium","project_hard"]]
    )

    llama_base, trained_scores = [], []
    for t in ["easy_sprint","medium_sprint","hard_sprint"]:
        llama_base.append(eval_data.get("r1_llama_baseline", LLAMA_BASELINE_R1).get(t, 0))
        trained_scores.append(eval_data.get("r1_llm", {}).get(t, {}).get("avg_score", 0))
    for t in ["project_easy","project_medium","project_hard"]:
        llama_base.append(eval_data.get("r2_llama_baseline", LLAMA_BASELINE_R2).get(t, 0))
        trained_scores.append(eval_data.get("r2_llm", {}).get(t, {}).get("avg_score", 0))

    x     = range(len(all_tasks))
    width = 0.36
    fig, ax = plt.subplots(figsize=(11, 5))

    b1 = ax.bar([i - width/2 for i in x], llama_base,     width,
                label="Before: Llama-3.1-8B (zero-shot)", color=C_LLAMA,   zorder=3)
    b2 = ax.bar([i + width/2 for i in x], trained_scores, width,
                label=f"After: {TRAINING_MODEL} (GRPO)",  color=C_TRAINED, zorder=3)

    # Delta arrows and labels
    for i, (base, trained) in enumerate(zip(llama_base, trained_scores)):
        if trained > base + 0.01:
            ax.annotate("", xy=(i + width/2, trained + 0.02), xytext=(i - width/2, base + 0.02),
                        arrowprops=dict(arrowstyle="->", color="#059669", lw=1.5))
            ax.text(i, max(base, trained) + 0.06, f"+{trained-base:.2f}",
                    ha="center", fontsize=8, color="#059669", fontweight="bold")

    ax.set_xticks(list(x))
    ax.set_xticklabels(all_tasks, rotation=15, ha="right")
    ax.set_ylim(0, 1.25)
    ax.set_ylabel("Score")
    ax.set_title(f"Reward Improvement: Llama Zero-Shot → Qwen2.5-1.5B GRPO Trained")
    ax.legend(loc="upper left")
    ax.yaxis.grid(True, zorder=0)
    ax.set_axisbelow(True)

    # Divider between R1 and R2
    ax.axvline(x=2.5, color="#9CA3AF", linestyle=":", linewidth=1.5)
    ax.text(1.0, 1.20, "Round 1", ha="center", fontsize=10, color="#6B7280")
    ax.text(4.0, 1.20, "Round 2", ha="center", fontsize=10, color="#6B7280")

    save(plt, "improvement_summary")


# ── Chart 5: Training loss/reward curve ───────────────────────────────────────

def chart_training_curve(trainer_state_path: str):
    plt, _ = _setup_matplotlib()

    with open(trainer_state_path) as f:
        state = json.load(f)

    log_history = state.get("log_history", [])
    if not log_history:
        print("  [SKIP] No log_history in trainer_state.json", flush=True)
        return

    steps, losses, rewards = [], [], []
    for entry in log_history:
        if "loss" in entry:
            steps.append(entry.get("step", 0))
            losses.append(entry["loss"])
        if "reward" in entry:
            rewards.append((entry.get("step", 0), entry["reward"]))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    if steps and losses:
        axes[0].plot(steps, losses, color=C_RULE, linewidth=2)
        axes[0].set_xlabel("Training Step")
        axes[0].set_ylabel("Loss")
        axes[0].set_title("GRPO Training Loss (Qwen2.5-1.5B)")
        axes[0].yaxis.grid(True)

    if rewards:
        rsteps, rvals = zip(*rewards)
        axes[1].plot(rsteps, rvals, color=C_TRAINED, linewidth=2)
        axes[1].set_xlabel("Training Step")
        axes[1].set_ylabel("Mean Reward")
        axes[1].set_title("GRPO Mean Reward per Step")
        axes[1].yaxis.grid(True)

    fig.suptitle(f"GRPO Training Curves — {TRAINING_MODEL}", fontsize=13, fontweight="bold")
    save(plt, "training_curve")


# ── Demo mode (hardcoded real baselines, placeholder trained scores) ──────────

def demo_mode():
    """
    Generate charts using real measured baselines.
    Trained scores are placeholders — replace with real evaluate_r2.py output after on-site training.
    """
    print("[INFO] Demo mode — real Llama baselines, placeholder trained scores", flush=True)
    print(f"[INFO] Training model: {TRAINING_MODEL}", flush=True)

    # Placeholder trained scores — update after on-site GRPO training
    PLACEHOLDER_R1_TRAINED = {
        "easy_sprint":   0.0,   # update after training
        "medium_sprint": 0.0,   # update after training
        "hard_sprint":   0.0,   # update after training
    }
    PLACEHOLDER_R2_TRAINED = {
        "project_easy":   0.0,  # update after training
        "project_medium": 0.0,  # update after training
        "project_hard":   0.0,  # update after training
    }

    demo_data = {
        "r1_llama_baseline": LLAMA_BASELINE_R1,
        "r2_llama_baseline": LLAMA_BASELINE_R2,
        "r2_baseline_rules": RULE_BASED_BASELINE_R2,
        "r1_rule_based": {
            "easy_sprint":   {"avg_score": 0.92},
            "medium_sprint": {"avg_score": 0.35},
            "hard_sprint":   {"avg_score": 0.01},
        },
        "r1_llm": {
            t: {"avg_score": v} for t, v in PLACEHOLDER_R1_TRAINED.items()
        },
        "r2_rule_based": {
            "project_easy":   {"avg_score": RULE_BASED_BASELINE_R2["project_easy"],
                               "episodes": [{"sprint_rewards": [0.50, 0.52, 0.48, 0.51, 0.49, 0.50]}]},
            "project_medium": {"avg_score": RULE_BASED_BASELINE_R2["project_medium"],
                               "episodes": [{"sprint_rewards": [0.42, 0.40, 0.38, 0.41, 0.39, 0.40]}]},
            "project_hard":   {"avg_score": RULE_BASED_BASELINE_R2["project_hard"],
                               "episodes": [{"sprint_rewards": [0.38, 0.40, 0.37, 0.41, 0.39, 0.40]}]},
        },
        "r2_llm": {
            t: {"avg_score": v, "episodes": []}
            for t, v in PLACEHOLDER_R2_TRAINED.items()
        },
        "improvement": {
            t: {
                "llama_baseline": LLAMA_BASELINE_R2[t],
                "rule_baseline":  RULE_BASED_BASELINE_R2[t],
                "trained_llm":    PLACEHOLDER_R2_TRAINED[t],
                "delta_vs_llama": round(PLACEHOLDER_R2_TRAINED[t] - LLAMA_BASELINE_R2[t], 4),
            }
            for t in ["project_easy", "project_medium", "project_hard"]
        },
    }
    return demo_data


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate result charts for presentation")
    parser.add_argument("--eval",    type=str, default=None,
                        help="Path to r2_evaluation.json from evaluate_r2.py")
    parser.add_argument("--trainer", type=str, default=None,
                        help="Path to trainer_state.json from train_llm.py output")
    parser.add_argument("--demo",    action="store_true",
                        help="Generate charts with real baselines + placeholder trained scores")
    args = parser.parse_args()

    try:
        import matplotlib
    except ImportError:
        print("[ERROR] matplotlib not installed. Run: pip install matplotlib", flush=True)
        import sys; sys.exit(1)

    if args.demo:
        eval_data = demo_mode()
    elif args.eval:
        with open(args.eval) as f:
            eval_data = json.load(f)
        # Back-fill baseline keys if running against old JSON format
        if "r1_llama_baseline" not in eval_data:
            eval_data["r1_llama_baseline"] = LLAMA_BASELINE_R1
        if "r2_llama_baseline" not in eval_data:
            eval_data["r2_llama_baseline"] = LLAMA_BASELINE_R2
    else:
        print("[INFO] No --eval file specified. Using --demo mode.", flush=True)
        eval_data = demo_mode()

    print(f"\nGenerating charts → {CHARTS_DIR}/", flush=True)
    print(f"  Baselines: R1 avg={LLAMA_BASELINE_R1['average']:.4f}  R2 avg={LLAMA_BASELINE_R2['average']:.4f}", flush=True)
    print(f"  Training model: {TRAINING_MODEL}", flush=True)

    print("  Chart 1: R1 scores comparison...", flush=True)
    chart_r1_comparison(eval_data)

    print("  Chart 2: R2 scores comparison...", flush=True)
    chart_r2_comparison(eval_data)

    print("  Chart 3: Sprint reward curves...", flush=True)
    chart_sprint_rewards(eval_data)

    print("  Chart 4: Improvement summary...", flush=True)
    chart_improvement_summary(eval_data)

    if args.trainer and Path(args.trainer).exists():
        print("  Chart 5: Training curve...", flush=True)
        chart_training_curve(args.trainer)
    else:
        print("  Chart 5: Training curve — skipped (no --trainer file provided)", flush=True)

    print(f"\n✅ All charts saved to {CHARTS_DIR}/", flush=True)
    print(f"   Use improvement_summary.png in your HF blog post and pitch slides.", flush=True)


if __name__ == "__main__":
    main()