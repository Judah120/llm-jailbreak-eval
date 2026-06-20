"""
Figure Generator
----------------
Produces all publication-quality figures.
Run after experiment.py: python src/figures.py
"""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

RESULTS_DIR = Path("results")
FIGURES_DIR = Path("figures")
FIGURES_DIR.mkdir(exist_ok=True)

# --- Design tokens ---
DARK_BG   = "#0d1117"
PANEL_BG  = "#161b22"
BORDER    = "#21262d"
TEXT      = "#c9d1d9"
MUTED     = "#6e7681"

C_STRONG  = "#3fb950"   # green  — strong safety
C_MOD     = "#58a6ff"   # blue   — moderate safety
C_WEAK    = "#f78166"   # red    — weak safety

MODEL_COLORS = {
    "Strong Safety":   C_STRONG,
    "Moderate Safety": C_MOD,
    "Weak Safety":     C_WEAK,
}

def style(ax, title="", xlabel="", ylabel=""):
    ax.set_facecolor(PANEL_BG)
    ax.tick_params(colors=TEXT, labelsize=9)
    ax.xaxis.label.set_color(TEXT)
    ax.yaxis.label.set_color(TEXT)
    ax.title.set_color(TEXT)
    for sp in ax.spines.values():
        sp.set_edgecolor(BORDER)
    ax.grid(True, color=BORDER, linewidth=0.5, linestyle="--", alpha=0.6)
    if title:  ax.set_title(title, fontsize=11, fontweight="bold", pad=10)
    if xlabel: ax.set_xlabel(xlabel, fontsize=9)
    if ylabel: ax.set_ylabel(ylabel, fontsize=9)


# ---------------------------------------------------------------------------
# Fig 1: Resistance Score Distribution (grouped bar)
# ---------------------------------------------------------------------------
def plot_score_distribution():
    metrics = json.load(open(RESULTS_DIR / "all_metrics.json"))

    labels = ["Score 3\n(Full Refusal)", "Score 2\n(Partial)", "Score 1\n(Complied+Warning)", "Score 0\n(Jailbroken)"]
    keys   = ["score_3", "score_2", "score_1", "score_0"]
    n_models = len(metrics)
    x = np.arange(len(labels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(11, 5))
    fig.patch.set_facecolor(DARK_BG)

    for i, m in enumerate(metrics):
        counts = [m["score_distribution"][k] for k in keys]
        color  = list(MODEL_COLORS.values())[i]
        bars   = ax.bar(x + i * width - width, counts, width, label=m["label"],
                        color=color, alpha=0.85, edgecolor=DARK_BG, linewidth=0.4)
        for bar, count in zip(bars, counts):
            if count > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.15,
                        str(count), ha="center", va="bottom", fontsize=8, color=TEXT)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    style(ax, "Resistance Score Distribution Across Models", "", "Number of Prompts")
    ax.legend(fontsize=9, facecolor=PANEL_BG, edgecolor=BORDER, labelcolor=TEXT)
    ax.set_ylim(0, max(
        m["score_distribution"][k] for m in metrics for k in keys
    ) + 3)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "score_distribution.png", dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close()
    print("[Figure] score_distribution.png")


# ---------------------------------------------------------------------------
# Fig 2: Key Metrics Radar / Summary Bar
# ---------------------------------------------------------------------------
def plot_summary_metrics():
    metrics = json.load(open(RESULTS_DIR / "all_metrics.json"))

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.patch.set_facecolor(DARK_BG)

    metric_defs = [
        ("mean_resistance_score", "Mean Resistance Score", 3.0),
        ("full_refusal_rate",     "Full Refusal Rate",     1.0),
        ("jailbreak_success_rate","Jailbreak Success Rate", 1.0),
    ]

    for ax, (key, label, max_val) in zip(axes, metric_defs):
        model_labels = [m["label"] for m in metrics]
        values       = [m[key] for m in metrics]
        colors       = [MODEL_COLORS[l] for l in model_labels]

        bars = ax.bar(model_labels, values, color=colors, edgecolor=DARK_BG,
                      linewidth=0.4, alpha=0.88)

        for bar, val in zip(bars, values):
            fmt = f"{val:.3f}" if key == "mean_resistance_score" else f"{val:.1%}"
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max_val * 0.02,
                    fmt, ha="center", va="bottom", fontsize=9, color=TEXT)

        style(ax, label, "", label)
        ax.set_ylim(0, max_val * 1.2)
        ax.set_xticklabels(model_labels, rotation=15, ha="right", fontsize=8)

    plt.suptitle("Model Safety Profile Comparison", fontsize=13, fontweight="bold",
                 color=TEXT, y=1.02)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "summary_metrics.png", dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close()
    print("[Figure] summary_metrics.png")


# ---------------------------------------------------------------------------
# Fig 3: Category Heatmap
# ---------------------------------------------------------------------------
def plot_category_heatmap():
    data = json.load(open(RESULTS_DIR / "category_heatmap.json"))
    metrics = json.load(open(RESULTS_DIR / "all_metrics.json"))

    model_labels = [m["label"] for m in metrics]
    categories   = sorted(list({d["category"] for d in data}))
    cat_labels   = [c.replace("_", " ").title() for c in categories]

    matrix = np.zeros((len(model_labels), len(categories)))
    for d in data:
        r = model_labels.index(d["model"])
        c = categories.index(d["category"])
        matrix[r, c] = d["mean_resistance"]

    fig, ax = plt.subplots(figsize=(12, 4))
    fig.patch.set_facecolor(DARK_BG)

    im = ax.imshow(matrix, cmap="RdYlGn", vmin=0, vmax=3, aspect="auto")

    ax.set_xticks(range(len(categories)))
    ax.set_xticklabels(cat_labels, rotation=25, ha="right", fontsize=9, color=TEXT)
    ax.set_yticks(range(len(model_labels)))
    ax.set_yticklabels(model_labels, fontsize=9, color=TEXT)

    for i in range(len(model_labels)):
        for j in range(len(categories)):
            val = matrix[i, j]
            txt_color = "black" if 0.8 < val < 2.2 else "white"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=9, color=txt_color, fontweight="bold")

    cbar = plt.colorbar(im, ax=ax, pad=0.02)
    cbar.ax.yaxis.set_tick_params(color=TEXT, labelsize=8)
    cbar.set_label("Mean Resistance Score (0–3)", color=TEXT, fontsize=8)
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color=TEXT)

    ax.set_facecolor(PANEL_BG)
    ax.title.set_color(TEXT)
    ax.set_title("Resistance Heatmap: Model × Attack Category", fontsize=11,
                 fontweight="bold", pad=12, color=TEXT)
    for sp in ax.spines.values():
        sp.set_edgecolor(BORDER)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "category_heatmap.png", dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close()
    print("[Figure] category_heatmap.png")


# ---------------------------------------------------------------------------
# Fig 4: Threshold Sensitivity
# ---------------------------------------------------------------------------
def plot_threshold_sensitivity():
    data = json.load(open(RESULTS_DIR / "threshold_sensitivity.json"))

    models    = list(dict.fromkeys(d["model"] for d in data))
    thresholds = sorted(list({d["fail_threshold"] for d in data}))
    x = np.arange(len(thresholds))
    width = 0.25

    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor(DARK_BG)

    for i, model in enumerate(models):
        rates = [
            next(d["fail_rate"] for d in data if d["model"] == model and d["fail_threshold"] == t)
            for t in thresholds
        ]
        color = list(MODEL_COLORS.values())[i]
        bars  = ax.bar(x + i * width - width, [r * 100 for r in rates],
                       width, label=model, color=color, alpha=0.85,
                       edgecolor=DARK_BG, linewidth=0.4)
        for bar, rate in zip(bars, rates):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.8,
                    f"{rate:.0%}", ha="center", va="bottom", fontsize=8, color=TEXT)

    ax.set_xticks(x)
    ax.set_xticklabels([f"Score ≤ {t}\n({'full jailbreak' if t==0 else 'incl. partial'})" 
                        for t in thresholds], fontsize=9)
    style(ax, "Failure Rate by Scoring Threshold", "Threshold Definition", "Failure Rate (%)")
    ax.set_ylim(0, 115)
    ax.legend(fontsize=9, facecolor=PANEL_BG, edgecolor=BORDER, labelcolor=TEXT)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "threshold_sensitivity.png", dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close()
    print("[Figure] threshold_sensitivity.png")


# ---------------------------------------------------------------------------
# Fig 5: Per-category resistance — all models overlaid
# ---------------------------------------------------------------------------
def plot_category_profiles():
    data    = json.load(open(RESULTS_DIR / "category_heatmap.json"))
    metrics = json.load(open(RESULTS_DIR / "all_metrics.json"))

    model_labels = [m["label"] for m in metrics]
    categories   = sorted(list({d["category"] for d in data}))
    cat_labels   = [c.replace("_", " ").title() for c in categories]

    fig, ax = plt.subplots(figsize=(12, 5))
    fig.patch.set_facecolor(DARK_BG)

    x = np.arange(len(categories))
    for i, model in enumerate(model_labels):
        scores = []
        for cat in categories:
            row = next((d for d in data if d["model"] == model and d["category"] == cat), None)
            scores.append(row["mean_resistance"] if row else 0)
        color = list(MODEL_COLORS.values())[i]
        ax.plot(x, scores, marker="o", linewidth=2.2, label=model,
                color=color, markersize=7)
        ax.fill_between(x, scores, alpha=0.06, color=color)

    ax.set_xticks(x)
    ax.set_xticklabels(cat_labels, rotation=20, ha="right", fontsize=9)
    ax.set_ylim(0, 3.3)
    ax.axhline(1.5, color=MUTED, linewidth=0.8, linestyle="--", label="Midpoint (1.5)")
    style(ax, "Resistance Profile by Attack Category", "", "Mean Resistance Score (0–3)")
    ax.legend(fontsize=9, facecolor=PANEL_BG, edgecolor=BORDER, labelcolor=TEXT)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "category_profiles.png", dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close()
    print("[Figure] category_profiles.png")


if __name__ == "__main__":
    plot_score_distribution()
    plot_summary_metrics()
    plot_category_heatmap()
    plot_threshold_sensitivity()
    plot_category_profiles()
    print("\n[Done] All figures saved to /figures")
