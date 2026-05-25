"""Generate a publication-ready sweep pipeline flowchart.

Two versions:
  --style=horizontal  : 5-stage left-to-right pipeline (default)
  --style=vertical    : 5-stage top-to-bottom pipeline
  --style=tikz        : print TikZ code to stdout (for LaTeX papers)

Output: logs/go2_sweep/figures/sweep_pipeline_flowchart.png
"""

import argparse
import os
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(SCRIPT_DIR, "..", "logs", "go2_sweep", "figures")
OUTPUT = os.path.join(FIG_DIR, "sweep_pipeline_flowchart.png")

# ── Color palette ──
C = {
    "blue":   "#3969ac",
    "orange": "#e4572e",
    "green":  "#3a9b6a",
    "purple": "#7b4ea3",
    "teal":   "#118c8b",
    "light":  "#f5f6fa",
    "dark":   "#2c3e50",
    "gray":   "#7f8c8d",
}


def draw_stage_box(ax, x, y, w, h, color):
    """Draw a rounded stage container."""
    box = FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.1",
        facecolor=color, edgecolor="#333333", linewidth=1.2, alpha=0.92,
    )
    ax.add_patch(box)


def draw_sub_box(ax, x, y, w, h, text, face="#f0f4ff", edge="#aaaaaa",
                 fontsize=8, text_color="#222222"):
    """Draw a small detail box."""
    box = FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.04",
        facecolor=face, edgecolor=edge, linewidth=0.8, alpha=0.9,
    )
    ax.add_patch(box)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fontsize, color=text_color)


def arrow(ax, x1, y1, x2, y2, color="#555555", lw=1.8):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", color=color, lw=lw))


def stage_label(ax, x, y, number, title, color):
    ax.text(x, y, f"{number}  {title}", ha="center", fontsize=11,
            weight="bold", color=color)


# ═══════════════════════════════════════════════════════════════════
# HORIZONTAL (5-stage, top row + bottom row)
# ═══════════════════════════════════════════════════════════════════

def draw_horizontal():
    fig, ax = plt.subplots(figsize=(16, 7))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 7)
    ax.set_aspect("equal")
    ax.axis("off")

    # ── Top row: ① Grid → ② Launch → ③ Train ──
    # Stage 1
    draw_stage_box(ax, 0.3, 4.2, 4.0, 2.0, C["blue"])
    stage_label(ax, 2.3, 5.95, "1", "Grid Construction", C["blue"])
    draw_sub_box(ax, 0.6, 5.1, 3.4, 0.42, r"solimp  =  [$d_{\rm min}$, $d_{\rm max}$, width]", fontsize=8.5)
    draw_sub_box(ax, 0.6, 4.7, 3.4, 0.30,
                 r"solimp[0] $\in$ {.015, .03, .1, .35, .7, .9}  |  solimp[1] = 0.95  |  solimp[2] $\in$ {.5, ..., .001}",
                 fontsize=7)
    draw_sub_box(ax, 0.6, 4.4, 3.4, 0.22, r"solref  =  [timeconst, 1.0]    timeconst $\in$ {0.1, 0.02, 0.004}", fontsize=8)
    ax.text(2.3, 4.1, "57 valid triplets  (filter: dmin ≤ dmax)",
            ha="center", fontsize=8, color=C["dark"], style="italic")

    arrow(ax, 4.3, 5.2, 5.6, 5.2)

    # Stage 2
    draw_stage_box(ax, 5.6, 4.2, 5.0, 2.0, C["orange"])
    stage_label(ax, 8.1, 5.95, "2", "Multi-GPU Launch", C["orange"])
    for i in range(4):
        gx = 5.9 + i * 1.15
        gy = 4.9
        FancyBboxPatch((gx, gy), 0.95, 0.75, boxstyle="round,pad=0.04",
                       facecolor="#fff5f0", edgecolor=C["orange"],
                       linewidth=0.8, alpha=0.85, transform=ax.transData, zorder=3)
        # We need to add each patch explicitly — use a loop with draw_sub_box
    for i in range(4):
        gx = 5.9 + i * 1.15
        gy = 4.85
        draw_sub_box(ax, gx, gy, 0.95, 0.70,
                     f"GPU{i}\nslots {i},{i+4},...", fontsize=7,
                     face="#fff5f0", edge=C["orange"])
    ax.text(8.1, 4.1, "Round-robin •  Dynamic scheduling  •  JIT cache sharing",
            ha="center", fontsize=8, color=C["dark"], style="italic")

    arrow(ax, 10.6, 5.2, 11.5, 5.2)

    # Stage 3
    draw_stage_box(ax, 11.5, 4.2, 4.2, 2.0, C["green"])
    stage_label(ax, 13.6, 5.95, "3", "Per-Config Training", C["green"])
    steps3 = [
        "Build train env (swept params) & eval env (fixed)",
        "Train PPO 10 iterations  (JAX / MJX backend)",
        "Render rollout video  (EGL off-screen)",
        "Log metrics to W&B (offline) → *.wandb",
    ]
    for i, s in enumerate(steps3):
        draw_sub_box(ax, 11.8, 5.1 - i * 0.28, 3.6, 0.22, s, fontsize=7, face="#e8f5e9")

    # ── Arrow down ──
    arrow(ax, 13.6, 4.2, 13.6, 2.5)

    # ── Bottom row (right to left): ④ Aggregate ← ⑤ Analyze ──
    # Stage 4
    draw_stage_box(ax, 10.8, 1.2, 5.8, 1.2, C["purple"])
    stage_label(ax, 13.7, 2.15, "4", "Result Aggregation", C["purple"])
    draw_sub_box(ax, 11.0, 1.55, 5.4, 0.25,
                 "Parse slot logs → extract W&B summary JSON →  sweep_results.csv",
                 fontsize=8, face="#f3e5f5")
    draw_sub_box(ax, 11.0, 1.3, 5.4, 0.22,
                 "Per-slot: episode_reward, IQM reward, episode_length, penetration_depth, ...",
                 fontsize=6.5, face="#f3e5f5")

    arrow(ax, 10.8, 1.8, 8.1, 1.8)

    # Stage 5
    draw_stage_box(ax, 0.3, 0.5, 7.8, 2.2, C["teal"])
    stage_label(ax, 4.2, 2.45, "5", "Analysis & Visualization", C["teal"])

    viz_items = [
        # Row 1
        [(0.6, 1.9, 2.3, 0.38, "Marginal Effects (Boxplots)"),
         (3.1, 1.9, 2.3, 0.38, "Factor Main Effects"),
         (5.5, 1.9, 2.3, 0.38, "3D Reward Surfaces")],
        # Row 2
        [(0.6, 1.4, 2.3, 0.38, "solref Sensitivity Trends"),
         (3.1, 1.4, 2.3, 0.38, "Softness Calibration"),
         (5.5, 1.4, 2.3, 0.38, "Collapsed 2D Trends")],
        # Row 3
        [(0.6, 0.9, 2.3, 0.38, "Ranked Bar Chart"),
         (3.1, 0.9, 2.3, 0.38, "Heatmap Grid"),
         (5.5, 0.9, 2.3, 0.38, "Softness vs Reward Scatter")],
    ]
    for row in viz_items:
        for (x, y, w, h, text) in row:
            draw_sub_box(ax, x, y, w, h, text, fontsize=7.5, face="#e0f2f1")

    # ── Key parameter legend ──
    ax.text(8.5, 0.35,
            "solimp = contact impedance (softness)   |   solref = contact reference (spring-damper)   |   "
            "Train with swept params, eval with fixed hard params",
            ha="center", fontsize=7.5, color=C["gray"], style="italic")

    os.makedirs(FIG_DIR, exist_ok=True)
    plt.savefig(OUTPUT, facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"Saved: {OUTPUT}")


# ═══════════════════════════════════════════════════════════════════
# VERTICAL (5-stage, top to bottom)
# ═══════════════════════════════════════════════════════════════════

def draw_vertical():
    fig, ax = plt.subplots(figsize=(8, 14))
    ax.set_xlim(0, 8)
    ax.set_ylim(0, 14)
    ax.set_aspect("equal")
    ax.axis("off")

    # Stage 1
    draw_stage_box(ax, 1.0, 10.5, 6.0, 2.2, C["blue"])
    stage_label(ax, 4.0, 12.4, "1", "Grid Construction", C["blue"])
    draw_sub_box(ax, 1.3, 11.5, 5.4, 0.48,
                 r"solimp = [$d_{\rm min}$, $d_{\rm max}$, width]   |   solref = [timeconst, dampratio]",
                 fontsize=9)
    draw_sub_box(ax, 1.3, 11.0, 5.4, 0.35,
                 r"solimp[0] $\in$ {.015, .03, .1, .35, .7, .9},  solimp[1] = 0.95,  solimp[2] $\in$ {.5, .1, .05, .03, .01, .006, .001}",
                 fontsize=7.5)
    draw_sub_box(ax, 1.3, 10.7, 5.4, 0.28,
                 "solref[0] ∈ {0.1, 0.02, 0.004},  solref[1] = 1.0",
                 fontsize=8)
    ax.text(4.0, 10.4, "filter: dmin ≤ dmax  →  57 valid parameter combinations",
            ha="center", fontsize=8, color=C["dark"], style="italic")

    arrow(ax, 4.0, 10.5, 4.0, 9.0)

    # Stage 2
    draw_stage_box(ax, 1.0, 7.0, 6.0, 1.8, C["orange"])
    stage_label(ax, 4.0, 8.55, "2", "Multi-GPU Distribution", C["orange"])
    for i in range(4):
        gx = 1.3 + i * 1.35
        draw_sub_box(ax, gx, 7.5, 1.15, 0.65,
                     f"GPU {i}\nslots {i},{i+4},..",
                     fontsize=7.5, face="#fff5f0", edge=C["orange"])
    ax.text(4.0, 6.95, "Dynamic scheduling via JIT_READY signal  •  JIT cache shared across slots",
            ha="center", fontsize=8, color=C["dark"], style="italic")

    arrow(ax, 4.0, 7.0, 4.0, 5.8)

    # Stage 3
    draw_stage_box(ax, 1.0, 3.6, 6.0, 2.0, C["green"])
    stage_label(ax, 4.0, 5.35, "3", "Per-Config Training", C["green"])
    steps3 = [
        ("Train Env", "swept solimp/solref"),
        ("Eval Env", "fixed solimp/solref"),
        ("PPO Training", "10 iterations, JAX/MJX"),
        ("Video Render", "EGL off-screen, 256 steps"),
    ]
    for i, (label, desc) in enumerate(steps3):
        gx = 1.3 + i * 1.35
        draw_sub_box(ax, gx, 4.15, 1.15, 0.55,
                     f"{label}\n{desc}", fontsize=7, face="#e8f5e9")

    arrow(ax, 4.0, 3.6, 4.0, 2.5)

    # Stage 4
    draw_stage_box(ax, 1.0, 1.4, 6.0, 0.9, C["purple"])
    stage_label(ax, 4.0, 2.05, "4", "Result Aggregation", C["purple"])
    draw_sub_box(ax, 1.3, 1.6, 5.4, 0.25,
                 "Parse log files → extract W&B summary → sweep_results.csv",
                 fontsize=8, face="#f3e5f5")

    arrow(ax, 4.0, 1.4, 4.0, 0.6)

    # Stage 5
    draw_stage_box(ax, 1.0, -1.0, 6.0, 1.4, C["teal"])
    stage_label(ax, 4.0, 0.15, "5", "Analysis & Visualization", C["teal"])
    vizs = [
        ("Marginal Effects", "Boxplots per factor"),
        ("3D Surfaces", "solimp0 × solimp2"),
        ("Softness Calib.", "Ball-drop → rank"),
        ("solref Sensitivity", "Per-group trends"),
    ]
    for i, (label, desc) in enumerate(vizs):
        gx = 1.3 + i * 1.35
        draw_sub_box(ax, gx, -0.55, 1.15, 0.48, f"{label}\n{desc}",
                     fontsize=7, face="#e0f2f1")

    out_v = OUTPUT.replace(".png", "_vertical.png")
    os.makedirs(FIG_DIR, exist_ok=True)
    plt.savefig(out_v, facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"Saved: {out_v}")


# ═══════════════════════════════════════════════════════════════════
# TIKZ (print to stdout)
# ═══════════════════════════════════════════════════════════════════

def print_tikz():
    print(r"""% sweep_pipeline.tex — drop into your LaTeX paper
% Requires: \usepackage{tikz}  \usetikzlibrary{shapes.geometric, arrows.meta, positioning, fit, backgrounds}

\begin{figure}[t]
\centering
\resizebox{\columnwidth}{!}{%
\begin{tikzpicture}[
    node distance=0.8cm,
    box/.style={rectangle, rounded corners=4pt, draw=#1!60!black, fill=#1!15, thick,
                minimum width=2.8cm, minimum height=1.1cm, align=center, font=\small},
    stage/.style={rectangle, rounded corners=6pt, draw=#1!80!black, fill=#1!8,
                  inner sep=6pt, thick, font=\footnotesize\bfseries},
    arrow/.style={->, >=Stealth, thick, draw=gray},
]

% ── Row 1: Grid → Launch → Train ──
\node[stage=blue] (s1) {\textcolor{blue!60!black}{1. Grid Construction}};
\node[right=1.2cm of s1, stage=orange] (s2) {\textcolor{orange!80!black}{2. Multi-GPU Launch}};
\node[right=1.2cm of s2, stage=green] (s3) {\textcolor{green!60!black}{3. Per-Config Training}};

\draw[arrow] (s1) -- (s2);
\draw[arrow] (s2) -- (s3);

% ── Details under each stage ──
\node[below=0.15cm of s1, box=blue, text width=3.8cm, font=\scriptsize\raggedright] (s1d) {
    solimp = [$d_{\min}$, $d_{\max}$, width] \\
    solref = [timeconst, dampratio] \\[2pt]
    6 $\times$ 1 $\times$ 7 $\times$ 3 = 126 \\
    \textbf{57 valid} (filter $d_{\min} \le d_{\max}$)
};

\node[below=0.15cm of s2, box=orange, text width=4.2cm, font=\scriptsize\raggedright] (s2d) {
    4 GPUs $\times$ N slots round-robin \\
    Dynamic scheduling (JIT\_READY) \\
    Persistent JIT cache sharing
};

\node[below=0.15cm of s3, box=green, text width=4.5cm, font=\scriptsize\raggedright] (s3d) {
    Train env: swept solimp/solref \\
    Eval env: fixed hard-contact params \\
    PPO 10 iters \texttt{→} render video \texttt{→} log W\&B
};

% ── Row 2: Aggregate ← Analyze ──
\node[below=1.8cm of s3d, stage=purple] (s4) {\textcolor{purple!70!black}{4. Aggregation}};
\node[left=1.2cm of s4, stage=teal] (s5) {\textcolor{teal!70!black}{5. Analysis \& Visualization}};

\draw[arrow] (s3d.south) |- (s4.east);
\draw[arrow] (s4) -- (s5);

% ── Details ──
\node[below=0.15cm of s4, box=purple, text width=5.5cm, font=\scriptsize\raggedright] (s4d) {
    Parse slot logs \texttt{→} extract W\&B summary JSON \\
    Output: \texttt{sweep\_results.csv} (reward, IQM, penetration, ...)
};

\node[below=0.15cm of s5, box=teal, text width=6.5cm, font=\scriptsize\raggedright] (s5d) {
    Marginal effects (boxplots) \texttt{|} Factor main effects \\
    3D reward surfaces \texttt{|} solref sensitivity trends \\
    Softness calibration (ball-drop \texttt{→} rank vs reward) \\
    Heatmap grids \texttt{|} Collapsed 2D trends
};

\end{tikzpicture}
}
\caption{Parameter sweep pipeline. We construct a grid of 57 MuJoCo contact
parameter combinations (solimp, solref), distribute training across 4 GPUs with
dynamic scheduling, and analyze the resulting policies through marginal effects
plots, 3D reward surfaces, and softness-calibrated rankings.}
\label{fig:sweep_pipeline}
\end{figure}
""")


# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--style", choices=["horizontal", "vertical", "tikz"],
                        default="horizontal")
    args = parser.parse_args()

    if args.style == "tikz":
        print_tikz()
        return

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })

    if args.style == "vertical":
        draw_vertical()
    else:
        draw_horizontal()


if __name__ == "__main__":
    main()
