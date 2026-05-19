"""Plot Go2 solimp+solref sweep results.

Reads eval/episode_reward from wandb summary JSONs.
Only uses the latest sweep timestamp.
Includes softness-based plots using ball-drop calibration.
"""

import argparse
import csv
import glob
import itertools
import json
import os
import re
import statistics

import matplotlib.pyplot as plt
import numpy as np

LOG_DIR = os.path.join(os.path.dirname(__file__), "..", "logs", "go2_sweep")
WANDB_DIR = os.path.join(os.path.dirname(__file__), "..", "wandb")
FIGURE_DIR = os.path.join(LOG_DIR, "figures")

BASE_SOLIMP0_VALUES = [0.015, 0.9]
SOLIMP0_VALUES = [0.015, 0.03, 0.1, 0.35, 0.7, 0.9]
SOLIMP1_VALUES = [0.5, 0.95]
BASE_SOLIMP2_VALUES = [0.03, 0.001, 0.5]
SOLIMP2_VALUES = [0.5, 0.1, 0.05, 0.03, 0.01, 0.006, 0.001]
SOLREF0_VALUES = [0.1, 0.02, 0.004]

LIGHT27_SOLIMP2_VALUES = [0.006, 0.01, 0.05, 0.1]
LIGHT27_FULL_PAIRS = [(0.015, 0.5), (0.015, 0.95)]
LIGHT27_DIAGNOSTIC_PAIR = (0.9, 0.95)
LIGHT27_DIAGNOSTIC_SOLIMP2_VALUES = [0.1]

MID42_SOLIMP0_VALUES = [0.03, 0.1, 0.35, 0.7]
MID42_SOLIMP2_VALUES = [0.03, 0.1]

SOLIMP2_PLOT_VALUES = SOLIMP2_VALUES  # soft to hard


def build_base_grid():
    return [
        combo
        for combo in itertools.product(
            BASE_SOLIMP0_VALUES,
            SOLIMP1_VALUES,
            BASE_SOLIMP2_VALUES,
            SOLREF0_VALUES,
        )
        if combo[0] <= combo[1]
    ]


def build_light27_grid():
    grid = []
    for sr0 in SOLREF0_VALUES:
        for s0, s1 in LIGHT27_FULL_PAIRS:
            for s2 in LIGHT27_SOLIMP2_VALUES:
                grid.append((s0, s1, s2, sr0))
        s0, s1 = LIGHT27_DIAGNOSTIC_PAIR
        for s2 in LIGHT27_DIAGNOSTIC_SOLIMP2_VALUES:
            grid.append((s0, s1, s2, sr0))
    return grid


def build_mid42_grid():
    return [
        combo
        for combo in itertools.product(
            MID42_SOLIMP0_VALUES,
            SOLIMP1_VALUES,
            MID42_SOLIMP2_VALUES,
            SOLREF0_VALUES,
        )
        if combo[0] <= combo[1]
    ]


GRID = build_base_grid() + build_light27_grid() + build_mid42_grid()

SOLIMP2_TICK_LABELS = [
    "0.5\nsoft",
    "0.1",
    "0.05",
    "0.03\nmid",
    "0.01",
    "0.006",
    "0.001\nhard",
]
SOLREF0_TICK_LABELS = ["0.1\nsoft", "0.02\nmid", "0.004\nstiff"]
PAIR_MARKERS = ["o", "s", "^", "D", "P", "X", "v", "<", ">"]

# Softness ranking from ball-drop calibration
SOFTNESS_CSV_CANDIDATES = [
    os.path.join(LOG_DIR, "softness_ranking_augmented_mid42.csv"),
    os.path.join(LOG_DIR, "softness_ranking_augmented_light36.csv"),
    os.path.join(LOG_DIR, "softness_ranking.csv"),
]
SOFTNESS_CSV = next(
    (path for path in SOFTNESS_CSV_CANDIDATES if os.path.exists(path)),
    SOFTNESS_CSV_CANDIDATES[-1],
)
_LEGACY_SUMMARY_CACHE = {}


def load_results(algo="apg"):
    pattern = os.path.join(LOG_DIR, f"{algo}_*_gpu*_slot*.log")
    latest_logs = {}
    for log_path in sorted(glob.glob(pattern)):
        slot_m = re.search(r"_slot(\d+)", os.path.basename(log_path))
        slot = int(slot_m.group(1))
        if slot >= len(GRID):
            continue
        ts_m = re.search(rf"{algo}_(\d{{8}}_\d{{6}})_", os.path.basename(log_path))
        ts = ts_m.group(1) if ts_m else ""

        with open(log_path, encoding="utf-8", errors="replace") as f:
            text = f.read()
        if "video done" not in text:
            continue

        prev = latest_logs.get(slot)
        if prev is None or (ts, log_path) > (prev[0], prev[1]):
            latest_logs[slot] = (ts, log_path, text)

    by_slot = {}
    for slot in sorted(latest_logs):
        ts, log_path, text = latest_logs[slot]
        summary = _load_wandb_summary(text)
        if not summary:
            summary = _load_legacy_summary(algo, slot)
            if not summary:
                continue

        s0, s1, s2, sr0 = GRID[slot]
        record = {
            "solimp0": s0, "solimp1": s1, "solimp2": s2, "solref0": sr0,
            "slot": slot, "path": os.path.basename(log_path),
            "eval/episode_reward": summary.get("eval/episode_reward"),
            "eval/iqm_episode_reward": summary.get("eval/iqm_episode_reward"),
            "eval/trimmed_episode_reward": summary.get("eval/trimmed_episode_reward"),
            "eval/avg_episode_length": summary.get("eval/avg_episode_length"),
            "eval/episode_ang_vel_xy": summary.get("eval/episode_ang_vel_xy"),
            "eval/episode_action_rate": summary.get("eval/episode_action_rate"),
            "max_penetration_depth": (
                summary.get("max_penetration_depth") or _parse_max_penetration(text)
            ),
        }
        record["_ts"] = ts
        record["_path"] = log_path
        by_slot[slot] = record

    return [by_slot[s] for s in sorted(by_slot)]


def _parse_json_value(value):
    try:
        return json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return value


def _summary_item_key(item):
    if item.nested_key:
        return "/".join(item.nested_key)
    return item.key


def _load_wandb_file_summary(wandb_file):
    """Read the latest summary values from a local offline .wandb file."""
    try:
        from wandb.proto import wandb_internal_pb2
        from wandb.sdk.internal.datastore import DataStore
    except ImportError:
        return {}

    summary = {}
    datastore = DataStore()
    try:
        datastore.open_for_scan(wandb_file)
        while True:
            data = datastore.scan_data()
            if data is None:
                break
            record = wandb_internal_pb2.Record()
            record.ParseFromString(data)
            record_type = record.WhichOneof("record_type")
            if record_type == "summary":
                for update in record.summary.update:
                    key = _summary_item_key(update)
                    if key:
                        summary[key] = _parse_json_value(update.value_json)
            elif record_type == "history":
                for item in record.history.item:
                    key = _summary_item_key(item)
                    if key:
                        summary[key] = _parse_json_value(item.value_json)
    except OSError:
        return {}
    return summary


def _find_wandb_run_dirs(text):
    """Return candidate local wandb run directories referenced by a log."""
    run_dirs = []

    run_m = re.search(r"runs/([A-Za-z0-9]+)", text)
    if run_m:
        run_id = run_m.group(1)
        run_dirs.extend(glob.glob(os.path.join(WANDB_DIR, f"run-*-{run_id}")))
        run_dirs.extend(glob.glob(os.path.join(WANDB_DIR, f"offline-run-*-{run_id}")))

    for run_dir in re.findall(r"Run data is saved locally in ([^\n\r]+)", text):
        run_dirs.append(run_dir.strip())
    for run_dir in re.findall(r"wandb sync ([^\n\r]+)", text):
        run_dirs.append(run_dir.strip())

    seen = set()
    unique = []
    for run_dir in run_dirs:
        run_dir = os.path.abspath(run_dir)
        if run_dir in seen:
            continue
        seen.add(run_dir)
        unique.append(run_dir)
    return unique


def _load_wandb_summary(text):
    """Load W&B summary from json summaries or offline .wandb records."""
    for run_dir in _find_wandb_run_dirs(text):
        json_path = os.path.join(run_dir, "files", "wandb-summary.json")
        if os.path.exists(json_path):
            with open(json_path) as f:
                return json.load(f)

        wandb_files = glob.glob(os.path.join(run_dir, "run-*.wandb"))
        for wandb_file in wandb_files:
            summary = _load_wandb_file_summary(wandb_file)
            if summary:
                return summary

    return {}


def _parse_max_penetration(text):
    match = re.search(r"video done \(max penetration: ([0-9.]+)\)", text)
    if match is None:
        return None
    return float(match.group(1))


def _load_legacy_summary(algo, slot):
    """Load metrics for old synced runs whose local W&B summaries are absent."""
    if algo not in _LEGACY_SUMMARY_CACHE:
        path = os.path.join(LOG_DIR, f"{algo}_legacy_base27_results.csv")
        rows_by_slot = {}
        if os.path.exists(path):
            with open(path, newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    parsed = {}
                    for key, value in row.items():
                        if key in ("path", ""):
                            parsed[key] = value
                        elif value == "":
                            parsed[key] = None
                        else:
                            parsed[key] = float(value)
                    rows_by_slot[int(parsed["slot"])] = parsed
        _LEGACY_SUMMARY_CACHE[algo] = rows_by_slot

    row = _LEGACY_SUMMARY_CACHE[algo].get(slot)
    if row is None:
        return {}
    return row


def save_csv(results, algo="apg"):
    os.makedirs(FIGURE_DIR, exist_ok=True)
    path = os.path.join(FIGURE_DIR, f"{algo}_go2_sweep_results.csv")
    fieldnames = [
        "solimp0", "solimp1", "solimp2", "solref0", "slot",
        "eval/episode_reward", "eval/iqm_episode_reward",
        "eval/trimmed_episode_reward", "eval/avg_episode_length",
        "eval/episode_ang_vel_xy", "eval/episode_action_rate",
        "max_penetration_depth", "path",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in sorted(results, key=lambda r: r["eval/episode_reward"] or -1e9, reverse=True):
            writer.writerow(row)
    return path


def plot_ranked(results, algo="apg", metric="eval/episode_reward"):
    sorted_results = sorted(results, key=lambda r: r[metric] or -1e9, reverse=True)
    labels = [
        f"si=[{r['solimp0']:.3f},{r['solimp1']:.3f},{r['solimp2']:.3f}] sr={r['solref0']:.3f}"
        for r in sorted_results
    ]
    values = [r[metric] or 0 for r in sorted_results]
    vmin, vmax = min(values), max(values)
    colors = [plt.cm.RdYlGn(0.15 + 0.7 * (v - vmin) / max(vmax - vmin, 1e-6)) for v in values]

    fig, ax = plt.subplots(figsize=(12, 8), constrained_layout=True)
    ax.barh(range(len(values)), values, color=colors)
    ax.set_yticks(range(len(values)), labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel(metric)
    ax.set_title(f"{algo.upper()} Go2 sweep — ranked by {metric}")
    ax.grid(axis="x", alpha=0.2)
    for i, v in enumerate(values):
        ax.text(v + 0.15, i, f"{v:.2f}", va="center", fontsize=7.5)
    p = os.path.join(FIGURE_DIR, f"{algo}_go2_ranked.png")
    fig.savefig(p)
    plt.close(fig)
    return p


def plot_main_effects(results, algo="apg", metric="eval/episode_reward"):
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.8), constrained_layout=True)
    params = [
        ("solimp0", SOLIMP0_VALUES, "solimp[0]"),
        ("solimp1", SOLIMP1_VALUES, "solimp[1]"),
        ("solimp2", SOLIMP2_PLOT_VALUES, "solimp[2]"),
        ("solref0", SOLREF0_VALUES, "solref[0]"),
    ]
    for ax, (key, levels, title) in zip(axes, params):
        grouped = [
            [r[metric] for r in results
             if abs(r[key] - lv) < 1e-9 and r[metric] is not None]
            for lv in levels
        ]
        filtered = [xs for xs in grouped if xs]
        if not filtered:
            continue
        xpos = [i + 1 for i, xs in enumerate(grouped) if xs]
        bp = ax.boxplot(filtered, positions=xpos, patch_artist=True, showmeans=True)
        for patch in bp["boxes"]:
            patch.set_facecolor("#d9e8ff")
            patch.set_edgecolor("#3b5b8a")
        for median in bp["medians"]:
            median.set_color("#1f1f1f")
        means = [statistics.mean(xs) for xs in filtered]
        ax.plot(xpos, means, color="#c03a2b", marker="o", linewidth=2)
        ax.set_title(title)
        ax.set_xticks(range(1, len(levels) + 1), [f"{v:.3f}" for v in levels])
        ax.grid(axis="y", alpha=0.2)
        ax.set_ylabel(metric)
    fig.suptitle(f"{algo.upper()} Go2 sweep — marginal effects")
    p = os.path.join(FIGURE_DIR, f"{algo}_go2_main_effects.png")
    fig.savefig(p)
    plt.close(fig)
    return p


def plot_factor_boxplots(results, algo="apg", metric="eval/episode_reward"):
    """Show main effects as grouped boxplots with all sweep points."""
    specs = [
        ("solimp0", SOLIMP0_VALUES, "solimp[0]", [f"{v:.3f}" for v in SOLIMP0_VALUES]),
        ("solimp1", SOLIMP1_VALUES, "solimp[1]", [f"{v:.3f}" for v in SOLIMP1_VALUES]),
        ("solimp2", SOLIMP2_PLOT_VALUES, "solimp[2]", SOLIMP2_TICK_LABELS),
        ("solref0", SOLREF0_VALUES, "solref[0]", SOLREF0_TICK_LABELS),
    ]
    colors = ["#4c78a8", "#f58518", "#54a24b", "#b279a2"]
    rng = np.random.default_rng(7)

    fig, axes = plt.subplots(
        2, 2, figsize=(12.5, 8.0), sharey=True, constrained_layout=True
    )

    for ax, (key, levels, title, tick_labels), color in zip(
        axes.ravel(), specs, colors
    ):
        grouped = [
            [
                r[metric]
                for r in results
                if abs(r[key] - level) < 1e-9 and r[metric] is not None
            ]
            for level in levels
        ]
        data = [vals for vals in grouped if vals]
        positions = [i + 1 for i, vals in enumerate(grouped) if vals]
        labels = [tick_labels[i] for i, vals in enumerate(grouped) if vals]
        if not data:
            ax.set_title(f"{title} (no data)")
            continue

        bp = ax.boxplot(
            data,
            positions=positions,
            widths=0.56,
            patch_artist=True,
            showmeans=True,
            meanprops={
                "marker": "D",
                "markerfacecolor": "#111111",
                "markeredgecolor": "#111111",
                "markersize": 4,
            },
        )
        for patch in bp["boxes"]:
            patch.set_facecolor(color)
            patch.set_alpha(0.35)
            patch.set_edgecolor(color)
        for median in bp["medians"]:
            median.set_color("#111111")
            median.set_linewidth(1.5)

        means = [statistics.mean(vals) for vals in data]
        ax.plot(positions, means, color=color, marker="o", linewidth=1.8)

        for x_pos, vals in zip(positions, data):
            jitter = rng.uniform(-0.08, 0.08, len(vals))
            ax.scatter(
                np.full(len(vals), x_pos) + jitter,
                vals,
                s=24,
                color=color,
                edgecolors="black",
                linewidths=0.35,
                alpha=0.8,
                zorder=3,
            )

        ax.set_title(title)
        ax.set_xticks(positions, labels)
        ax.grid(axis="y", alpha=0.22)

    axes[0, 0].set_ylabel(metric)
    axes[1, 0].set_ylabel(metric)
    fig.suptitle(
        f"{algo.upper()} Go2 sweep - factor main effects / grouped boxplots"
    )
    p = os.path.join(FIGURE_DIR, f"{algo}_go2_factor_boxplots.png")
    fig.savefig(p)
    plt.close(fig)
    return p


def plot_heatmap(results, algo="apg", metric="eval/episode_reward"):
    values = np.array([r[metric] or 0 for r in results])
    vmin, vmax = values.min(), values.max()

    fig, axes = plt.subplots(
        len(SOLIMP2_PLOT_VALUES), len(SOLREF0_VALUES),
        figsize=(4 * len(SOLREF0_VALUES), 3.5 * len(SOLIMP2_PLOT_VALUES)),
        sharex=True, sharey=True, constrained_layout=True,
    )

    for i, s2 in enumerate(SOLIMP2_PLOT_VALUES):
        for j, sr0 in enumerate(SOLREF0_VALUES):
            ax = axes[i, j] if len(SOLIMP2_PLOT_VALUES) > 1 else axes[j]
            grid = np.full((len(SOLIMP1_VALUES), len(SOLIMP0_VALUES)), np.nan)
            for mi, s0 in enumerate(SOLIMP0_VALUES):
                for mj, s1 in enumerate(SOLIMP1_VALUES):
                    if s0 > s1:
                        continue
                    match = [r for r in results
                             if abs(r["solimp0"] - s0) < 1e-9
                             and abs(r["solimp1"] - s1) < 1e-9
                             and abs(r["solimp2"] - s2) < 1e-9
                             and abs(r["solref0"] - sr0) < 1e-9]
                    if match and match[0][metric] is not None:
                        grid[mj, mi] = match[0][metric]

            im = ax.imshow(
                grid,
                cmap="RdYlGn",
                vmin=vmin,
                vmax=vmax,
                origin="lower",
                extent=[
                    -0.5,
                    len(SOLIMP0_VALUES) - 0.5,
                    -0.5,
                    len(SOLIMP1_VALUES) - 0.5,
                ],
            )
            for mi, s0 in enumerate(SOLIMP0_VALUES):
                for mj, s1 in enumerate(SOLIMP1_VALUES):
                    if s0 > s1:
                        ax.text(mi, mj, "X", ha="center", va="center", fontsize=14, color="gray")
                        continue
                    v = grid[mj, mi]
                    if not np.isnan(v):
                        tc = "white" if v < vmin + 0.35 * (vmax - vmin) else "black"
                        ax.text(mi, mj, f"{v:.1f}", ha="center", va="center", fontsize=11, color=tc)
            ax.set_title(f"solimp[2]={s2:.3f}  solref[0]={sr0:.3f}")
            ax.set_xticks(range(len(SOLIMP0_VALUES)), [f"{v:.3f}" for v in SOLIMP0_VALUES])
            ax.set_yticks(range(len(SOLIMP1_VALUES)), [f"{v:.3f}" for v in SOLIMP1_VALUES])
            if j == 0:
                ax.set_ylabel("solimp[1]")
            if i == len(SOLIMP2_PLOT_VALUES) - 1:
                ax.set_xlabel("solimp[0]")

    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.85)
    cbar.set_label(metric)
    fig.suptitle(f"{algo.upper()} Go2 sweep — solimp[0] × solimp[1], faceted by solimp[2], solref[0]")
    p = os.path.join(FIGURE_DIR, f"{algo}_go2_heatmap.png")
    fig.savefig(p)
    plt.close(fig)
    return p


def plot_best_solref(results, algo="apg", metric="eval/episode_reward"):
    """For each solimp combo, show how reward varies with solref[0]."""
    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)

    x = np.arange(len(SOLREF0_VALUES))
    # Group by solimp triplet
    solimp_triplets = sorted(
        {(r["solimp0"], r["solimp1"], r["solimp2"]) for r in results}
    )
    for idx, (s0, s1, s2) in enumerate(solimp_triplets):
        series = []
        for sr0 in SOLREF0_VALUES:
            match = [r[metric] for r in results
                     if abs(r["solimp0"] - s0) < 1e-9
                     and abs(r["solimp1"] - s1) < 1e-9
                     and abs(r["solimp2"] - s2) < 1e-9
                     and abs(r["solref0"] - sr0) < 1e-9
                     and r[metric] is not None]
            series.append(match[0] if match else np.nan)
        label = f"si=[{s0:.3f},{s1:.3f},{s2:.3f}]"
        ax.plot(x, series, marker="o", linewidth=1.5, alpha=0.85, label=label)

    ax.set_xticks(x, SOLREF0_TICK_LABELS)
    ax.set_xlabel("solref[0] (timeconst)")
    ax.set_ylabel(metric)
    ax.legend(fontsize=7.5, frameon=False, ncol=2)
    ax.grid(axis="y", alpha=0.2)
    ax.set_title(f"{algo.upper()} Go2 sweep — solref[0] sensitivity per solimp combo")
    p = os.path.join(FIGURE_DIR, f"{algo}_go2_solref_trend.png")
    fig.savefig(p)
    plt.close(fig)
    return p


def plot_solimp2_trend(results, algo="apg"):
    """solimp[2] trend, averaging over solref[0]."""
    fig, axes = plt.subplots(1, 2, figsize=(13.8, 4.8), sharex=True, constrained_layout=True)

    solimp_pairs = sorted(
        {(r["solimp0"], r["solimp1"]) for r in results
         if r["eval/episode_reward"] is not None}
    )
    x = np.arange(len(SOLIMP2_PLOT_VALUES))
    specs = [
        ("eval/episode_reward", "eval/episode_reward"),
        ("eval/avg_episode_length", "eval/avg_episode_length"),
    ]

    for ax, (metric, ylabel) in zip(axes, specs):
        if metric == "eval/episode_reward":
            for idx, (s0, s1) in enumerate(solimp_pairs):
                # Average over solref0
                ys = []
                for s2 in SOLIMP2_PLOT_VALUES:
                    vals = [r[metric] for r in results
                            if abs(r["solimp0"] - s0) < 1e-9
                            and abs(r["solimp1"] - s1) < 1e-9
                            and abs(r["solimp2"] - s2) < 1e-9
                            and r[metric] is not None]
                    ys.append(statistics.mean(vals) if vals else np.nan)
                label = f"si=[{s0:.3f},{s1:.3f}]"
                color = plt.cm.tab20(idx % 20)
                marker = PAIR_MARKERS[idx % len(PAIR_MARKERS)]
                ax.plot(x, ys, color=color, marker=marker,
                        linewidth=1.8, alpha=0.88, label=label)

        # Overall mean per solimp2
        grouped = [
            [r[metric] for r in results
             if abs(r["solimp2"] - s2) < 1e-9 and r[metric] is not None]
            for s2 in SOLIMP2_PLOT_VALUES
        ]
        means = [statistics.mean(xs) if xs else np.nan for xs in grouped]
        lows = [min(xs) if xs else np.nan for xs in grouped]
        highs = [max(xs) if xs else np.nan for xs in grouped]
        valid = ~np.isnan(np.array(means))

        ax.fill_between(x, lows, highs, color="#666666", alpha=0.08, zorder=0)
        ax.plot(x, means, color="#111111", marker="D", linewidth=2.4, zorder=3)
        if valid.any():
            best_idx = int(np.nanargmax(means))
            ax.scatter([best_idx], [means[best_idx]], marker="*", s=160,
                       color="gold", edgecolors="black", linewidths=0.6,
                       zorder=4)

        for xi, yi in zip(x, means):
            if np.isnan(yi):
                continue
            fmt = "{:.1f}" if metric == "eval/episode_reward" else "{:.0f}"
            ax.annotate(fmt.format(yi), (xi, yi), xytext=(0, 8),
                        textcoords="offset points", ha="center", fontsize=7,
                        color="#111111")

        ax.set_xticks(x, SOLIMP2_TICK_LABELS)
        ax.set_xlabel("solimp[2]")
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.2)
        ax.set_title("reward" if metric == "eval/episode_reward" else "episode length")

    axes[0].legend(loc="lower left", fontsize=8, frameon=False)
    fig.suptitle(f"{algo.upper()} Go2 sweep — solimp[2] trend", fontsize=15)
    p = os.path.join(FIGURE_DIR, f"{algo}_go2_solimp2_trend.png")
    fig.savefig(p)
    plt.close(fig)
    return p


def plot_3d_grid(results, algo="apg", metric="eval/episode_reward"):
    """3D scatter: axes = solimp[0], solimp[1], solimp[2], color = solref[0]."""
    from matplotlib.colors import Normalize

    valid = [r for r in results if r[metric] is not None]
    xs = np.array([r["solimp0"] for r in valid])
    ys = np.array([r["solimp1"] for r in valid])
    zs = np.array([r["solimp2"] for r in valid])
    vs = np.array([r[metric] for r in valid])
    srs = np.array([r["solref0"] for r in valid])

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(1, 1, 1, projection="3d")

    norm = Normalize(vmin=srs.min(), vmax=srs.max())
    sc = ax.scatter(xs, ys, zs, c=vs, s=120, cmap="RdYlGn", edgecolors="black", linewidths=0.3)

    # Add solref labels using different marker edge styles would be complex,
    # so color-code solref separately via the scatter cmap based on reward.

    for x, y, z, v, sr in zip(xs, ys, zs, vs, srs):
        ax.text(x, y, z + 0.02, f"{v:.1f}", ha="center", va="bottom", fontsize=7)

    cbar = fig.colorbar(sc, ax=ax, shrink=0.5, pad=0.1)
    cbar.set_label(metric)

    ax.set_xlabel("solimp[0]")
    ax.set_ylabel("solimp[1]")
    ax.set_zlabel("solimp[2]")

    ax.set_xlim(min(SOLIMP0_VALUES) - 0.05, max(SOLIMP0_VALUES) + 0.05)
    ax.set_ylim(min(SOLIMP1_VALUES) - 0.05, max(SOLIMP1_VALUES) + 0.05)
    ax.set_zlim(min(SOLIMP2_VALUES) - 0.05, max(SOLIMP2_VALUES) + 0.05)

    ax.set_title(f"{algo.upper()} Go2 sweep — 4D grid (color=reward)")
    ax.view_init(elev=21, azim=-50)

    p = os.path.join(FIGURE_DIR, f"{algo}_go2_3dgrid.png")
    fig.savefig(p)
    plt.close(fig)
    return p


def load_softness():
    """Return { (solimp0, solimp1, solimp2, solref0): (rank, penetration_m) }."""
    mapping = {}
    if not os.path.exists(SOFTNESS_CSV):
        print(f"Softness CSV not found: {SOFTNESS_CSV}, skipping softness plots.")
        return mapping
    with open(SOFTNESS_CSV, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (
                float(row["solimp0"]),
                float(row["solimp1"]),
                float(row["solimp2"]),
                float(row["solref0"]),
            )
            mapping[key] = (int(row["softness_rank"]), float(row["penetration_m"]))
    return mapping


def _match_key(r):
    """Return the softness lookup key for a result row."""
    return (r["solimp0"], r["solimp1"], r["solimp2"], r["solref0"])


def _solref_label(sr0):
    if abs(sr0 - 0.1) < 1e-9:
        return "0.1 (soft)"
    if abs(sr0 - 0.02) < 1e-9:
        return "0.02 (mid)"
    return "0.004 (stiff)"


def plot_softness_rank_vs_reward(results, algo="apg", metric="eval/episode_reward"):
    """X = softness rank (1=softest), Y = reward. Simple scatter+line."""
    softness = load_softness()
    if not softness:
        return None

    points = []
    for r in results:
        key = _match_key(r)
        if key in softness and r[metric] is not None:
            rank, pen_mm = softness[key]
            sr_label = _solref_label(r["solref0"])
            points.append((rank, r[metric], r["solimp0"], r["solimp1"], r["solimp2"],
                          r["solref0"], sr_label, pen_mm * 1000))

    ranks = [p[0] for p in points]
    rewards = [p[1] for p in points]
    sr_labels = [p[6] for p in points]
    if not points:
        return None
    max_rank = max(rank for rank, _ in softness.values())
    tick_step = max(1, max_rank // 14)

    sr_colors = {"0.1 (soft)": "#d95f02", "0.02 (mid)": "#7570b3", "0.004 (stiff)": "#1b9e77"}

    fig, ax = plt.subplots(figsize=(14, 7), constrained_layout=True)

    for label, color in sr_colors.items():
        idxs = [i for i, lb in enumerate(sr_labels) if lb == label]
        if not idxs:
            continue
        xs = [ranks[i] for i in idxs]
        ys = [rewards[i] for i in idxs]
        ax.scatter(xs, ys, c=color, s=70, alpha=0.85, edgecolors="black",
                   linewidths=0.3, label=label, zorder=3)

    # Annotate each point with its params
    for rank, rew, s0, s1, s2, sr0 in [
        (p[0], p[1], p[2], p[3], p[4], p[5]) for p in points
    ]:
        label = f"[{s0:.3f},{s1:.3f},{s2:.3f}|{sr0:.3f}]"
        ax.annotate(label, (rank, rew), textcoords="offset points",
                   xytext=(5, 5), fontsize=5.5, color="#444444",
                   rotation=25, alpha=0.85)

    # Trend line
    sorted_pairs = sorted(zip(ranks, rewards))
    ax.plot([s[0] for s in sorted_pairs], [s[1] for s in sorted_pairs],
            color="#666666", linewidth=1, alpha=0.5, zorder=2)

    ax.set_xlabel(
        f"train-env softness rank  (1 = softest, {max_rank} = hardest)"
    )
    ax.set_ylabel(metric)
    ax.legend(title="train solref[0]", fontsize=8)
    ax.grid(alpha=0.2)
    ax.set_xticks(range(1, max_rank + 1, tick_step))
    ax.set_title(f"{algo.upper()} Go2 sweep — reward vs train-env softness rank  [si0,si1,si2|sr0]")
    p = os.path.join(FIGURE_DIR, f"{algo}_go2_softness_rank_vs_reward.png")
    fig.savefig(p)
    plt.close(fig)
    return p


def plot_softness_true_spacing_vs_reward(results, algo="apg", metric="eval/episode_reward"):
    """X = calibrated max penetration depth, ordered hard to soft."""
    softness = load_softness()
    if not softness:
        return None

    rank_positions = []
    for key, (rank, pen_m) in softness.items():
        rank_positions.append((rank, pen_m * 1000, key[3]))
    if not rank_positions:
        return None

    points = []
    for r in results:
        key = _match_key(r)
        if key in softness and r[metric] is not None:
            rank, pen_m = softness[key]
            pen_mm = pen_m * 1000
            sr_label = _solref_label(r["solref0"])
            points.append((
                rank, pen_mm, pen_mm, r[metric], r["solimp0"],
                r["solimp1"], r["solimp2"], r["solref0"], sr_label,
            ))

    if not points:
        return None

    sr_colors = {
        "0.1 (soft)": "#d95f02",
        "0.02 (mid)": "#7570b3",
        "0.004 (stiff)": "#1b9e77",
    }
    sr_value_colors = {
        0.1: "#d95f02",
        0.02: "#7570b3",
        0.004: "#1b9e77",
    }

    fig, (ax_rank, ax) = plt.subplots(
        2, 1, figsize=(14, 8), sharex=True, constrained_layout=True,
        gridspec_kw={"height_ratios": [0.9, 4.0]},
    )

    sorted_rank_positions = sorted(rank_positions)
    xs_all = [pen_mm for _, pen_mm, _ in sorted_rank_positions]
    xmin, xmax = min(xs_all), max(xs_all)
    span = max(xmax - xmin, 1e-6)
    ax_rank.hlines(0, xmin, xmax, color="#8a8a8a", linewidth=1.0, zorder=1)
    for rank, pen_mm, sr0 in sorted_rank_positions:
        x = pen_mm
        color = sr_value_colors.get(sr0, "#666666")
        label_y = 0.22 + 0.09 * ((rank - 1) % 3)
        ax_rank.vlines(x, 0, label_y - 0.02, color=color, linewidth=0.8, alpha=0.55)
        ax_rank.scatter([x], [0], c=color, s=28, edgecolors="black",
                        linewidths=0.25, zorder=3)
        ax_rank.text(x, label_y, str(rank), ha="center", va="bottom",
                     fontsize=6.5, rotation=90, color="#333333")
    ax_rank.set_ylim(-0.08, 0.56)
    ax_rank.set_yticks([])
    ax_rank.set_ylabel("rank", rotation=0, labelpad=24)
    ax_rank.set_title(
        "softness ranks positioned by ball-drop max penetration (hard -> soft)"
    )
    ax_rank.grid(False)
    for spine in ["left", "right", "top"]:
        ax_rank.spines[spine].set_visible(False)

    sorted_points = sorted(points, key=lambda p: p[1])
    ax.plot([p[1] for p in sorted_points], [p[3] for p in sorted_points],
            color="#666666", linewidth=1, alpha=0.45, zorder=2)

    sr_labels = [p[8] for p in points]
    for label, color in sr_colors.items():
        idxs = [i for i, lb in enumerate(sr_labels) if lb == label]
        if not idxs:
            continue
        xs = [points[i][1] for i in idxs]
        ys = [points[i][3] for i in idxs]
        ax.scatter(xs, ys, c=color, s=70, alpha=0.85, edgecolors="black",
                   linewidths=0.3, label=label, zorder=3)

    for rank, x, _, rew, *_ in sorted_points:
        ax.annotate(f"#{rank}", (x, rew), textcoords="offset points",
                    xytext=(4, 4), fontsize=6, color="#444444", alpha=0.85)

    ax.set_xlim(xmin - 0.02 * span, xmax + 0.02 * span)
    ax.set_xlabel("ball-drop max penetration depth (mm): hard -> soft")
    ax.set_ylabel(metric)
    ax.legend(title="train solref[0]", fontsize=8)
    ax.grid(alpha=0.2)
    ax.set_title(
        f"{algo.upper()} Go2 sweep — reward vs calibrated softness"
    )

    p = os.path.join(FIGURE_DIR, f"{algo}_go2_softness_true_spacing_vs_reward.png")
    fig.savefig(p)
    plt.close(fig)
    return p


def _solref_filename_value(sr0):
    return f"{sr0:g}".replace(".", "p")


def _dodged_x_positions(sorted_points, span):
    """Return lightly dodged x positions for dense scatter readability."""
    if not sorted_points:
        return []

    cluster_width = max(0.10, min(0.45, 0.025 * span))
    dodge_step = max(0.05, min(0.22, 0.012 * span))
    lanes = np.array([0, -1, 1, -2, 2, -3, 3], dtype=float) * dodge_step

    display_x = [0.0] * len(sorted_points)
    cluster = [0]
    for i in range(1, len(sorted_points)):
        prev_x = sorted_points[i - 1][1]
        this_x = sorted_points[i][1]
        if this_x - prev_x <= cluster_width:
            cluster.append(i)
            continue

        for lane_i, point_i in enumerate(cluster):
            display_x[point_i] = sorted_points[point_i][1] + lanes[lane_i % len(lanes)]
        cluster = [i]

    for lane_i, point_i in enumerate(cluster):
        display_x[point_i] = sorted_points[point_i][1] + lanes[lane_i % len(lanes)]

    return display_x


def plot_softness_true_spacing_by_solref(
    results, algo="apg", metric="eval/episode_reward"
):
    """One reward-vs-penetration plot per solref[0] value."""
    softness = load_softness()
    if not softness:
        return []

    paths = []
    sr_colors = {0.1: "#d95f02", 0.02: "#7570b3", 0.004: "#1b9e77"}
    for sr0 in SOLREF0_VALUES:
        rank_positions = []
        for key, (rank, pen_m) in softness.items():
            if abs(key[3] - sr0) < 1e-9:
                rank_positions.append((rank, pen_m * 1000))
        if not rank_positions:
            continue

        points = []
        for r in results:
            key = _match_key(r)
            if key not in softness or r[metric] is None:
                continue
            if abs(r["solref0"] - sr0) >= 1e-9:
                continue
            rank, pen_m = softness[key]
            pen_mm = pen_m * 1000
            points.append((
                rank, pen_mm, pen_mm, r[metric], r["solimp0"],
                r["solimp1"], r["solimp2"],
            ))

        if not points:
            continue

        fig, (ax_rank, ax) = plt.subplots(
            2, 1, figsize=(16, 7.2), sharex=True, constrained_layout=True,
            gridspec_kw={"height_ratios": [0.9, 4.0]},
        )

        sorted_rank_positions = sorted(rank_positions)
        xs_all = [pen_mm for _, pen_mm in sorted_rank_positions]
        xmin, xmax = min(xs_all), max(xs_all)
        span = max(xmax - xmin, 1e-6)
        color = sr_colors[sr0]

        ax_rank.hlines(0, xmin, xmax, color="#8a8a8a", linewidth=1.0, zorder=1)
        for rank, pen_mm in sorted_rank_positions:
            x = pen_mm
            label_y = 0.22 + 0.09 * ((rank - 1) % 3)
            ax_rank.vlines(
                x, 0, label_y - 0.02, color=color, linewidth=0.8, alpha=0.55
            )
            ax_rank.scatter([x], [0], c=color, s=32, edgecolors="black",
                            linewidths=0.25, zorder=3)
            ax_rank.text(x, label_y, str(rank), ha="center", va="bottom",
                         fontsize=6, rotation=90, color="#333333")
        ax_rank.set_ylim(-0.08, 0.56)
        ax_rank.set_yticks([])
        ax_rank.set_ylabel("rank", rotation=0, labelpad=24)
        ax_rank.set_title(
            f"softness ranks for train solref[0]={sr0:g}, "
            "ball-drop max penetration (hard -> soft)"
        )
        ax_rank.grid(False)
        for spine in ["left", "right", "top"]:
            ax_rank.spines[spine].set_visible(False)

        sorted_points = sorted(points, key=lambda p: p[1])
        display_x = _dodged_x_positions(sorted_points, span)
        ax.plot(display_x, [p[3] for p in sorted_points],
                color="#666666", linewidth=0.8, alpha=0.22, zorder=2)
        ax.scatter(display_x, [p[3] for p in sorted_points],
                   c=color, s=46, alpha=0.82, edgecolors="black",
                   linewidths=0.3, label=_solref_label(sr0), zorder=3)

        label_offsets = [(5, 7), (5, 21), (-5, -19), (-5, -7)]
        for i, ((rank, _, pen_mm, rew, s0, s1, s2), x) in enumerate(
            zip(sorted_points, display_x)
        ):
            dx, dy = label_offsets[i % len(label_offsets)]
            label = f"#{rank} [{s0:.3g},{s1:.3g},{s2:.3g}]"
            ax.annotate(
                label,
                (x, rew),
                textcoords="offset points",
                xytext=(dx, dy),
                fontsize=5.4,
                ha="left" if dx > 0 else "right",
                va="bottom" if dy > 0 else "top",
                color="#444444",
                alpha=0.82,
            )

        xmin_plot = min(xmin, min(display_x))
        xmax_plot = max(xmax, max(display_x))
        plot_span = max(xmax_plot - xmin_plot, 1e-6)
        ax.set_xlim(xmin_plot - 0.06 * plot_span, xmax_plot + 0.06 * plot_span)
        ax.set_xlabel(
            f"ball-drop max penetration depth within solref[0]={sr0:g} "
            "(mm): hard -> soft; markers lightly dodged"
        )
        ax.set_ylabel(metric)
        ax.legend(title="train solref[0]", fontsize=8)
        ax.grid(alpha=0.2)
        ax.set_title(
            f"{algo.upper()} Go2 sweep — reward vs calibrated softness "
            f"(solref[0]={sr0:g})"
        )

        sr_name = _solref_filename_value(sr0)
        p = os.path.join(
            FIGURE_DIR,
            f"{algo}_go2_softness_true_spacing_solref_{sr_name}_vs_reward.png",
        )
        fig.savefig(p)
        plt.close(fig)
        paths.append(p)

    return paths


def plot_softness_scatter(results, algo="apg", metric="eval/episode_reward"):
    """Scatter: X = penetration depth (continuous softness), Y = reward."""
    softness = load_softness()
    if not softness:
        return None

    points = []
    for r in results:
        key = _match_key(r)
        if key in softness and r[metric] is not None:
            rank, pen_m = softness[key]
            points.append((pen_m * 1000, r[metric], r["solref0"],
                          r["solimp0"], r["solimp1"], r["solimp2"], rank))

    pen_mm, rewards, srs, _, _, _, ranks = zip(*points)

    fig, ax = plt.subplots(figsize=(9, 5.5), constrained_layout=True)

    sr_groups = {0.1: [], 0.02: [], 0.004: []}
    for i, sr in enumerate(srs):
        sr_groups[sr].append(i)
    sr_colors = {0.1: "#d95f02", 0.02: "#7570b3", 0.004: "#1b9e77"}

    for sr_val, idxs in sr_groups.items():
        if not idxs:
            continue
        xs = [pen_mm[i] for i in idxs]
        ys = [rewards[i] for i in idxs]
        ax.scatter(xs, ys, c=sr_colors[sr_val], s=60, alpha=0.8,
                   edgecolors="black", linewidths=0.3,
                   label=_solref_label(sr_val))
        for i in idxs:
            ax.annotate(str(ranks[i]), (pen_mm[i], rewards[i]),
                       textcoords="offset points", xytext=(3, 3), fontsize=6, color="gray")

    ax.set_xlabel("max penetration depth (mm) — softer →")
    ax.set_ylabel(metric)
    ax.legend(title="train solref[0]", fontsize=8)
    ax.grid(alpha=0.2)
    ax.set_title(f"{algo.upper()} Go2 sweep — policy reward vs train-env softness")
    p = os.path.join(FIGURE_DIR, f"{algo}_go2_softness_scatter.png")
    fig.savefig(p)
    plt.close(fig)
    return p


def plot_softness_ranked(results, algo="apg", metric="eval/episode_reward"):
    """Bar chart: results sorted by softness rank (softest on top)."""
    softness = load_softness()
    if not softness:
        return None

    enriched = []
    for r in results:
        key = _match_key(r)
        if key in softness and r[metric] is not None:
            rank, pen_mm = softness[key]
            enriched.append((rank, pen_mm, r))

    enriched.sort(key=lambda x: x[0])

    labels = [
        f"#{r} si=[{e[2]['solimp0']:.3f},{e[2]['solimp1']:.3f},{e[2]['solimp2']:.3f}] sr={e[2]['solref0']:.3f} ({e[1]*1000:.1f}mm)"
        for e in enriched
    ]
    values = [e[2][metric] for e in enriched]
    pen_mm = [e[1] * 1000 for e in enriched]

    vmin, vmax = min(values), max(values)

    fig, ax = plt.subplots(figsize=(13, 8.5))
    fig.subplots_adjust(left=0.42, right=0.98, top=0.93, bottom=0.08)
    colors = [plt.cm.RdYlGn(0.15 + 0.7 * (v - vmin) / max(vmax - vmin, 1e-6)) for v in values]

    ax.barh(range(len(values)), values, color=colors)
    ax.set_yticks(range(len(values)), labels, fontsize=7.5)
    ax.invert_yaxis()
    ax.set_xlabel(metric)
    ax.set_title(f"{algo.upper()} Go2 sweep — ordered by train-env softness (top=soft, bottom=hard)")
    ax.grid(axis="x", alpha=0.2)
    for i, v in enumerate(values):
        ax.text(v + 0.15, i, f"{v:.1f}", va="center", fontsize=7)
    p = os.path.join(FIGURE_DIR, f"{algo}_go2_softness_ranked.png")
    fig.savefig(p)
    plt.close(fig)
    return p


def plot_softness_vs_reward_by_solref(results, algo="apg", metric="eval/episode_reward"):
    """Connected line: X = softness rank within each solref[0] group, Y = reward."""
    softness = load_softness()
    if not softness:
        return None

    enriched = []
    for r in results:
        key = _match_key(r)
        if key in softness and r[metric] is not None:
            rank, pen_mm = softness[key]
            enriched.append((rank, pen_mm, r))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True, constrained_layout=True)

    for ax, sr_val in zip(axes, [0.1, 0.02, 0.004]):
        group = sorted(
            [e for e in enriched if abs(e[2]["solref0"] - sr_val) < 1e-9],
            key=lambda x: x[0],
        )
        if not group:
            ax.set_title(f"solref[0]={sr_val} (no data)")
            continue

        ranks_in_group = [e[0] for e in group]
        rewards_in_group = [e[2][metric] for e in group]
        labels_in_group = [
            f"si=[{e[2]['solimp0']:.3f},{e[2]['solimp1']:.3f},{e[2]['solimp2']:.3f}]"
            for e in group
        ]

        ax.plot(ranks_in_group, rewards_in_group, "o-", linewidth=1.5, color="#3b5b8a",
                markersize=7, markerfacecolor="white", markeredgewidth=1.2)
        for x, y, lbl in zip(ranks_in_group, rewards_in_group, labels_in_group):
            ax.annotate(lbl, (x, y), textcoords="offset points", xytext=(0, 9),
                       fontsize=6, ha="center", rotation=30, color="gray")
        ax.set_title(f"train solref[0]={sr_val} ({_solref_label(sr_val).split('(')[1].rstrip(')')})")
        ax.set_xlabel("softness rank (lower=softer)")
        ax.grid(alpha=0.2)
        ax.set_ylim(min(rewards_in_group) - 0.5, max(rewards_in_group) + 0.5)

    axes[0].set_ylabel(metric)
    fig.suptitle(f"{algo.upper()} Go2 sweep — reward vs softness within each solref[0] group",
                 fontsize=13)
    p = os.path.join(FIGURE_DIR, f"{algo}_go2_softness_by_solref.png")
    fig.savefig(p)
    plt.close(fig)
    return p


def plot_softness_boxplot(results, algo="apg", metric="eval/episode_reward"):
    """Box plot of reward grouped by softness clusters (solref[0])."""
    softness = load_softness()
    if not softness:
        return None

    groups = {"soft\n(sr=0.1)": [], "mid\n(sr=0.02)": [], "stiff\n(sr=0.004)": []}
    group_colors = ["#d95f02", "#7570b3", "#1b9e77"]

    for r in results:
        key = _match_key(r)
        if r[metric] is None:
            continue
        if abs(r["solref0"] - 0.1) < 1e-9:
            groups["soft\n(sr=0.1)"].append(r[metric])
        elif abs(r["solref0"] - 0.02) < 1e-9:
            groups["mid\n(sr=0.02)"].append(r[metric])
        else:
            groups["stiff\n(sr=0.004)"].append(r[metric])

    fig, ax = plt.subplots(figsize=(6.5, 4.5), constrained_layout=True)

    names, data, colors = [], [], []
    for name, vals in groups.items():
        if vals:
            names.append(name)
            data.append(vals)
            colors.append(group_colors[len(names) - 1])

    bp = ax.boxplot(data, patch_artist=True, showmeans=True)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.45)
    for median in bp["medians"]:
        median.set_color("#1f1f1f")

    # Overlay individual points
    for i, vals in enumerate(data):
        jitter = np.random.default_rng(42).uniform(-0.08, 0.08, len(vals))
        ax.scatter(np.full(len(vals), i + 1) + jitter, vals, s=22,
                  color=colors[i], edgecolors="black", linewidths=0.3, alpha=0.7)

    ax.set_xticklabels(names)
    ax.set_ylabel(metric)
    ax.set_title(f"{algo.upper()} Go2 sweep — reward by train-env softness group")
    ax.grid(axis="y", alpha=0.2)

    # Stats annotation
    for i, vals in enumerate(data):
        mean_v = statistics.mean(vals)
        ax.annotate(f"mean={mean_v:.2f}\nn={len(vals)}", (i + 1, mean_v),
                   textcoords="offset points", xytext=(12, -12), fontsize=7.5, color="black")

    p = os.path.join(FIGURE_DIR, f"{algo}_go2_softness_boxplot.png")
    fig.savefig(p)
    plt.close(fig)
    return p


def _smooth_surface_grid(points, values, grid_x, grid_y, vmin=None, vmax=None):
    """Interpolate sparse sweep samples onto a smooth regular grid."""
    from scipy.interpolate import CloughTocher2DInterpolator
    from scipy.interpolate import NearestNDInterpolator

    interp = CloughTocher2DInterpolator(points, values)
    grid = interp(grid_x, grid_y)

    if np.isnan(grid).any():
        nearest = NearestNDInterpolator(points, values)
        grid = np.where(np.isnan(grid), nearest(grid_x, grid_y), grid)

    if vmin is not None and vmax is not None:
        grid = np.clip(grid, vmin, vmax)
    return grid


def _hillshade_facecolors(facecolors, height_grid):
    """Add slope-based shading without changing the hardness colormap order."""
    from matplotlib.colors import LightSource

    light = LightSource(azdeg=320, altdeg=38)
    shade = light.hillshade(height_grid, vert_exag=0.75, fraction=1.15)
    shade = 0.70 + 0.30 * shade

    shaded = facecolors.copy()
    shaded[..., :3] = np.clip(facecolors[..., :3] * shade[..., None], 0.0, 1.0)
    return shaded


def plot_solimp0_solimp2_reward_surface_by_solref(
    results, algo="apg", metric="eval/episode_reward", fixed_solimp1=0.95
):
    """3D reward surfaces at fixed solimp[1], colored by calibrated hardness."""
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.colors import PowerNorm

    softness = load_softness()
    if not softness:
        return []

    valid = [
        r
        for r in results
        if r[metric] is not None
        and abs(r["solimp1"] - fixed_solimp1) < 1e-9
        and _match_key(r) in softness
    ]
    if not valid:
        return []

    reward_min = min(r[metric] for r in valid)
    reward_max = max(r[metric] for r in valid)
    cmap = LinearSegmentedColormap.from_list(
        "hardness_gray", ["#242424", "#f0f0f0"]
    )

    x_values = sorted({r["solimp0"] for r in valid})
    y_values = sorted({r["solimp2"] for r in valid})
    x_log_values = np.log10(np.array(x_values))
    y_log_values = np.log10(np.array(y_values))
    x_grid_values = np.linspace(min(x_log_values), max(x_log_values), 110)
    y_grid_values = np.linspace(min(y_log_values), max(y_log_values), 110)
    grid_x, grid_y = np.meshgrid(x_grid_values, y_grid_values)

    paths = []
    for sr0 in SOLREF0_VALUES:
        group = [
            r for r in valid if abs(r["solref0"] - sr0) < 1e-9
        ]
        if len(group) < 4:
            continue

        points = np.array([
            [np.log10(r["solimp0"]), np.log10(r["solimp2"])]
            for r in group
        ])
        rewards = np.array([r[metric] for r in group], dtype=float)
        log_pen = np.array([
            np.log10(softness[_match_key(r)][1] * 1000.0) for r in group
        ])
        color_log_pen_min = float(np.min(log_pen))
        color_log_pen_max = float(np.quantile(log_pen, 0.90))
        if color_log_pen_max <= color_log_pen_min:
            color_log_pen_max = float(np.max(log_pen))
        pen_norm = PowerNorm(
            gamma=0.65, vmin=color_log_pen_min, vmax=color_log_pen_max
        )
        mappable = plt.cm.ScalarMappable(norm=pen_norm, cmap=cmap)
        mappable.set_array([])

        reward_grid = _smooth_surface_grid(
            points, rewards, grid_x, grid_y, reward_min, reward_max
        )
        log_pen_grid = _smooth_surface_grid(
            points, log_pen, grid_x, grid_y, color_log_pen_min,
            color_log_pen_max
        )
        facecolors = _hillshade_facecolors(cmap(pen_norm(log_pen_grid)), reward_grid)

        fig = plt.figure(figsize=(10.2, 7.2))
        fig.subplots_adjust(left=0.00, right=0.80, top=0.90, bottom=0.02)
        ax = fig.add_subplot(1, 1, 1, projection="3d")
        ax.plot_surface(
            grid_x,
            grid_y,
            reward_grid,
            facecolors=facecolors,
            rstride=1,
            cstride=1,
            linewidth=0.08,
            edgecolor=(0.12, 0.12, 0.12, 0.12),
            antialiased=True,
            shade=False,
            alpha=0.97,
        )
        ax.contour(
            grid_x,
            grid_y,
            reward_grid,
            zdir="z",
            offset=reward_min - 0.03,
            levels=9,
            colors="#303030",
            linewidths=0.45,
            alpha=0.68,
        )
        ax.set_xlabel("solimp[0] (log scale)")
        ax.set_ylabel("solimp[2] (log scale)")
        ax.set_zlabel(metric)
        ax.set_xlim(min(x_log_values), max(x_log_values))
        ax.set_ylim(min(y_log_values), max(y_log_values))
        ax.set_zlim(reward_min - 0.03, reward_max + 0.03)
        ax.set_xticks(np.log10(np.array(x_values)))
        ax.set_xticklabels([f"{value:g}" for value in x_values])
        ax.set_yticks(np.log10(np.array(SOLIMP2_PLOT_VALUES)))
        ax.set_yticklabels([f"{value:g}" for value in SOLIMP2_PLOT_VALUES])
        ax.tick_params(axis="both", which="major", labelsize=8, pad=1)
        ax.zaxis.set_tick_params(labelsize=8, pad=2)
        ax.view_init(elev=27, azim=-52)
        ax.set_box_aspect((1.2, 1.0, 0.62))
        ax.xaxis.set_pane_color((0.96, 0.96, 0.96, 1.0))
        ax.yaxis.set_pane_color((0.96, 0.96, 0.96, 1.0))
        ax.zaxis.set_pane_color((0.985, 0.985, 0.985, 1.0))
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis._axinfo["grid"]["color"] = (0.68, 0.68, 0.68, 0.75)
            axis._axinfo["grid"]["linewidth"] = 0.75
        ax.set_title(
            f"{algo.upper()} Go2 reward surface, solimp[1]={fixed_solimp1:g}, "
            f"solref[0]={sr0:g}"
        )

        cbar = fig.colorbar(
            mappable, ax=ax, shrink=0.64, pad=0.12, extend="max"
        )
        tick_mm = np.geomspace(
            10 ** color_log_pen_min, 10 ** color_log_pen_max, num=5
        )
        cbar.set_ticks(np.log10(tick_mm))
        cbar.set_ticklabels([f"{value:.1f}" for value in tick_mm])
        cbar.set_label(
            "penetration (mm), clipped; darker = harder"
        )

        sr_name = _solref_filename_value(sr0)
        if abs(fixed_solimp1 - 0.95) < 1e-9:
            filename = (
                f"{algo}_go2_solimp0_solimp2_reward_surface_"
                f"solref_{sr_name}.png"
            )
        else:
            solimp1_name = _solref_filename_value(fixed_solimp1)
            filename = (
                f"{algo}_go2_solimp0_solimp2_reward_surface_"
                f"solimp1_{solimp1_name}_solref_{sr_name}.png"
            )
        p = os.path.join(
            FIGURE_DIR,
            filename,
        )
        fig.savefig(p)
        plt.close(fig)
        paths.append(p)

    return paths


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Plot Go2 solimp/solref sweep results."
    )
    parser.add_argument(
        "--algo",
        choices=("apg", "ppo", "both"),
        default="apg",
        help="Which algorithm results to plot. Default: apg.",
    )
    return parser.parse_args()


def main():
    args = _parse_args()
    os.makedirs(FIGURE_DIR, exist_ok=True)
    plt.rcParams.update({
        "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
        "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
    })

    metric = "eval/episode_reward"
    algos = ["apg", "ppo"] if args.algo == "both" else [args.algo]
    for algo in algos:
        results = load_results(algo)
        if not results:
            print(f"No {algo.upper()} results found.")
            continue

        vals = [r[metric] for r in results if r[metric] is not None]
        print(f"{algo.upper()}: {len(results)} results, "
              f"{metric}: {min(vals):.2f} – {max(vals):.2f} (mean {statistics.mean(vals):.2f})")

        paths = [
            save_csv(results, algo),
            plot_ranked(results, algo, metric),
            plot_main_effects(results, algo, metric),
            plot_factor_boxplots(results, algo, metric),
            plot_solimp2_trend(results, algo),
            plot_best_solref(results, algo, metric),
            plot_heatmap(results, algo, metric),
            plot_3d_grid(results, algo, metric),
            *plot_solimp0_solimp2_reward_surface_by_solref(
                results, algo, metric
            ),
            # Softness-based plots
            plot_softness_rank_vs_reward(results, algo, metric),
            plot_softness_true_spacing_vs_reward(results, algo, metric),
            *plot_softness_true_spacing_by_solref(results, algo, metric),
            plot_softness_scatter(results, algo, metric),
            plot_softness_ranked(results, algo, metric),
            plot_softness_vs_reward_by_solref(results, algo, metric),
            plot_softness_boxplot(results, algo, metric),
        ]
        paths = [p for p in paths if p is not None]
        for p in paths:
            print(f"  {p}")


if __name__ == "__main__":
    main()
