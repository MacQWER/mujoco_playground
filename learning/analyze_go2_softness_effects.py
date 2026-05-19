"""Analyze solimp parameter effects on Go2 softness calibration."""

import csv
import itertools
import math
import os
import statistics
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np


LOG_DIR = os.path.join(os.path.dirname(__file__), "..", "logs", "go2_sweep")
FIGURE_DIR = os.path.join(LOG_DIR, "figures")
CSV_PATH = os.path.join(LOG_DIR, "softness_ranking_augmented_mid42.csv")


def load_rows():
  rows = []
  with open(CSV_PATH, newline="", encoding="utf-8") as f:
    for row in csv.DictReader(f):
      parsed = {
          key: float(value)
          for key, value in row.items()
          if key != "softness_rank"
      }
      parsed["softness_rank"] = int(row["softness_rank"])
      parsed["penetration_mm"] = parsed["penetration_m"] * 1000.0
      rows.append(parsed)
  return rows


def matched_ranges(rows, param, include_solref=True):
  keys = ["solimp0", "solimp1", "solimp2"]
  if include_solref:
    keys.append("solref0")
  others = [key for key in keys if key != param]

  groups = defaultdict(list)
  for row in rows:
    groups[tuple(row[key] for key in others)].append(row)

  effects = []
  details = []
  for key, group in groups.items():
    levels = sorted({row[param] for row in group})
    if len(levels) < 2:
      continue

    means = []
    for level in levels:
      values = [
          row["penetration_mm"]
          for row in group
          if abs(row[param] - level) < 1e-12
      ]
      means.append((level, statistics.mean(values)))

    values = [value for _, value in means]
    effect = max(values) - min(values)
    effects.append(effect)
    details.append((key, means, effect))
  return effects, details


def pairwise_effects(rows, param):
  others = [
      key
      for key in ["solimp0", "solimp1", "solimp2", "solref0"]
      if key != param
  ]
  groups = defaultdict(list)
  for row in rows:
    groups[tuple(row[key] for key in others)].append(row)

  diffs = defaultdict(list)
  for group in groups.values():
    by_level = defaultdict(list)
    for row in group:
      by_level[row[param]].append(row["penetration_mm"])
    for low, high in itertools.combinations(sorted(by_level), 2):
      low_mean = statistics.mean(by_level[low])
      high_mean = statistics.mean(by_level[high])
      diffs[(low, high)].append(high_mean - low_mean)

  summary = []
  for pair, values in diffs.items():
    summary.append((
        statistics.mean(abs(value) for value in values),
        statistics.mean(values),
        len(values),
        pair,
    ))
  return sorted(summary, reverse=True)


def design_matrix(rows, cols):
  columns = [np.ones(len(rows))]
  names = ["intercept"]
  for col in cols:
    levels = sorted({row[col] for row in rows})
    for level in levels[1:]:
      columns.append(np.array([
          1.0 if abs(row[col] - level) < 1e-12 else 0.0
          for row in rows
      ]))
      names.append(f"{col}={level:g}")
  return np.vstack(columns).T, names


def fit_r2(rows, cols, y):
  x, names = design_matrix(rows, cols)
  beta = np.linalg.lstsq(x, y, rcond=None)[0]
  pred = x @ beta
  ss_res = float(np.sum((y - pred) ** 2))
  ss_tot = float(np.sum((y - np.mean(y)) ** 2))
  return 1.0 - ss_res / ss_tot, ss_res, len(names)


def partial_r2(rows, log_scale=False):
  solref_means = {}
  for solref in sorted({row["solref0"] for row in rows}):
    values = [
        math.log(row["penetration_mm"]) if log_scale else row["penetration_mm"]
        for row in rows
        if abs(row["solref0"] - solref) < 1e-12
    ]
    solref_means[solref] = statistics.mean(values)

  y = np.array([
      (math.log(row["penetration_mm"]) if log_scale else row["penetration_mm"])
      - solref_means[row["solref0"]]
      for row in rows
  ])

  params = ["solimp0", "solimp1", "solimp2"]
  full_r2, full_ss, n_params = fit_r2(rows, params, y)
  out = {}
  for param in params:
    cols = [col for col in params if col != param]
    r2, ss, _ = fit_r2(rows, cols, y)
    out[param] = {
        "drop_delta_r2": full_r2 - r2,
        "partial_r2": (ss - full_ss) / ss,
    }
  return full_r2, full_ss, n_params, out


def write_summary_csv(rows, effect_rows):
  out_path = os.path.join(FIGURE_DIR, "go2_solimp_penetration_effects.csv")
  os.makedirs(FIGURE_DIR, exist_ok=True)
  with open(out_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "parameter",
            "matched_groups",
            "mean_matched_range_mm",
            "median_matched_range_mm",
            "max_matched_range_mm",
            "partial_r2_linear",
            "partial_r2_log",
        ],
    )
    writer.writeheader()
    for row in effect_rows:
      writer.writerow(row)
  return out_path


def plot_effects(effect_rows):
  os.makedirs(FIGURE_DIR, exist_ok=True)
  params = [row["parameter"] for row in effect_rows]
  mean_ranges = [row["mean_matched_range_mm"] for row in effect_rows]
  partial = [100.0 * row["partial_r2_linear"] for row in effect_rows]
  partial_log = [100.0 * row["partial_r2_log"] for row in effect_rows]

  fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), constrained_layout=True)
  colors = ["#4c78a8", "#f58518", "#54a24b"]

  axes[0].bar(params, mean_ranges, color=colors, alpha=0.8)
  axes[0].set_ylabel("mean matched range (mm)")
  axes[0].set_title("Direct matched effect size")
  axes[0].grid(axis="y", alpha=0.22)
  for i, value in enumerate(mean_ranges):
    axes[0].text(i, value, f"{value:.2f}", ha="center", va="bottom")

  x = np.arange(len(params))
  width = 0.36
  axes[1].bar(x - width / 2, partial, width, label="linear mm", color="#6f8fb9")
  axes[1].bar(x + width / 2, partial_log, width, label="log mm", color="#72a86f")
  axes[1].set_xticks(x, params)
  axes[1].set_ylabel("partial R2 after controlling solref0 (%)")
  axes[1].set_title("Model-based unique contribution")
  axes[1].legend(frameon=False)
  axes[1].grid(axis="y", alpha=0.22)
  for offset, values in [(-width / 2, partial), (width / 2, partial_log)]:
    for i, value in enumerate(values):
      axes[1].text(i + offset, value, f"{value:.0f}", ha="center", va="bottom")

  fig.suptitle("Go2 ball-drop penetration: solimp parameter effects")
  out_path = os.path.join(FIGURE_DIR, "go2_solimp_penetration_effects.png")
  fig.savefig(out_path, dpi=300, bbox_inches="tight")
  plt.close(fig)
  return out_path


def main():
  rows = load_rows()
  params = ["solimp0", "solimp1", "solimp2"]

  print(f"N={len(rows)}")
  print("Penetration by solref0:")
  for solref in sorted({row["solref0"] for row in rows}, reverse=True):
    values = [
        row["penetration_mm"]
        for row in rows
        if abs(row["solref0"] - solref) < 1e-12
    ]
    print(
        f"  solref0={solref:g}: n={len(values)}, "
        f"mean={statistics.mean(values):.3f}mm, "
        f"range={min(values):.3f}-{max(values):.3f}mm"
    )

  full_r2, full_ss, n_params, linear_partial = partial_r2(rows, log_scale=False)
  log_full_r2, _, _, log_partial = partial_r2(rows, log_scale=True)

  print()
  print("Matched ranges, controlling all other solimp params and solref0:")
  effect_rows = []
  detail_by_param = {}
  for param in params:
    effects, details = matched_ranges(rows, param)
    detail_by_param[param] = details
    effect_rows.append({
        "parameter": param,
        "matched_groups": len(effects),
        "mean_matched_range_mm": statistics.mean(effects),
        "median_matched_range_mm": statistics.median(effects),
        "max_matched_range_mm": max(effects),
        "partial_r2_linear": linear_partial[param]["partial_r2"],
        "partial_r2_log": log_partial[param]["partial_r2"],
    })
    print(
        f"  {param}: groups={len(effects)}, "
        f"mean={statistics.mean(effects):.3f}mm, "
        f"median={statistics.median(effects):.3f}mm, max={max(effects):.3f}mm"
    )

  print()
  print("By solref0, mean matched range:")
  for solref in sorted({row["solref0"] for row in rows}, reverse=True):
    subset = [row for row in rows if abs(row["solref0"] - solref) < 1e-12]
    print(f"  solref0={solref:g}")
    for param in params:
      effects, _ = matched_ranges(subset, param, include_solref=False)
      print(
          f"    {param}: mean={statistics.mean(effects):.3f}mm, "
          f"median={statistics.median(effects):.3f}mm, max={max(effects):.3f}mm"
      )

  print()
  print("Model partial R2 after centering within solref0:")
  print(f"  linear full R2={full_r2:.3f}")
  for param in params:
    print(
        f"    {param}: partial_R2={linear_partial[param]['partial_r2']:.3f}, "
        f"delta_R2={linear_partial[param]['drop_delta_r2']:.3f}"
    )
  print(f"  log full R2={log_full_r2:.3f}")
  for param in params:
    print(
        f"    {param}: partial_R2={log_partial[param]['partial_r2']:.3f}, "
        f"delta_R2={log_partial[param]['drop_delta_r2']:.3f}"
    )

  print()
  print("Top matched cases per parameter:")
  for param in params:
    print(f"  {param}")
    for key, means, effect in sorted(
        detail_by_param[param], key=lambda item: item[2], reverse=True
    )[:5]:
      print(f"    key={key}, range={effect:.3f}mm, means={means}")

  print()
  print("Largest matched pairwise effects:")
  for param in params:
    print(f"  {param}")
    for mean_abs, mean_signed, n, pair in pairwise_effects(rows, param)[:6]:
      print(
          f"    {pair}: n={n}, mean_abs={mean_abs:.3f}mm, "
          f"mean_signed={mean_signed:.3f}mm"
      )

  csv_path = write_summary_csv(rows, effect_rows)
  fig_path = plot_effects(effect_rows)
  print()
  print(f"Wrote {csv_path}")
  print(f"Wrote {fig_path}")


if __name__ == "__main__":
  main()
