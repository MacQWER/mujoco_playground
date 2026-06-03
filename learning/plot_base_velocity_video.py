"""Render base velocity CSV data as a standalone 16:9 MP4."""

from absl import app
from absl import flags
from pathlib import Path
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import mediapy as media
import numpy as np


_CSV = flags.DEFINE_string(
    "csv", None, "CSV produced by train_jax_apg.py --velocity_report."
)
_OUTPUT = flags.DEFINE_string(
    "output", None, "Output MP4 path. Defaults to CSV stem + .mp4."
)
_FPS = flags.DEFINE_float("fps", 25.0, "Output video FPS.")
_WIDTH = flags.DEFINE_integer("width", 1920, "Output video width.")
_HEIGHT = flags.DEFINE_integer("height", 1080, "Output video height.")
_TITLE = flags.DEFINE_string("title", None, "Optional figure title.")
_MAX_STEPS = flags.DEFINE_integer(
    "max_steps",
    None,
    "Maximum CSV rows to render. Use this to limit rollout video steps.",
)


def _load_velocity_csv(path: str) -> dict[str, np.ndarray]:
  data = np.genfromtxt(path, delimiter=",", names=True)
  if data.ndim == 0:
    data = data.reshape(1)
  required = [
      "time_s",
      "local_vx_mps",
      "local_vy_mps",
      "yaw_rate_radps",
      "speed_xy_mps",
      "command_vx_mps",
      "command_vy_mps",
      "command_yaw_radps",
  ]
  missing = [name for name in required if name not in data.dtype.names]
  if missing:
    raise ValueError(f"Missing required CSV columns: {missing}")
  return {name: np.asarray(data[name]) for name in data.dtype.names}


def _limits(values: np.ndarray, refs: np.ndarray) -> tuple[float, float]:
  all_values = np.concatenate([values.reshape(-1), refs.reshape(-1)])
  lo = float(np.min(all_values))
  hi = float(np.max(all_values))
  span = hi - lo
  pad = max(0.08 * span, 0.08)
  return lo - pad, hi + pad


def _rgb_from_figure(fig, height: int, width: int) -> np.ndarray:
  fig.canvas.draw()
  image = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
  if image.shape[:2] != (height, width):
    raise ValueError(
        f"Figure produced {image.shape[:2]}, expected {(height, width)}."
    )
  return image


def _frame_indices(times_s: np.ndarray, fps: float) -> np.ndarray:
  if len(times_s) < 2:
    return np.array([0], dtype=int)
  data_dt = float(np.median(np.diff(times_s)))
  stride = max(1, int(round((1.0 / fps) / data_dt)))
  return np.arange(0, len(times_s), stride, dtype=int)


def _make_figure(
    title: str,
    times_s: np.ndarray,
    vx: np.ndarray,
    vy: np.ndarray,
    yaw_rate: np.ndarray,
    speed_xy: np.ndarray,
    command: np.ndarray,
    width: int,
    height: int,
):
  dpi = 120
  fig, (linear_ax, yaw_ax) = plt.subplots(
      2, 1, figsize=(width / dpi, height / dpi), dpi=dpi, sharex=True
  )
  fig.patch.set_facecolor("white")
  fig.subplots_adjust(left=0.055, right=0.985, top=0.64, bottom=0.105, hspace=0.46)
  fig.suptitle(title, fontsize=32, fontweight="bold", y=0.965)

  command_label = (
      f"command: vx={command[0]:.2f} m/s, "
      f"vy={command[1]:.2f} m/s, yaw={command[2]:.2f} rad/s"
  )
  average_label = (
      f"average speed={float(np.mean(speed_xy)):.3f} m/s, "
      f"average vx={float(np.mean(vx)):.3f} m/s, "
      f"average vy={float(np.mean(vy)):.3f} m/s, "
      f"average yaw={float(np.mean(yaw_rate)):.3f} rad/s"
  )
  fig.text(0.055, 0.865, command_label, fontsize=22, color="#222222")
  fig.text(0.055, 0.815, average_label, fontsize=22, color="#222222")

  linear_ax.axhline(
      command[0], color="#1f77b4", linestyle="--",
      linewidth=2.5, label="cmd vx"
  )
  linear_ax.axhline(
      command[1], color="#2ca02c", linestyle="--",
      linewidth=2.5, label="cmd vy"
  )
  vx_line, = linear_ax.plot(
      [], [], color="#1f77b4", linewidth=3.2, label="base vx"
  )
  vy_line, = linear_ax.plot(
      [], [], color="#2ca02c", linewidth=3.2, label="base vy"
  )
  linear_cursor = linear_ax.axvline(
      times_s[0], color="#111111", linewidth=2.0, alpha=0.45
  )
  linear_ax.set_xlim(0.0, float(times_s[-1]))
  linear_ax.set_ylim(*_limits(np.concatenate([vx, vy]), command[:2]))
  linear_ax.set_ylabel("linear velocity (m/s)", fontsize=20)
  linear_ax.tick_params(labelsize=16)
  linear_ax.grid(True, alpha=0.28)
  linear_ax.legend(
      loc="lower left", bbox_to_anchor=(0.0, 1.10),
      ncols=4, fontsize=16, frameon=False, borderaxespad=0.0
  )

  yaw_ax.axhline(
      command[2], color="#9467bd", linestyle="--",
      linewidth=2.5, label="cmd yaw"
  )
  yaw_line, = yaw_ax.plot(
      [], [], color="#9467bd", linewidth=3.2, label="base yaw rate"
  )
  yaw_cursor = yaw_ax.axvline(
      times_s[0], color="#111111", linewidth=2.0, alpha=0.45
  )
  yaw_ax.set_xlim(0.0, float(times_s[-1]))
  yaw_ax.set_ylim(*_limits(yaw_rate, command[2:3]))
  yaw_ax.set_xlabel("time (s)", fontsize=20)
  yaw_ax.set_ylabel("yaw rate (rad/s)", fontsize=20)
  yaw_ax.tick_params(labelsize=16)
  yaw_ax.grid(True, alpha=0.28)
  yaw_ax.legend(
      loc="lower left", bbox_to_anchor=(0.0, 1.10),
      fontsize=16, frameon=False, borderaxespad=0.0
  )

  current_text = fig.text(
      0.055, 0.765, "", fontsize=21, color="#111111", fontweight="bold"
  )
  artists = {
      "vx_line": vx_line,
      "vy_line": vy_line,
      "yaw_line": yaw_line,
      "linear_cursor": linear_cursor,
      "yaw_cursor": yaw_cursor,
      "current_text": current_text,
  }
  return fig, artists


def _render_video(
    csv_path: str,
    output_path: str,
    fps: float,
    width: int,
    height: int,
    title: str,
    max_steps: int | None,
) -> None:
  values = _load_velocity_csv(csv_path)
  if max_steps is not None:
    if max_steps <= 0:
      raise ValueError("--max_steps must be positive when set.")
    values = {name: value[:max_steps] for name, value in values.items()}
  times_s = values["time_s"]
  if len(times_s) == 0:
    raise ValueError("CSV has no rows to render.")
  vx = values["local_vx_mps"]
  vy = values["local_vy_mps"]
  yaw_rate = values["yaw_rate_radps"]
  speed_xy = values["speed_xy_mps"]
  command = np.array([
      values["command_vx_mps"][0],
      values["command_vy_mps"][0],
      values["command_yaw_radps"][0],
  ])

  fig, artists = _make_figure(
      title, times_s, vx, vy, yaw_rate, speed_xy, command, width, height
  )
  frame_indices = _frame_indices(times_s, fps)
  print(f"Writing {len(frame_indices)} frames to {output_path}")

  with media.VideoWriter(output_path, shape=(height, width), fps=fps, crf=18) as writer:
    for frame_num, idx in enumerate(frame_indices):
      end = int(idx) + 1
      t_now = times_s[idx]
      artists["vx_line"].set_data(times_s[:end], vx[:end])
      artists["vy_line"].set_data(times_s[:end], vy[:end])
      artists["yaw_line"].set_data(times_s[:end], yaw_rate[:end])
      artists["linear_cursor"].set_xdata([t_now, t_now])
      artists["yaw_cursor"].set_xdata([t_now, t_now])
      artists["current_text"].set_text(
          f"t={t_now:5.2f}s    "
          f"vx={vx[idx]: .3f} m/s    vy={vy[idx]: .3f} m/s    "
          f"yaw={yaw_rate[idx]: .3f} rad/s"
      )
      writer.add_image(_rgb_from_figure(fig, height, width))
      if (frame_num + 1) % 100 == 0 or frame_num + 1 == len(frame_indices):
        print(f"  wrote {frame_num + 1}/{len(frame_indices)} frames")
  plt.close(fig)


def main(argv):
  del argv
  if _CSV.value is None:
    raise flags.Error("--csv is required")
  output = _OUTPUT.value
  if output is None:
    output = str(Path(_CSV.value).with_suffix(".mp4"))
  title = _TITLE.value or "velocity record"
  _render_video(
      _CSV.value,
      output,
      _FPS.value,
      _WIDTH.value,
      _HEIGHT.value,
      title,
      _MAX_STEPS.value,
  )


if __name__ == "__main__":
  app.run(main)
