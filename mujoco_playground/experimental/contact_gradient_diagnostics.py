"""G1 vs Go2 contact-gradient diagnostics.

Default mode plots the recorded results from the 2026-05-11 checks. Use
--recompute to rerun the expensive MJX/JAX checks.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import tempfile
import xml.etree.ElementTree as ET


# kind, robot, variable, eps, autodiff/JVP gradient, central finite difference.
RECORDED = [
    ("proxy", "G1", "root_z", 1e-4, -47.000656, -47.008396),
    ("proxy", "G1", "root_z", 1e-5, -47.000656, -47.063828),
    ("proxy", "G1", "root_z", 1e-6, -47.000656, -47.624111),
    ("proxy", "G1", "left_ankle_roll", 1e-4, 0.704997, 0.0),
    ("proxy", "G1", "left_ankle_roll", 1e-5, 0.704997, 0.0),
    ("proxy", "G1", "left_ankle_roll", 1e-6, 0.704997, 0.0),
    ("proxy", "G1", "left_ankle_pitch", 1e-4, -1.167303, -1.167059),
    ("proxy", "G1", "left_ankle_pitch", 1e-5, -1.167303, -1.174212),
    ("proxy", "G1", "left_ankle_pitch", 1e-6, -1.167303, -1.192093),
    ("proxy", "Go2", "root_z", 1e-4, -87.157990, -87.145569),
    ("proxy", "Go2", "root_z", 1e-5, -87.157990, -87.285042),
    ("proxy", "Go2", "root_z", 1e-6, -87.157990, -88.214874),
    ("proxy", "Go2", "FL_calf", 1e-4, 3.608459, 3.607273),
    ("proxy", "Go2", "FL_calf", 1e-5, 3.608459, 3.647804),
    ("proxy", "Go2", "FL_calf", 1e-6, 3.608459, 3.576279),
    ("dynamics", "G1", "root_z", 1e-4, 1.143456, 6.524324),
    ("dynamics", "G1", "root_z", 3e-5, 1.143456, 19.238394),
    ("dynamics", "G1", "root_z", 1e-5, 1.143456, 55.301189),
    ("dynamics", "G1", "left_ankle_roll", 1e-4, -0.019491, -11.940598),
    ("dynamics", "G1", "left_ankle_roll", 3e-5, -0.019491, -40.700043),
    ("dynamics", "G1", "left_ankle_roll", 1e-5, -0.019491, -118.064880),
    ("dynamics", "G1", "left_ankle_pitch", 1e-4, -0.010768, -6.039739),
    ("dynamics", "G1", "left_ankle_pitch", 3e-5, -0.010768, -20.122530),
    ("dynamics", "G1", "left_ankle_pitch", 1e-5, -0.010768, -59.622528),
    ("dynamics", "Go2", "root_z", 1e-4, 1.687071, 1.659989),
    ("dynamics", "Go2", "root_z", 3e-5, 1.687071, 1.672904),
    ("dynamics", "Go2", "root_z", 1e-5, 1.687071, 1.651049),
    ("dynamics", "Go2", "FL_hip", 1e-4, -0.025040, -0.025630),
    ("dynamics", "Go2", "FL_hip", 3e-5, -0.025040, -0.139078),
    ("dynamics", "Go2", "FL_hip", 1e-5, -0.025040, -0.035763),
    ("dynamics", "Go2", "FL_calf", 1e-4, -0.037023, -0.038147),
    ("dynamics", "Go2", "FL_calf", 3e-5, -0.037023, -0.045697),
    ("dynamics", "Go2", "FL_calf", 1e-5, -0.037023, -0.065565),
]


def relerr(a: float, b: float) -> float:
  return abs(a - b) / (abs(b) + 1e-8)


def recorded_rows() -> list[dict[str, float | str]]:
  rows = []
  for kind, robot, variable, eps, grad, fd in RECORDED:
    rows.append({
        "kind": kind,
        "robot": robot,
        "variable": variable,
        "eps": eps,
        "gradient": grad,
        "finite_diff": fd,
        "relerr": relerr(grad, fd),
    })
  return rows


def repo_path(*parts: str) -> Path:
  return Path(__file__).resolve().parents[2].joinpath(*parts)


def make_g1_visual_free_model():
  """Loads the G1 model after stripping unavailable visual meshes."""
  import mujoco

  src = repo_path("mujoco_playground", "_src", "locomotion", "g1", "xmls")
  tmp = Path(tempfile.mkdtemp(prefix="g1_contact_grad_"))
  for name in ["scene_mjx_feetonly_flat_terrain.xml", "sensor.xml"]:
    shutil.copy(src / name, tmp / name)

  tree = ET.parse(src / "g1_mjx_feetonly.xml")
  root = tree.getroot()
  for parent in root.iter():
    for child in list(parent):
      if child.tag == "mesh":
        parent.remove(child)
  for parent in root.iter():
    for child in list(parent):
      mesh_geom = child.tag == "geom" and child.get("mesh") is not None
      visual_geom = child.tag == "geom" and child.get("class") == "visual"
      if mesh_geom or visual_geom:
        parent.remove(child)
  tree.write(tmp / "g1_mjx_feetonly.xml")
  return mujoco.MjModel.from_xml_path(
      str(tmp / "scene_mjx_feetonly_flat_terrain.xml")
  )


def reset_lift(model, keyframe: str):
  import mujoco

  data = mujoco.MjData(model)
  mujoco.mj_resetDataKeyframe(model, data, model.key(keyframe).id)
  mujoco.mj_forward(model, data)
  min_dist = min((data.contact[i].dist for i in range(data.ncon)), default=0.0)
  qpos = data.qpos.copy()
  qpos[2] -= min(min_dist, 0.0)
  data.qpos[:] = qpos
  mujoco.mj_forward(model, data)
  return qpos


def append_fd_rows(
    rows,
    kind,
    robot,
    variable,
    gradient,
    fn,
    x,
    idx,
    eps_values,
    fd_mode="central",
):
  base = fn(x)
  for eps in eps_values:
    if fd_mode == "central":
      fd = (fn(x.at[idx].add(eps)) - fn(x.at[idx].add(-eps))) / (2 * eps)
    elif fd_mode == "forward":
      fd = (fn(x.at[idx].add(eps)) - base) / eps
    else:
      raise ValueError(f"Unknown finite-difference mode: {fd_mode}")
    grad = float(gradient)
    fd = float(fd)
    rows.append({
        "kind": kind,
        "robot": robot,
        "variable": variable,
        "eps": float(eps),
        "fd_mode": fd_mode,
        "gradient": grad,
        "finite_diff": fd,
        "relerr": relerr(grad, fd),
    })


def recompute_rows(fd_mode: str = "central") -> list[dict[str, float | str]]:
  """Reruns the expensive MJX checks."""
  import jax
  import jax.numpy as jp
  import mujoco
  import mujoco.mjx as mjx
  import numpy as np

  rows = []

  g1 = make_g1_visual_free_model()
  g1_q0 = jp.array(reset_lift(g1, "knees_bent"))
  g1_mjx = mjx.put_model(g1, impl="jax")
  g1_data = mjx.make_data(g1_mjx)
  g1_qvel = jp.zeros(g1.nv)
  g1_ctrl = jp.zeros(g1.nu)
  g1_gids_np = np.array(
      [g1.geom(name).id for name in ["left_foot", "right_foot"]],
      dtype=np.int32,
  )
  g1_gids = jp.array(g1_gids_np)
  g1_size = jp.array(g1.geom_size[g1_gids_np])

  def g1_bottom(qpos):
    data = g1_data.replace(qpos=qpos, qvel=g1_qvel, ctrl=g1_ctrl)
    data = mjx.forward(g1_mjx, data)
    rot = data.geom_xmat[g1_gids].reshape((2, 3, 3))
    vertical_radius = jp.sum(jp.abs(rot[:, 2, :]) * g1_size, axis=-1)
    return data.geom_xpos[g1_gids, 2] - vertical_radius

  def g1_proxy(qpos):
    return jp.sum(jax.nn.sigmoid((0.005 - g1_bottom(qpos)) * 100.0))

  g1_proxy_grad = jax.jit(jax.grad(g1_proxy))(g1_q0)
  for idx, name in [
      (2, "root_z"),
      (int(g1.joint("left_ankle_roll_joint").qposadr[0]), "left_ankle_roll"),
      (int(g1.joint("left_ankle_pitch_joint").qposadr[0]), "left_ankle_pitch"),
  ]:
    append_fd_rows(
        rows, "proxy", "G1", name, g1_proxy_grad[idx],
        g1_proxy, g1_q0, idx, [1e-4, 1e-5, 1e-6], fd_mode
    )

  go2 = mujoco.MjModel.from_xml_path(str(repo_path(
      "mujoco_playground",
      "_src",
      "locomotion",
      "go2",
      "xmls",
      "scene_mjx_collision_free.xml",
  )))
  go2_q0 = jp.array(reset_lift(go2, "home"))
  go2_mjx = mjx.put_model(go2, impl="jax")
  go2_data = mjx.make_data(go2_mjx)
  go2_qvel = jp.zeros(go2.nv)
  go2_ctrl = jp.zeros(go2.nu)
  go2_sids = jp.array([
      go2.site(name).id
      for name in ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]
  ])

  def go2_site_z(qpos):
    data = go2_data.replace(qpos=qpos, qvel=go2_qvel, ctrl=go2_ctrl)
    data = mjx.forward(go2_mjx, data)
    return data.site_xpos[go2_sids, 2]

  def go2_proxy(qpos):
    return jp.sum(jax.nn.sigmoid((0.025 - go2_site_z(qpos)) * 100.0))

  go2_proxy_grad = jax.jit(jax.grad(go2_proxy))(go2_q0)
  for idx, name in [
      (2, "root_z"),
      (int(go2.joint("FL_calf_joint").qposadr[0]), "FL_calf"),
  ]:
    append_fd_rows(
        rows, "proxy", "Go2", name, go2_proxy_grad[idx],
        go2_proxy, go2_q0, idx, [1e-4, 1e-5, 1e-6], fd_mode
    )

  def add_dynamics(robot, model, q0, index_names):
    mjx_model = mjx.put_model(model, impl="jax")
    base_data = mjx.make_data(mjx_model)
    q0 = jp.array(q0)
    qvel = jp.zeros(model.nv).at[0].set(0.25).at[1].set(-0.15)
    ctrl = jp.array(q0[7:7 + model.nu]) if model.nu else jp.zeros((0,))

    def loss(qpos):
      data = base_data.replace(qpos=qpos, qvel=qvel, ctrl=ctrl)
      data = mjx.forward(mjx_model, data)
      data = mjx.step(mjx_model, data)
      return (
          10.0 * jp.sum(data.qvel[:2] ** 2)
          + data.qpos[2]
          + 0.01 * jp.sum(data.qvel[6:] ** 2)
      )

    loss = jax.jit(loss)
    for idx, name in index_names:
      direction = jp.zeros_like(q0).at[idx].set(1.0)
      _, tangent = jax.jvp(loss, (q0,), (direction,))
      append_fd_rows(
          rows, "dynamics", robot, name, tangent,
          loss, q0, idx, [1e-4, 3e-5, 1e-5], fd_mode
      )

  add_dynamics(
      "G1",
      g1,
      reset_lift(g1, "knees_bent"),
      [
          (2, "root_z"),
          (int(g1.joint("left_ankle_roll_joint").qposadr[0]), "left_ankle_roll"),
          (int(g1.joint("left_ankle_pitch_joint").qposadr[0]), "left_ankle_pitch"),
      ],
  )
  add_dynamics(
      "Go2",
      go2,
      reset_lift(go2, "home"),
      [
          (2, "root_z"),
          (int(go2.joint("FL_hip_joint").qposadr[0]), "FL_hip"),
          (int(go2.joint("FL_calf_joint").qposadr[0]), "FL_calf"),
      ],
  )
  return rows


def filter_rows(rows, *, kind=None, robot=None, eps=None):
  out = rows
  if kind is not None:
    out = [row for row in out if row["kind"] == kind]
  if robot is not None:
    out = [row for row in out if row["robot"] == robot]
  if eps is not None:
    out = [row for row in out if abs(row["eps"] - eps) < 1e-12]
  return out


def plot_relerr(ax, rows, title):
  labels = [f"{row['robot']}\n{row['variable']}" for row in rows]
  values = [row["relerr"] for row in rows]
  colors = ["#c83f49" if row["robot"] == "G1" else "#2878b5" for row in rows]
  ax.bar(range(len(rows)), values, color=colors)
  ax.set_xticks(range(len(rows)), labels, rotation=30, ha="right")
  ax.set_yscale("log")
  ax.axhline(0.05, color="#555555", linestyle="--", linewidth=1)
  ax.set_title(title)
  ax.set_ylabel("relative error")
  ax.grid(True, axis="y", alpha=0.25)


def plot_fd_sweep(ax, rows, title):
  for variable in sorted({row["variable"] for row in rows}):
    series = sorted(
        [row for row in rows if row["variable"] == variable],
        key=lambda row: row["eps"],
        reverse=True,
    )
    ax.plot(
        [row["eps"] for row in series],
        [abs(row["finite_diff"]) for row in series],
        marker="o",
        label=variable,
    )
  ax.set_xscale("log")
  ax.set_yscale("log")
  ax.invert_xaxis()
  ax.set_title(title)
  ax.set_xlabel("finite-difference epsilon")
  ax.set_ylabel("|finite-difference derivative|")
  ax.grid(True, which="both", alpha=0.25)
  ax.legend(fontsize=8)


def plot_rows(rows, output: Path, fd_mode: str) -> None:
  import matplotlib.pyplot as plt

  output.parent.mkdir(parents=True, exist_ok=True)
  fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
  fig.suptitle(
      f"G1 vs Go2 contact-gradient diagnostics ({fd_mode} FD)",
      fontsize=14,
  )
  plot_relerr(
      axes[0, 0],
      filter_rows(rows, kind="proxy", eps=1e-4),
      "Proxy-only gradient check",
  )
  plot_relerr(
      axes[0, 1],
      filter_rows(rows, kind="dynamics", eps=1e-4),
      "One-step dynamics-through-contact check",
  )
  plot_fd_sweep(
      axes[1, 0],
      filter_rows(rows, kind="dynamics", robot="G1"),
      "G1 finite differences near contact",
  )
  plot_fd_sweep(
      axes[1, 1],
      filter_rows(rows, kind="dynamics", robot="Go2"),
      "Go2 finite differences near contact",
  )
  fig.savefig(output, dpi=180)
  print(f"Wrote plot to {output}")


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--recompute", action="store_true")
  parser.add_argument(
      "--fd-mode",
      choices=["central", "forward"],
      default="central",
  )
  parser.add_argument(
      "--output",
      type=Path,
      default=Path("/tmp/contact_gradient_diagnostics.png"),
  )
  parser.add_argument("--json", type=Path, default=None)
  args = parser.parse_args()

  if args.fd_mode != "central" and not args.recompute:
    parser.error("--fd-mode forward requires --recompute")

  rows = recompute_rows(args.fd_mode) if args.recompute else recorded_rows()
  plot_rows(rows, args.output, args.fd_mode)
  if args.json is not None:
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(rows, indent=2) + "\n")
    print(f"Wrote rows to {args.json}")


if __name__ == "__main__":
  main()
