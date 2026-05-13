"""Ball-drop softness calibration for Go2 solimp/solref sweep parameters.

Drops a ball (r=0.01m, m=1kg) from 1m onto a ground plane.
The solimp/solref under test are applied to the ball geom.
Records max penetration depth as a softness proxy.
"""

import csv
import itertools
import os

import mujoco
import numpy as np

SOLIMP0_VALUES = [0.015, 0.9]
SOLIMP1_VALUES = [0.5, 0.95]
SOLIMP2_VALUES = [0.03, 0.001, 0.5]
SOLREF0_VALUES = [0.1, 0.02, 0.004]

# Eval defaults for the ground (fixed, stiff reference surface)
GROUND_SOLIMP = [0.9, 0.95, 0.001, 0.5, 2]
GROUND_SOLREF = [0.004, 1.0]

XML_TEMPLATE = """<mujoco>
  <option timestep="0.0001"/>
  <worldbody>
    <light name="light" pos="1 1 2"/>
    <camera name="track" pos="0 -2 1" xyaxes="1 0 0 0 0.4 0.9"/>
    <geom name="ground" type="plane" size="0.5 0.5 0.05"
          solimp="{g_si0} {g_si1} {g_si2} {g_si3} {g_si4}"
          solref="{g_sr0} {g_sr1}"/>
    <body name="ball" pos="0 0 1">
      <freejoint/>
      <geom name="ball_geom" type="sphere" size="0.01" mass="1"
            solimp="{b_si0} {b_si1} {b_si2} {b_si3} {b_si4}"
            solref="{b_sr0} {b_sr1}"/>
    </body>
  </worldbody>
</mujoco>"""


def max_penetration(s0, s1, s2, sr0):
    """Simulate ball drop and return max penetration depth (mm)."""
    xml = XML_TEMPLATE.format(
        g_si0=GROUND_SOLIMP[0], g_si1=GROUND_SOLIMP[1], g_si2=GROUND_SOLIMP[2],
        g_si3=GROUND_SOLIMP[3], g_si4=GROUND_SOLIMP[4],
        g_sr0=GROUND_SOLREF[0], g_sr1=GROUND_SOLREF[1],
        b_si0=s0, b_si1=s1, b_si2=s2, b_si3=0.5, b_si4=2,
        b_sr0=sr0, b_sr1=1.0,
    )
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)

    max_pen = 0.0
    for _ in range(30000):  # 30000 * 0.0001s = 3s, plenty for bounce+settle
        mujoco.mj_step(model, data)
        for i in range(data.ncon):
            dist = data.contact.dist[i]
            if dist < 0:
                max_pen = max(max_pen, -dist)

    return float(max_pen)


def main():
    grid = [
        combo
        for combo in itertools.product(
            SOLIMP0_VALUES, SOLIMP1_VALUES, SOLIMP2_VALUES, SOLREF0_VALUES
        )
        if combo[0] <= combo[1]
    ]

    print(f"Testing {len(grid)} parameter combinations...")
    print(f"{'solimp0':>10} {'solimp1':>10} {'solimp2':>10} {'solref0':>10} {'penetration_mm':>16}")
    print("-" * 60)

    results = []
    for i, (s0, s1, s2, sr0) in enumerate(grid):
        pen = max_penetration(s0, s1, s2, sr0)
        results.append((s0, s1, s2, sr0, pen))
        print(f"{s0:10.4f} {s1:10.4f} {s2:10.4f} {sr0:10.4f} {pen * 1000:16.6f}")
        if (i + 1) % 5 == 0:
            print(f"  ... {i+1}/{len(grid)} done")

    print()
    print("=" * 60)
    print("Sorted by softness (penetration descending):")
    print(f"{'rank':>5} {'solimp0':>10} {'solimp1':>10} {'solimp2':>10} {'solref0':>10} {'penetration_mm':>16}")
    print("-" * 60)
    results.sort(key=lambda x: x[4], reverse=True)
    for rank, (s0, s1, s2, sr0, pen) in enumerate(results, 1):
        print(f"{rank:5d} {s0:10.4f} {s1:10.4f} {s2:10.4f} {sr0:10.4f} {pen * 1000:16.6f}")

    # Save CSV for the plot script
    csv_path = os.path.join(os.path.dirname(__file__), "..", "logs", "go2_sweep", "softness_ranking.csv")
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["softness_rank", "solimp0", "solimp1", "solimp2", "solref0", "penetration_m"])
        for rank, (s0, s1, s2, sr0, pen) in enumerate(results, 1):
            writer.writerow([rank, s0, s1, s2, sr0, f"{pen:.10f}"])
    print(f"\nSaved softness ranking to {csv_path}")


if __name__ == "__main__":
    main()
