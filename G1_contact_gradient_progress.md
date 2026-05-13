# G1 Contact-Gradient Progress

Date: 2026-05-11

## Goal

Track why G1 contact gradients look unstable compared with Go2, and keep enough context to continue later.

## What was checked

- Existing diagnostic script: `mujoco_playground/experimental/contact_gradient_diagnostics.py`
- Central finite-difference plot: `/tmp/contact_gradient_diagnostics.png`
- Forward finite-difference plot: `contact_gradient_diagnostics_forward_fd.png`

## Main findings

- The left-top proxy-only check is fine.
- That proxy is just a soft sigmoid of foot height, and autodiff vs finite diff agree well for both robots.
- The big mismatch is in the one-step dynamics-through-contact check.
- For G1, tiny perturbations can change the contact manifold itself:
  - `root_z + 1e-4` can drop `ncon` from 4 to 0.
  - Tiny ankle perturbations can reduce the number of active contacts and change which points are active.
- Go2 is much more stable under the same perturbations:
  - `ncon` stays at 4.
  - Contact distances and positions change smoothly.

## Interpretation

- The issue is not best described as "contact force is non-continuous" by itself.
- The stronger signal is contact active-set switching:
  - which geoms are touching
  - how many contact points are active
  - whether the constraint set changes under small perturbations
- The finite difference plots are a local numerical reference, not ground truth.
- The forward-FD rerun supports the same conclusion: G1 is locally much less stable than Go2 near contact.

## Code details worth remembering

- G1 foot contact proxy uses `_get_foot_bottom_z()` in `mujoco_playground/_src/locomotion/g1/joystick2.py`
- G1 foot geom is a box: `mujoco_playground/_src/locomotion/g1/xmls/g1_mjx_feetonly.xml`
- G1 foot-floor pair is `condim=3` with no extra margin in the pair definition
- Go2 foot geoms use a small margin and `condim=6` in `mujoco_playground/_src/locomotion/go2/xmls/go2_mjx_collision_free.xml`
- Go2 main contact check is still binary in `mujoco_playground/_src/locomotion/go2/joystick.py`

## Next step

- Run a multi-seed version of the diagnostic.
- Log mean/std plus contact-set flip rate across resets.
- Record `ncon`, contact pairs, `contact.dist`, `contact.frame`, and possibly contact forces.
- If needed, test whether adding margin / changing foot contact geometry for G1 reduces flip rate and gradient mismatch.

