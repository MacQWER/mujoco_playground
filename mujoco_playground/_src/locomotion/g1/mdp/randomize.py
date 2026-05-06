# Copyright 2025 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# ...
# ==============================================================================
"""Domain randomization for G1 environment."""

import jax
from mujoco import mjx

TORSO_BODY_ID = 16


def domain_randomize(model: mjx.Model, rng: jax.Array):
    @jax.vmap
    def rand_dynamics(rng):
        # Floor/foot friction.
        rng, key = jax.random.split(rng)
        pair_friction = model.pair_friction
        pair_friction = pair_friction.at[0:2, 0:2].set(
            jax.random.uniform(key, (2, 2), minval=0.4, maxval=1.0)
        )

        # Dof frictionloss.
        rng, key = jax.random.split(rng)
        dof_frictionloss = model.dof_frictionloss
        dof_frictionloss = dof_frictionloss.at[6:].set(
            dof_frictionloss[6:]
            * jax.random.uniform(
                key,
                (model.nv - 6,),
                minval=0.5,
                maxval=2.0,
            )
        )

        # Dof armature.
        rng, key = jax.random.split(rng)
        dof_armature = model.dof_armature
        dof_armature = dof_armature.at[6:].set(
            dof_armature[6:]
            * jax.random.uniform(
                key,
                (model.nv - 6,),
                minval=1.0,
                maxval=1.05,
            )
        )

        # Body masses.
        rng, key = jax.random.split(rng)
        body_mass = model.body_mass
        body_mass = body_mass.at[:].set(
            body_mass[:]
            * jax.random.uniform(
                key,
                (model.nbody,),
                minval=0.9,
                maxval=1.1,
            )
        )

        # Torso mass extra perturbation.
        rng, key = jax.random.split(rng)
        body_mass = body_mass.at[TORSO_BODY_ID].add(
            jax.random.uniform(key, (), minval=-1.0, maxval=1.0)
        )

        # qpos0 jitter.
        rng, key = jax.random.split(rng)
        qpos0 = model.qpos0
        qpos0 = qpos0.at[7:].set(
            qpos0[7:]
            + jax.random.uniform(
                key,
                (model.nq - 7,),
                minval=-0.05,
                maxval=0.05,
            )
        )

        return (
            pair_friction,
            dof_frictionloss,
            dof_armature,
            body_mass,
            qpos0,
        )

    (
        pair_friction,
        dof_frictionloss,
        dof_armature,
        body_mass,
        qpos0,
    ) = rand_dynamics(rng)

    in_axes = jax.tree_util.tree_map(lambda x: None, model)
    in_axes = in_axes.tree_replace({
        "pair_friction": 0,
        "dof_frictionloss": 0,
        "dof_armature": 0,
        "body_mass": 0,
        "qpos0": 0,
    })

    model = model.tree_replace({
        "pair_friction": pair_friction,
        "dof_frictionloss": dof_frictionloss,
        "dof_armature": dof_armature,
        "body_mass": body_mass,
        "qpos0": qpos0,
    })

    return model, in_axes
