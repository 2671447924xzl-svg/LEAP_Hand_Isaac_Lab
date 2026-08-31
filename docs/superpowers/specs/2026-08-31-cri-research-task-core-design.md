# CRI Research Task Core Design

## Status and scope

- Status: approved for implementation on 2026-08-31.
- Protocol ID: `CRI_RESEARCH_TASK_CORE_IMPLEMENTATION`.
- Task ID: `Isaac-Reorient-Cube-Leap-CRI`.
- Factor schema: `CRI_FACTOR_SCHEMA_V1`.
- Scope: deterministic, episode-static object mass, object-side contact friction, and motor-capacity factors.
- Excluded: PPO training, reward/observation/action changes, latency, controller-gain uncertainty, object-shape uncertainty, changing dynamics, and adaptation methods.

The existing `Isaac-Reorient-Cube-Leap` task and its checkpoints remain unchanged. The CRI task inherits the baseline behavior and only adds isolated factor configuration, reset-time assignment, simulator read-back, and research metadata.

## Architecture

Create `LEAP_Isaaclab.tasks.leap_hand_reorient_cri` with four focused modules:

1. `factor_conditions.py`: the only source of factor codes, levels, schema constants, and validation.
2. `leap_hand_cri_env_cfg.py`: inherits `LeapHandEnvCfg`, disables baseline ADR/events, fixes object geometry, and binds explicit hand/object PhysX materials.
3. `reorientation_cri_env.py`: inherits `ReorientationEnv`, assigns factors after every environment reset, performs direct simulator read-back, and owns per-environment factor metadata.
4. `__init__.py`: registers `Isaac-Reorient-Cube-Leap-CRI` without changing the baseline registration.

Add `scripts/p0_cri_factor_smoke.py` for the authorized four-environment, zero-action read-back and isolation smoke test. It is diagnostic only and does not enter the PPO pipeline.

## Factor schema

The bit order is `[mass, friction, motor_capacity]`.

| Code | Mass scale | Object static/dynamic friction | Motor scale | Effective effort limit |
|---|---:|---:|---:|---:|
| `000` | 1.00 | 0.80 / 0.80 | 1.00 | 0.50 |
| `100` | 1.35 | 0.80 / 0.80 | 1.00 | 0.50 |
| `010` | 1.00 | 0.45 / 0.45 | 1.00 | 0.50 |
| `001` | 1.00 | 0.80 / 0.80 | 0.80 | 0.40 |

All sixteen LEAP joints share the same motor-capacity scale. Kp, Kd, action scale, target range, action mapping, timing, dt, and decimation remain unchanged.

Factor codes come only from an explicit per-environment assignment buffer/config. The CRI task defaults every environment to `000`; the four-environment smoke test explicitly supplies `[000, 100, 010, 001]`. Reset never samples or silently changes a factor code.

## Friction semantics

`CRI_FACTOR_SCHEMA_V1` defines friction as object-side material uncertainty:

- Hand static and dynamic friction are fixed at `0.80`.
- Object static and dynamic friction are `0.80` for nominal and `0.45` for low friction.
- Hand and object materials both explicitly use `friction_combine_mode="min"`.
- Therefore the hand-object pair resolves to `0.80` for nominal and `0.45` for low friction while only the object-side coefficient changes.

The configuration binds explicit PhysX materials to the hand and cube with stronger-than-descendant semantics. Reset-time writes change only object material coefficients; combine mode remains fixed. The starting value `0.45` is not claimed to be the final paper range and remains subject to later P0-5/floor-ceiling validation.

## Object geometry isolation

The CRI configuration sets `events=None` and `enable_adr=False`. This removes the baseline prestartup `object_scale_size` event and all reset-time ADR terms. The cube keeps the explicit baseline spawn scale `(1.2, 1.2, 1.2)` in every environment. Mass uncertainty is produced only by indexed mass/inertia writes.

## Reset-time data flow

1. The inherited baseline reset restores object and articulation state.
2. The CRI reset reads each reset environment's canonical factor code from the explicit assignment buffer without random sampling.
3. Requested mass is computed from a captured nominal PhysX mass and written with `set_masses_index`.
4. Requested inertia is computed from the captured nominal PhysX inertia using the same mass ratio and written with `set_inertias_index`.
5. Object material static/dynamic friction is written through the PhysX material tensor API for only the selected environment indices.
6. The nominal joint effort limit is captured from `get_dof_max_forces`; the scaled limit is written using `write_joint_effort_limit_to_sim_index` for all sixteen joints in only the selected environments.
7. Actual mass, inertia, hand/object material properties, and effort limits are independently read from PhysX getters and stored separately from requested values.
8. `episode_id` and `reset_count` advance only for reset environments. No factor changes occur between resets.

## Factor metadata contract

The environment exposes one per-environment snapshot containing:

- `env_id`, canonical `factor_code`, and factor-code bits;
- `mass_scale`, requested/actual mass, requested/actual inertia;
- `friction_condition`, requested/actual object static and dynamic friction;
- actual hand static and dynamic friction and the fixed combine mode;
- `motor_capacity_scale`, requested/actual effort limit for all sixteen joints;
- `episode_id`, `reset_count`, `protocol_id`, `task_id`, and `factor_schema_version`;
- availability flags for every actual field.

Requested values are computed only from the canonical condition table. Actual values are populated only from simulator getters; unavailable reads are represented by an explicit false availability flag, never by copying requested values.

## Error handling

Construction or reset fails explicitly if a factor code is unsupported, condition assignment length is incompatible with `num_envs`, the assignment buffer changes unexpectedly, body/joint/material shapes are unexpected, hand or object material is not uniform where required, a PhysX getter is unavailable, or requested and read-back values disagree beyond the smoke-test tolerance. No fallback randomly samples factor codes or changes controller gains, action semantics, or the baseline task.

## Verification

Static tests verify the canonical condition table, one-factor-only invariants, task registration isolation, disabled ADR/events, fixed object scale, and unchanged baseline registration/configuration.

The authorized smoke test runs four environments with `[000, 100, 010, 001]`, zero actions, and checks:

- requested versus direct PhysX read-back for mass, inertia, both materials, and all joint effort limits;
- only the intended primary factor changes in each environment;
- resetting or assigning one environment does not alter the other three;
- factor values remain constant during the smoke episode;
- reset leaves hand/object initial state valid and finite, returns a finite observation of the declared shape, preserves normal termination output, and correctly resets/increments episode bookkeeping despite `events=None`;
- the original baseline task remains registered and its source/checkpoint files are unmodified.

The smoke test does not evaluate reward, rotation, slip, drops, saturation behavior, or policy performance.
