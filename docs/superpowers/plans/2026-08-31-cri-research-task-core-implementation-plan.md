# CRI Research Task Core Implementation Plan

> Status: approved for execution on 2026-08-31. This plan implements the frozen `CRI_RESEARCH_TASK_CORE_IMPLEMENTATION` design only.

## Goal and stop boundary

Implement an isolated `Isaac-Reorient-Cube-Leap-CRI` task that applies deterministic per-environment mass, object-friction, and motor-capacity conditions, reads the resulting values directly from PhysX, and proves the four-environment contract with one zero-action smoke test. Stop after static tests and smoke evidence. Do not train PPO and do not enter B0/O0.

## Frozen constraints

- Keep the baseline task, PPO files, reward, observation, action processing, timing, actuator gains, dt, and decimation unchanged.
- Use `CRI_FACTOR_SCHEMA_V1` exactly: `000`, `100`, `010`, `001` with bit order `[mass, friction, motor_capacity]`.
- Source factor codes only from an explicit per-environment assignment config/buffer. Default all environments to `000`; the smoke test explicitly assigns `[000, 100, 010, 001]`. Never sample factor codes during reset.
- Keep `events=None`, `enable_adr=False`, and fixed cube scale `(1.2, 1.2, 1.2)`.
- Bind hand and object materials explicitly with `friction_combine_mode="min"`; keep hand friction at `0.80/0.80`; vary only object friction between `0.80/0.80` and `0.45/0.45`.
- Apply mass and inertia together with indexed writes; apply per-environment effort limits with the indexed articulation writer.
- Populate actual fields only from direct PhysX getters. Never copy requested values into actual fields.

## Phase 0 — lock API contracts and baseline boundaries

**Read-only references**

- `source/LEAP_Isaaclab/LEAP_Isaaclab/tasks/leap_hand_reorient/__init__.py`
- `source/LEAP_Isaaclab/LEAP_Isaaclab/tasks/leap_hand_reorient/leap_hand_env_cfg.py`
- `source/LEAP_Isaaclab/LEAP_Isaaclab/tasks/leap_hand_reorient/reorientation_env.py`
- `source/LEAP_Isaaclab/LEAP_Isaaclab/assets/leap.py`
- Isaac Lab `DirectRLEnv`, rigid-object, articulation, material-spawner, and event-manager implementations.

**Allowed runtime APIs**

- `RigidObject.set_masses_index`
- `RigidObject.set_inertias_index`
- `RigidObject.root_view.get_masses`
- `RigidObject.root_view.get_inertias`
- `RigidObject.root_view.get_material_properties` / `set_material_properties`
- `Articulation.write_joint_effort_limit_to_sim_index`
- `Articulation.root_view.get_dof_max_forces`
- USD `physxMaterial:frictionCombineMode` attribute read-back

**Protected baseline files**

- `tasks/leap_hand_reorient/**`
- `scripts/rl_games/**`
- all existing checkpoints
- the user-owned `scripts/p0_timing_audit.py`

Record their status/hash evidence in the smoke metadata; do not stage or edit them.

## Phase 1 — canonical factor schema

**Create** `source/LEAP_Isaaclab/LEAP_Isaaclab/tasks/leap_hand_reorient_cri/factor_conditions.py`.

Implement immutable factor records and validators for the four approved codes. Assert one-factor-only differences from `000`, exact friction semantics, exact effective effort limits, and explicit assignment length rules. This module contains no Isaac Sim imports so its invariants can be tested without launching simulation.

**Test** `tests/test_cri_factor_conditions.py`.

Validate code order, values, unsupported-code rejection, singleton broadcast, exact-length acceptance, and all other length rejection.

## Phase 2 — isolated task registration and configuration

**Create**:

- `source/LEAP_Isaaclab/LEAP_Isaaclab/tasks/leap_hand_reorient_cri/__init__.py`
- `source/LEAP_Isaaclab/LEAP_Isaaclab/tasks/leap_hand_reorient_cri/leap_hand_cri_env_cfg.py`

Register only `Isaac-Reorient-Cube-Leap-CRI`; expose no PPO configuration entry point. Subclass `LeapHandEnvCfg`, set `events=None` and `enable_adr=False`, preserve fixed cube geometry, and replace nested hand/object spawn configs with explicit `0.80/0.80`, `min` materials. Add the explicit `factor_codes` config with default `("000",)`.

**Test** `tests/test_cri_task_contract.py` under Isaac Python without launching simulation. Verify baseline and CRI registrations, CRI config values, baseline config isolation, smoke assignment, and absence of reward/observation/action overrides in the CRI subclass.

## Phase 3 — indexed factor application and direct read-back

**Create** `source/LEAP_Isaaclab/LEAP_Isaaclab/tasks/leap_hand_reorient_cri/reorientation_cri_env.py`.

After inherited construction:

1. Validate and freeze explicit per-environment factor labels and bit tensor.
2. Capture nominal mass, inertia, material shape, and joint effort limits from PhysX.
3. Allocate requested/actual/availability tensors and reset bookkeeping.

After each inherited `_reset_idx(env_ids)`:

1. Read factor codes only from the assignment buffer.
2. Compute requested mass and inertia from captured nominal values.
3. Apply indexed mass/inertia, selected object material rows, and indexed effort limits.
4. Read mass, inertia, object/hand material properties, effort limits, and both material combine modes independently from PhysX/USD.
5. Update only reset environments' `episode_id` and `reset_count`.

Expose a cloned snapshot method for smoke logging. Do not override `_pre_physics_step`, `_apply_action`, `_get_observations`, `_get_rewards`, or `_get_dones`.

The inherited history buffer is preserved exactly. The smoke test audits observation finiteness and lifecycle but does not silently clear cross-episode history, because that would change the frozen observation semantics.

## Phase 4 — four-environment read-back smoke

**Create** `scripts/p0_cri_factor_smoke.py`.

Launch one headless environment instance with `num_envs=4`, fixed seed, zero actions, and explicit config assignment `("000", "100", "010", "001")`.

The smoke must verify:

1. Both baseline and CRI task IDs remain registered.
2. Reset returns finite policy observations with declared shape.
3. Hand joint position/velocity and object pose/velocity are finite; initial positions match inherited deterministic reset definitions within tolerance.
4. `episode_length_buf` is zero after reset and increments after one zero-action step; returned termination/truncation tensors are valid and finite.
5. Every requested factor matches its independent direct PhysX read-back.
6. `100`, `010`, and `001` differ from `000` only in their approved primary factor.
7. Reapplying one indexed environment leaves the other three PhysX rows unchanged.
8. Read-back factor values stay constant across the short smoke episode.
9. Hand/object combine-mode USD attributes both equal `min` in every environment.
10. No PPO module or checkpoint is loaded.

Write one CSV row per environment, a JSON report with named checks and environment metadata, and SHA-256 checksums. A failure produces nonzero exit status and preserves diagnostic output.

## Phase 5 — verification and stop

Run in order:

1. Python compile check for new modules, tests, and smoke script.
2. Pure schema unit tests.
3. Isaac-Python static contract tests without simulation.
4. Confirm no conflicting Isaac Sim/Kit process is running.
5. Run exactly one authorized four-environment headless smoke test.
6. Inspect CSV/JSON/checksums and `git diff --check`.
7. Verify protected baseline files and the pre-existing user changes remain untouched.

Pass requires all named static and smoke checks to pass with actual-value availability true. If a direct getter, indexed write, or material path is incompatible with the runtime, stop at this stage with evidence; do not substitute commanded/requested values and do not proceed to PPO, B0, or O0.

