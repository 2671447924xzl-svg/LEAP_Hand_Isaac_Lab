# R7-O0 Phase 4.2b: 6144-env Formal Backward Feasibility Gate

## Status and authority

This design records the user-approved Phase 4.2b execution contract. It extends engineering evidence from commit `f868ecc2df0475a49f12451f0ad56b23fde75444` without changing the frozen O0 scientific protocol. The gate may execute one PPO epoch per branch but may not initiate formal 5,000-epoch training.

## Objective and boundary

Run `history_only` followed by `oracle` in two independent fresh processes. Each process constructs the frozen 6,144-environment formal stack, runs the mixed-factor integrity gate, strict-loads the same 99-D initialization, performs exactly one full PPO epoch, and exits immediately after the resulting optimizer updates.

This is an engineering capacity test. It computes no branch-performance metric, retains no trained checkpoint, performs no evaluation or statistics, and always records `scientific_conclusion_generated=false`.

## Frozen runtime contract

- Environment count: 6,144.
- Fixed D0/D1/D2 assignment: 2,048/2,048/2,048.
- Horizon/minibatch/mini-epochs/sequence: 32/32,768/5/4.
- Network input/action: 99-D/16-D; unchanged shared recurrent actor-critic.
- Seed: 7100 for both processes.
- Configuration source: committed `HEAD` YAML only; the dirty 4,096 minibatch override is audit-only and must not be consumed.
- Checkpoint, network, reward, controller, physics, factor schema, precision and observation/action semantics are immutable.
- The realized formal `max_epochs=5000` value remains unchanged. The feasibility driver limits execution by calling the single-epoch method exactly once, not by changing PPO configuration.

## Implementation boundary

A dedicated `scripts/r7_o0_backward_feasibility.py` imports and reuses the Phase 4.2 formal stack. It does not expose the formal training authorization route and never calls `agent.train()` or `agent.save()`.

The branch worker manually performs the same single-epoch sequence used by RL-Games training:

1. `agent.init_tensors()` through the formal stack;
2. `agent.env_reset()`;
3. exactly one `agent.update_epoch()`;
4. exactly one `agent.train_epoch()`;
5. frame accounting and dataset cleanup;
6. immediate stack and SimulationApp shutdown.

Runtime wrappers are observation-only:

- model forward hook counts 99-D forward calls;
- GRU input-weight gradient hook checks finite/nonzero gradients and Oracle columns 96:99;
- `play_steps`/`play_steps_rnn` wrapper records B0 before rollout and B1 after rollout;
- `calc_gradients` wrapper records B2 before the first backward path;
- `trancate_gradients_and_step` wrapper records B3 at entry, after backward and before gradient clipping/step;
- `optimizer.step` wrapper records B4 after the first real update.

No wrapper changes tensors, losses, gradients, optimizer arguments, or call counts.

## Process-order and failure rules

- Oracle refuses to launch unless the current output root contains a PASS History result from a different process and the same implementation commit.
- Each process applies the existing clean-process gate before AppLauncher construction.
- A History failure ends the phase; Oracle must not run.
- No capacity fallback or automatic protocol mutation is implemented.
- Resource/memory failure at rollout, backward or optimizer boundaries yields `FORMAL_6144_BACKWARD_CAPACITY_BLOCKED` and `HARDWARE_OR_PROTOCOL_REVISION_REQUIRED`.
- A non-resource software failure yields `FORMAL_TRAINING_PIPELINE_BLOCKED`.

Failure evidence contains the exact completed/failed boundary, traceback, CUDA/PhysX message text when present, and the final resource snapshot.

## Artifacts

Root: `logs/r7_o0_formal_training/backward_feasibility/`

- `history_only/result.json`
- `oracle/result.json`
- branch `allocation_timeline.json`, `resource_timeline.json`, `preflight_system_state.json`, and console log
- aggregate `resource_timeline.json`
- `fairness_report.json`
- `checkpoint_hash.json`
- `artifact_checksums.json`
- `decision.json`

The aggregate fairness report checks equal frozen contracts, seed, factor counts, initialization/model hashes, process isolation, exactly one epoch, positive update evidence, and the permitted branch context difference only.

## Acceptance

Each branch must establish environment creation, mixed-factor integrity, 99-D input, strict checkpoint load, fresh optimizer state, positive forward/finite/nonzero gradient/optimizer counters, populated optimizer state, changed model hash, no prohibited failure signature, and no retained trained checkpoint. Oracle additionally requires a nonzero gradient on GRU input columns 96:99.

Both branches passing yields:

```text
FORMAL_6144_BACKWARD_PASS
READY_FOR_FORMAL_O0_TRAINING
```

Passing does not authorize formal O0 training.

