"""Offline H1 ambiguity-candidate screening over authoritative Pre-B0 trajectories.

This script never starts Isaac Sim, changes the task, or claims formal H1 proof.
It calibrates exploratory thresholds only from same-dynamics trajectory pairs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


PROTOCOL_ID = "H1_DYNAMICS_AMBIGUITY_CANDIDATE_SCREEN_V1"
PRE_B0_PROTOCOL_ID = "PRE_B0_DYNAMICS_SENSITIVITY_BASELINE_V1"
EXPECTED_RUN_ID = "20260904_113624_524310"
EXPECTED_CONDITIONS = ("D0", "D1", "D2")
EXPECTED_FACTOR_CODES = ("000", "010", "100")
EXPECTED_SEEDS = tuple(range(4200, 4224))
PAIR_IDS = (("D0", "D1"), ("D0", "D2"), ("D1", "D2"))
HISTORY_WINDOWS = (3, 5, 10)
HISTORY_QUANTILE = 0.10
ACTION_QUANTILE = 0.90
PERFORMANCE_QUANTILE = 0.90
MIN_CALIBRATION_SAMPLES = 100
STD_FLOOR = 1.0e-6
MAX_SNAPSHOTS_PER_PAIR_WINDOW = 25
MAX_TOP_CANDIDATES_IN_SUMMARY = 20
REFERENCE_CHECKPOINT = Path("logs/rl_games/leap_hand_reorient/pretrained/nn/leap_hand_reorient.pth")
REFERENCE_CHECKPOINT_SHA256 = "46675eecc4e51372ec2c336d9637ef588a61b80e24aeb796de263bd1e75ce364"
DEFAULT_INPUT_DIR = Path("logs/b0_dynamics_sensitivity") / EXPECTED_RUN_ID
DEFAULT_OUTPUT_ROOT = Path("logs/h1_dynamics_ambiguity_search")
REQUIRED_ARRAYS = (
    "valid_mask",
    "policy_obs",
    "raw_action",
    "q",
    "qdot",
    "object_pose",
    "object_velocity",
    "cube_rotation_angle",
    "contact_state",
    "contact_force_norm_n",
    "drop_event",
    "episode_time_s",
    "condition_ids",
    "factor_codes",
    "seeds",
    "planned_horizon_steps",
    "completed_steps",
    "initial_state_hashes",
    "control_dt",
)
PROTECTED_PATHS = (
    "scripts/p0_timing_audit.py",
    "scripts/p0_physics_validation.py",
    "scripts/p0_cri_factor_smoke.py",
    "scripts/p0_friction_process_isolated_qualification.py",
    "scripts/b0_dynamics_sensitivity_baseline.py",
    "source/LEAP_Isaaclab/LEAP_Isaaclab/tasks/leap_hand_reorient/leap_hand_env_cfg.py",
    "source/LEAP_Isaaclab/LEAP_Isaaclab/tasks/leap_hand_reorient/reorientation_env.py",
    "source/LEAP_Isaaclab/LEAP_Isaaclab/tasks/leap_hand_reorient_cri/factor_conditions.py",
    "source/LEAP_Isaaclab/LEAP_Isaaclab/tasks/leap_hand_reorient_cri/leap_hand_cri_env_cfg.py",
    "source/LEAP_Isaaclab/LEAP_Isaaclab/tasks/leap_hand_reorient_cri/reorientation_cri_env.py",
    "source/LEAP_Isaaclab/LEAP_Isaaclab/tasks/leap_hand_reorient/agents/rl_games_ppo_cfg.yaml",
    REFERENCE_CHECKPOINT.as_posix(),
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Offline H1 dynamics-ambiguity candidate screen")
    parser.add_argument("--input_dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output_root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--self_test", action="store_true")
    parser.add_argument("--static_only", action="store_true")
    return parser


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_safe(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json_create(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(json_safe(payload), stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def git_text(repo_root: Path, *args: str) -> str:
    completed = subprocess.run(
        ("git", "-C", str(repo_root), *args), capture_output=True, text=True, check=False
    )
    return completed.stdout.strip() if completed.returncode == 0 else f"UNAVAILABLE:{completed.stderr.strip()}"


def protected_hashes(repo_root: Path) -> dict[str, str | None]:
    return {
        relative: sha256_file(repo_root / relative) if (repo_root / relative).is_file() else None
        for relative in PROTECTED_PATHS
    }


def validate_constants() -> None:
    assert HISTORY_WINDOWS == (3, 5, 10)
    assert EXPECTED_SEEDS == tuple(range(4200, 4224))
    assert len(EXPECTED_SEEDS) == 24
    assert 0.0 < HISTORY_QUANTILE < 0.5
    assert 0.5 < ACTION_QUANTILE < 1.0
    assert 0.5 < PERFORMANCE_QUANTILE < 1.0
    assert "PASS" not in ("candidate_found", "no_candidate", "inconclusive")


def validate_input_summary(summary: dict[str, Any]) -> None:
    assert summary.get("protocol_id") == PRE_B0_PROTOCOL_ID
    assert summary.get("run_id") == EXPECTED_RUN_ID
    assert summary.get("overall_status") == "COMPLETE"
    assert summary.get("scientific_eligible") is True
    pairing = summary.get("pairing", {})
    assert pairing.get("paired_seed_count") == 24
    assert pairing.get("initial_state_hashes_match") is True
    assert pairing.get("native_horizon_steps_match") is True
    assert pairing.get("condition_processes_unique") is True


def validate_arrays(arrays: dict[str, np.ndarray]) -> dict[str, Any]:
    missing = sorted(set(REQUIRED_ARRAYS) - set(arrays))
    if missing:
        raise AssertionError(f"missing arrays: {missing}")
    conditions = tuple(str(item) for item in arrays["condition_ids"].tolist())
    factors = tuple(str(item) for item in arrays["factor_codes"].tolist())
    seeds = tuple(int(item) for item in arrays["seeds"].tolist())
    if conditions != EXPECTED_CONDITIONS:
        raise AssertionError(f"condition_ids={conditions}")
    if factors != EXPECTED_FACTOR_CODES:
        raise AssertionError(f"factor_codes={factors}")
    if seeds != EXPECTED_SEEDS:
        raise AssertionError(f"seeds={seeds}")
    if arrays["policy_obs"].shape[:3] != arrays["valid_mask"].shape:
        raise AssertionError("policy_obs/valid_mask shape mismatch")
    if arrays["raw_action"].shape[:3] != arrays["valid_mask"].shape:
        raise AssertionError("raw_action/valid_mask shape mismatch")
    if arrays["policy_obs"].shape[-1] != 96 or arrays["raw_action"].shape[-1] != 16:
        raise AssertionError("unexpected observation/action shape")
    if not np.all(arrays["initial_state_hashes"] == arrays["initial_state_hashes"][0:1]):
        raise AssertionError("initial state hashes do not match")
    if not np.all(arrays["planned_horizon_steps"] == arrays["planned_horizon_steps"][0:1]):
        raise AssertionError("planned horizons do not match")
    for condition in range(3):
        for episode in range(24):
            mask = arrays["valid_mask"][condition, episode]
            count = int(mask.sum())
            if count != int(arrays["completed_steps"][condition, episode]):
                raise AssertionError("valid mask/completed_steps mismatch")
            if not np.all(mask[:count]) or np.any(mask[count:]):
                raise AssertionError("valid mask is not a contiguous prefix")
            for key in REQUIRED_ARRAYS[1:12]:
                values = arrays[key][condition, episode, :count]
                if np.issubdtype(values.dtype, np.floating) and not np.all(np.isfinite(values)):
                    raise AssertionError(f"non-finite values in {key}")
    return {
        "conditions": conditions,
        "factor_codes": factors,
        "seeds": seeds,
        "policy_observation_dim": int(arrays["policy_obs"].shape[-1]),
        "action_dim": int(arrays["raw_action"].shape[-1]),
        "initial_state_hashes_match": True,
        "planned_horizons_match": True,
    }


def compute_observation_standardization(arrays: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, int]:
    count = 0
    total = np.zeros(arrays["policy_obs"].shape[-1], dtype=np.float64)
    total_sq = np.zeros_like(total)
    for condition in range(3):
        for episode in range(24):
            n = int(arrays["completed_steps"][condition, episode])
            values = arrays["policy_obs"][condition, episode, :n].astype(np.float64)
            total += values.sum(axis=0)
            total_sq += np.square(values).sum(axis=0)
            count += n
    mean = total / count
    variance = np.maximum(total_sq / count - np.square(mean), 0.0)
    std = np.maximum(np.sqrt(variance), STD_FLOOR)
    return mean, std, count


def rolling_pair_distances(
    obs_left: np.ndarray,
    obs_right: np.ndarray,
    action_left: np.ndarray,
    action_right: np.ndarray,
    window: int,
    obs_std: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = min(len(obs_left), len(obs_right), len(action_left), len(action_right))
    if n < 2 * window:
        return (
            np.empty(0, dtype=np.int32),
            np.empty(0, dtype=np.float64),
            np.empty(0, dtype=np.float64),
        )
    normalized_delta = (obs_left[:n].astype(np.float64) - obs_right[:n].astype(np.float64)) / obs_std
    history_step_sq = np.mean(np.square(normalized_delta), axis=1)
    action_step_sq = np.mean(
        np.square(action_left[:n].astype(np.float64) - action_right[:n].astype(np.float64)), axis=1
    )
    history_cumsum = np.concatenate(([0.0], np.cumsum(history_step_sq)))
    action_cumsum = np.concatenate(([0.0], np.cumsum(action_step_sq)))
    endpoints = np.arange(window - 1, n - window, dtype=np.int32)
    history_sum = history_cumsum[endpoints + 1] - history_cumsum[endpoints - window + 1]
    future_action_sum = action_cumsum[endpoints + window + 1] - action_cumsum[endpoints + 1]
    return endpoints, np.sqrt(history_sum / window), np.sqrt(future_action_sum / window)


def episode_j_rot(arrays: dict[str, np.ndarray]) -> np.ndarray:
    result = np.full((3, 24), np.nan, dtype=np.float64)
    for condition in range(3):
        for episode in range(24):
            n = int(arrays["completed_steps"][condition, episode])
            last = n - 1
            duration = float(arrays["episode_time_s"][condition, episode, last])
            if duration <= 0.0:
                raise AssertionError("non-positive episode duration")
            result[condition, episode] = float(
                arrays["cube_rotation_angle"][condition, episode, last] / duration
            )
    return result


def calibrate_negative_controls(
    arrays: dict[str, np.ndarray], obs_std: np.ndarray, j_rot: np.ndarray
) -> tuple[dict[int, dict[str, Any]], dict[str, np.ndarray], np.ndarray]:
    calibration: dict[int, dict[str, Any]] = {}
    calibration_arrays: dict[str, np.ndarray] = {}
    performance_gaps: list[float] = []
    for condition in range(3):
        for episode in range(24):
            neighbor = (episode + 1) % 24  # 固定环形配对，避免结果驱动选邻居。
            performance_gaps.append(abs(float(j_rot[condition, episode] - j_rot[condition, neighbor])))
    performance = np.asarray(performance_gaps, dtype=np.float64)
    epsilon_j = float(np.quantile(performance, PERFORMANCE_QUANTILE))
    for window in HISTORY_WINDOWS:
        history_parts: list[np.ndarray] = []
        action_parts: list[np.ndarray] = []
        for condition in range(3):
            for episode in range(24):
                neighbor = (episode + 1) % 24  # 同动力学、不同seed负对照。
                n_left = int(arrays["completed_steps"][condition, episode])
                n_right = int(arrays["completed_steps"][condition, neighbor])
                endpoints, history, action = rolling_pair_distances(
                    arrays["policy_obs"][condition, episode, :n_left],
                    arrays["policy_obs"][condition, neighbor, :n_right],
                    arrays["raw_action"][condition, episode, :n_left],
                    arrays["raw_action"][condition, neighbor, :n_right],
                    window,
                    obs_std,
                )
                if len(endpoints):
                    history_parts.append(history)
                    action_parts.append(action)
        history_all = np.concatenate(history_parts)
        action_all = np.concatenate(action_parts)
        epsilon_h = float(np.quantile(history_all, HISTORY_QUANTILE))
        similarity_mask = history_all <= epsilon_h
        similarity_count = int(similarity_mask.sum())
        if similarity_count < MIN_CALIBRATION_SAMPLES:
            raise AssertionError(
                f"window {window}: only {similarity_count} similarity-qualified calibration samples"
            )
        epsilon_a = float(np.quantile(action_all[similarity_mask], ACTION_QUANTILE))
        calibration[window] = {
            "history_window_steps": window,
            "future_action_window_steps": window,
            "history_quantile": HISTORY_QUANTILE,
            "action_quantile": ACTION_QUANTILE,
            "performance_quantile": PERFORMANCE_QUANTILE,
            "epsilon_history": epsilon_h,
            "epsilon_future_action": epsilon_a,
            "epsilon_performance_j_rot": epsilon_j,
            "negative_control_samples": int(len(history_all)),
            "similarity_qualified_samples": similarity_count,
            "pairing": "within-condition distinct-seed circular shift +1, same timestep",
        }
        calibration_arrays[f"history_distance_w{window}"] = history_all.astype(np.float32)
        calibration_arrays[f"future_action_divergence_w{window}"] = action_all.astype(np.float32)
        calibration_arrays[f"history_qualified_w{window}"] = similarity_mask
    calibration_arrays["same_dynamics_performance_gap"] = performance.astype(np.float32)
    return calibration, calibration_arrays, performance


def search_candidates(
    arrays: dict[str, np.ndarray],
    obs_std: np.ndarray,
    j_rot: np.ndarray,
    calibration: dict[int, dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    condition_index = {name: index for index, name in enumerate(EXPECTED_CONDITIONS)}
    candidates: list[dict[str, Any]] = []
    diagnostics: dict[str, Any] = {}
    for left_name, right_name in PAIR_IDS:
        left = condition_index[left_name]
        right = condition_index[right_name]
        pair_id = f"{left_name}-{right_name}"
        diagnostics[pair_id] = {}
        for window in HISTORY_WINDOWS:
            epsilon_h = float(calibration[window]["epsilon_history"])
            epsilon_a = float(calibration[window]["epsilon_future_action"])
            epsilon_j = float(calibration[window]["epsilon_performance_j_rot"])
            raw_count = 0
            relevant_count = 0
            evaluated_count = 0
            seed_coverage: set[int] = set()
            for episode, seed in enumerate(EXPECTED_SEEDS):
                n_left = int(arrays["completed_steps"][left, episode])
                n_right = int(arrays["completed_steps"][right, episode])
                endpoints, history, action = rolling_pair_distances(
                    arrays["policy_obs"][left, episode, :n_left],
                    arrays["policy_obs"][right, episode, :n_right],
                    arrays["raw_action"][left, episode, :n_left],
                    arrays["raw_action"][right, episode, :n_right],
                    window,
                    obs_std,
                )
                evaluated_count += len(endpoints)
                mask = (history <= epsilon_h) & (action >= epsilon_a)
                performance_gap = abs(float(j_rot[left, episode] - j_rot[right, episode]))
                performance_relevant = performance_gap >= epsilon_j
                for local_index in np.flatnonzero(mask):
                    endpoint = int(endpoints[local_index])
                    raw_count += 1
                    relevant_count += int(performance_relevant)
                    seed_coverage.add(seed)
                    history_distance = float(history[local_index])
                    action_divergence = float(action[local_index])
                    score = (
                        epsilon_h / max(history_distance, 1.0e-12)
                        * action_divergence / max(epsilon_a, 1.0e-12)
                        * performance_gap / max(epsilon_j, 1.0e-12)
                    )
                    candidates.append(
                        {
                            "pair_id": pair_id,
                            "left_condition": left_name,
                            "right_condition": right_name,
                            "left_factor_code": EXPECTED_FACTOR_CODES[left],
                            "right_factor_code": EXPECTED_FACTOR_CODES[right],
                            "seed": seed,
                            "episode_index": episode,
                            "history_window_steps": window,
                            "endpoint_step": endpoint,
                            "future_start_step": endpoint + 1,
                            "future_end_step_exclusive": endpoint + window + 1,
                            "history_distance": history_distance,
                            "epsilon_history": epsilon_h,
                            "future_action_divergence": action_divergence,
                            "epsilon_future_action": epsilon_a,
                            "j_rot_left": float(j_rot[left, episode]),
                            "j_rot_right": float(j_rot[right, episode]),
                            "performance_gap_j_rot": performance_gap,
                            "epsilon_performance_j_rot": epsilon_j,
                            "performance_relevant": bool(performance_relevant),
                            "screen_score": float(score),
                            "formal_crda": False,
                        }
                    )
            diagnostics[pair_id][str(window)] = {
                "evaluated_windows": evaluated_count,
                "raw_candidate_count": raw_count,
                "performance_relevant_candidate_count": relevant_count,
                "candidate_seed_count": len(seed_coverage),
                "candidate_seeds": sorted(seed_coverage),
            }
    candidates.sort(key=lambda item: item["screen_score"], reverse=True)
    for index, candidate in enumerate(candidates):
        candidate["candidate_id"] = f"candidate_{index:06d}"  # 跨CSV、summary与NPZ保持稳定主键。
    return candidates, diagnostics


def write_candidate_csv(path: Path, candidates: list[dict[str, Any]]) -> None:
    fieldnames = [
        "candidate_id",
        "pair_id",
        "left_condition",
        "right_condition",
        "left_factor_code",
        "right_factor_code",
        "seed",
        "episode_index",
        "history_window_steps",
        "endpoint_step",
        "future_start_step",
        "future_end_step_exclusive",
        "history_distance",
        "epsilon_history",
        "future_action_divergence",
        "epsilon_future_action",
        "j_rot_left",
        "j_rot_right",
        "performance_gap_j_rot",
        "epsilon_performance_j_rot",
        "performance_relevant",
        "screen_score",
        "formal_crda",
    ]
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(candidates)


def snapshot_candidates(
    arrays: dict[str, np.ndarray], candidates: list[dict[str, Any]], output_path: Path
) -> int:
    condition_index = {name: index for index, name in enumerate(EXPECTED_CONDITIONS)}
    selected: list[dict[str, Any]] = []
    group_counts: dict[tuple[str, int], int] = {}
    prioritized = sorted(
        candidates,
        key=lambda item: (bool(item["performance_relevant"]), float(item["screen_score"])),
        reverse=True,
    )
    for candidate in prioritized:
        group = (candidate["pair_id"], int(candidate["history_window_steps"]))
        if group_counts.get(group, 0) >= MAX_SNAPSHOTS_PER_PAIR_WINDOW:
            continue
        selected.append(candidate)
        group_counts[group] = group_counts.get(group, 0) + 1
    count = len(selected)
    max_window = max(HISTORY_WINDOWS)
    history = np.full((count, 2, max_window, 96), np.nan, dtype=np.float32)
    future_action = np.full((count, 2, max_window, 16), np.nan, dtype=np.float32)
    q = np.empty((count, 2, 16), dtype=np.float32)
    qdot = np.empty_like(q)
    object_pose = np.empty((count, 2, 7), dtype=np.float32)
    object_velocity = np.empty((count, 2, 6), dtype=np.float32)
    contact_state = np.empty((count, 2), dtype=bool)
    contact_force = np.empty((count, 2), dtype=np.float32)
    rotation = np.empty((count, 2), dtype=np.float32)
    drop = np.empty((count, 2), dtype=bool)
    for row, candidate in enumerate(selected):
        episode = int(candidate["episode_index"])
        endpoint = int(candidate["endpoint_step"])
        window = int(candidate["history_window_steps"])
        for side, name in enumerate((candidate["left_condition"], candidate["right_condition"])):
            condition = condition_index[str(name)]
            start = endpoint - window + 1
            future_start = endpoint + 1
            history[row, side, :window] = arrays["policy_obs"][condition, episode, start : endpoint + 1]
            future_action[row, side, :window] = arrays["raw_action"][
                condition, episode, future_start : future_start + window
            ]
            q[row, side] = arrays["q"][condition, episode, endpoint]
            qdot[row, side] = arrays["qdot"][condition, episode, endpoint]
            object_pose[row, side] = arrays["object_pose"][condition, episode, endpoint]
            object_velocity[row, side] = arrays["object_velocity"][condition, episode, endpoint]
            contact_state[row, side] = arrays["contact_state"][condition, episode, endpoint]
            contact_force[row, side] = arrays["contact_force_norm_n"][condition, episode, endpoint]
            rotation[row, side] = arrays["cube_rotation_angle"][condition, episode, endpoint]
            drop[row, side] = arrays["drop_event"][condition, episode, endpoint]
    np.savez_compressed(
        output_path,
        snapshot_type=np.asarray("TRAJECTORY_STATE_SNAPSHOT_NOT_SIMULATOR_REPLAY_COMPLETE"),
        candidate_id=np.asarray([item["candidate_id"] for item in selected], dtype="U32"),
        pair_id=np.asarray([item["pair_id"] for item in selected], dtype="U8"),
        conditions=np.asarray(
            [[item["left_condition"], item["right_condition"]] for item in selected], dtype="U2"
        ),
        factor_codes=np.asarray(
            [[item["left_factor_code"], item["right_factor_code"]] for item in selected], dtype="U3"
        ),
        seed=np.asarray([item["seed"] for item in selected], dtype=np.int64),
        episode_index=np.asarray([item["episode_index"] for item in selected], dtype=np.int32),
        history_window_steps=np.asarray([item["history_window_steps"] for item in selected], dtype=np.int32),
        endpoint_step=np.asarray([item["endpoint_step"] for item in selected], dtype=np.int32),
        history_distance=np.asarray([item["history_distance"] for item in selected], dtype=np.float32),
        future_action_divergence=np.asarray(
            [item["future_action_divergence"] for item in selected], dtype=np.float32
        ),
        performance_gap_j_rot=np.asarray(
            [item["performance_gap_j_rot"] for item in selected], dtype=np.float32
        ),
        performance_relevant=np.asarray([item["performance_relevant"] for item in selected], dtype=bool),
        screen_score=np.asarray([item["screen_score"] for item in selected], dtype=np.float32),
        policy_observation_history=history,
        future_raw_action=future_action,
        q=q,
        qdot=qdot,
        object_pose=object_pose,
        object_velocity=object_velocity,
        contact_state=contact_state,
        contact_force_norm_n=contact_force,
        cube_rotation_angle=rotation,
        drop_event=drop,
    )
    return count


def self_test() -> None:
    obs_left = np.zeros((20, 4), dtype=np.float32)
    obs_right = np.zeros_like(obs_left)
    action_left = np.zeros((20, 2), dtype=np.float32)
    action_right = np.zeros_like(action_left)
    action_right[4:7] = 1.0
    endpoints, history, action = rolling_pair_distances(
        obs_left, obs_right, action_left, action_right, 3, np.ones(4, dtype=np.float64)
    )
    assert len(endpoints) == 15
    assert np.allclose(history, 0.0)
    endpoint_index = int(np.flatnonzero(endpoints == 3)[0])
    assert math.isclose(float(action[endpoint_index]), 1.0, abs_tol=1.0e-12)
    forbidden_claim = "H1 " + "PASS"  # 避免脚本意外输出正式H1结论。
    assert forbidden_claim not in Path(__file__).read_text(encoding="utf-8")
    print("H1_CANDIDATE_SCREEN_SELF_TEST_PASS")


def run_analysis(repo_root: Path, input_dir: Path, output_dir: Path) -> dict[str, Any]:
    trajectory_path = input_dir / "trajectory_data.npz"
    pre_b0_summary_path = input_dir / "summary.json"
    checkpoint_path = repo_root / REFERENCE_CHECKPOINT
    for path in (trajectory_path, pre_b0_summary_path, checkpoint_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    checkpoint_hash = sha256_file(checkpoint_path)
    if checkpoint_hash != REFERENCE_CHECKPOINT_SHA256:
        raise AssertionError(f"checkpoint SHA mismatch: {checkpoint_hash}")
    pre_b0_summary = read_json(pre_b0_summary_path)
    validate_input_summary(pre_b0_summary)
    with np.load(trajectory_path, allow_pickle=False) as loaded:
        arrays = {key: loaded[key] for key in loaded.files}
    input_validation = validate_arrays(arrays)
    obs_mean, obs_std, standardization_count = compute_observation_standardization(arrays)
    j_rot = episode_j_rot(arrays)
    calibration, calibration_arrays, _ = calibrate_negative_controls(arrays, obs_std, j_rot)
    candidates, diagnostics = search_candidates(arrays, obs_std, j_rot, calibration)
    candidate_csv_path = output_dir / "candidate_pairs.csv"
    snapshot_path = output_dir / "candidate_state_snapshots.npz"
    calibration_path = output_dir / "negative_control_calibration.npz"
    write_candidate_csv(candidate_csv_path, candidates)
    snapshot_count = snapshot_candidates(arrays, candidates, snapshot_path)
    np.savez_compressed(
        calibration_path,
        observation_mean=obs_mean.astype(np.float32),
        observation_std=obs_std.astype(np.float32),
        **calibration_arrays,
    )
    relevant_candidates = [item for item in candidates if item["performance_relevant"]]
    screen_status = "candidate_found" if relevant_candidates else "no_candidate"
    manifest = {
        "protocol_id": PROTOCOL_ID,
        "created_local": datetime.now().isoformat(),
        "claim_role": "exploratory candidate prioritization only",
        "formal_h1_status": "NOT_TESTED_O0_COUNTERFACTUAL_REQUIRED",
        "input_dir": str(input_dir.resolve()),
        "input_sha256": {
            "trajectory_data.npz": sha256_file(trajectory_path),
            "summary.json": sha256_file(pre_b0_summary_path),
            "checkpoint": checkpoint_hash,
        },
        "history_windows_steps": HISTORY_WINDOWS,
        "future_action_windows_steps": HISTORY_WINDOWS,
        "history_distance": "standardized RMS L2 over policy-observation history",
        "future_action_divergence": "RMS L2 over next window-length raw-action segment",
        "negative_control_pairing": "within-condition distinct-seed circular shift +1, same timestep",
        "calibration_quantiles": {
            "history": HISTORY_QUANTILE,
            "future_action": ACTION_QUANTILE,
            "performance_j_rot": PERFORMANCE_QUANTILE,
        },
        "standardization": {
            "sample_count": standardization_count,
            "std_floor": STD_FLOOR,
            "mean_sha256": hashlib.sha256(obs_mean.tobytes()).hexdigest(),
            "std_sha256": hashlib.sha256(obs_std.tobytes()).hexdigest(),
        },
        "snapshot_semantics": "trajectory state only; not replay-complete simulator state",
    }
    manifest_path = output_dir / "analysis_manifest.json"
    write_json_create(manifest_path, manifest)
    summary = {
        "protocol_id": PROTOCOL_ID,
        "candidate_screen_status": screen_status,
        "formal_h1_status": "NOT_TESTED_O0_COUNTERFACTUAL_REQUIRED",
        "h1_pass_claim_allowed": False,
        "input_validation": input_validation,
        "checkpoint": {
            "path": str(checkpoint_path.resolve()),
            "expected_sha256": REFERENCE_CHECKPOINT_SHA256,
            "actual_sha256": checkpoint_hash,
            "identity_status": "PASS",
            "policy_instantiated": False,
            "reason": "offline analysis reuses recorded frozen-policy actions",
        },
        "thresholds": {str(window): calibration[window] for window in HISTORY_WINDOWS},
        "cross_context_diagnostics": diagnostics,
        "candidate_counts": {
            "raw": len(candidates),
            "performance_relevant": len(relevant_candidates),
            "trajectory_state_snapshots": snapshot_count,
        },
        "top_candidates": relevant_candidates[:MAX_TOP_CANDIDATES_IN_SUMMARY]
        or candidates[:MAX_TOP_CANDIDATES_IN_SUMMARY],
        "j_rot_radps": {
            name: [float(value) for value in j_rot[index]]
            for index, name in enumerate(EXPECTED_CONDITIONS)
        },
        "limitations": [
            "same-dynamics negative controls use distinct seeds, not identical-state repeated rollouts",
            "recorded policy-action divergence does not identify the required or optimal control action",
            "trajectory-state snapshots are not replay-complete simulator snapshots",
            "formal H1 requires O0 context value and own/cross-action counterfactual consequence",
        ],
        "next_authorized_action": "review candidates and O0 interface; do not run O0 without approval",
    }
    return summary


def write_checksums(
    repo_root: Path, input_dir: Path, output_dir: Path, protected_before: dict[str, str | None]
) -> dict[str, Any]:
    generated = {}
    for path in sorted(output_dir.iterdir()):
        if path.is_file() and path.name != "artifact_checksums.json":
            generated[path.name] = sha256_file(path)
    protected_after = protected_hashes(repo_root)
    payload = {
        "generated_artifacts": generated,
        "source_inputs": {
            str((input_dir / "trajectory_data.npz").resolve()): sha256_file(input_dir / "trajectory_data.npz"),
            str((input_dir / "summary.json").resolve()): sha256_file(input_dir / "summary.json"),
            str((repo_root / REFERENCE_CHECKPOINT).resolve()): sha256_file(repo_root / REFERENCE_CHECKPOINT),
            str(Path(__file__).resolve()): sha256_file(Path(__file__).resolve()),
        },
        "protected_before": protected_before,
        "protected_after": protected_after,
        "protected_unchanged": protected_before == protected_after,
    }
    write_json_create(output_dir / "artifact_checksums.json", payload)
    return payload


def main() -> int:
    args = build_parser().parse_args()
    validate_constants()
    if args.static_only:
        print("H1_CANDIDATE_SCREEN_STATIC_CONTRACT_PASS")
        return 0
    if args.self_test:
        self_test()
        return 0
    repo_root = Path(__file__).resolve().parents[1]
    input_dir = args.input_dir if args.input_dir.is_absolute() else repo_root / args.input_dir
    output_root = args.output_root if args.output_root.is_absolute() else repo_root / args.output_root
    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output_dir = output_root / run_id
    output_dir.mkdir(parents=True, exist_ok=False)
    protected_before = protected_hashes(repo_root)
    exit_code = 0
    try:
        summary = run_analysis(repo_root, input_dir, output_dir)
    except Exception as error:  # 始终保留可审计的inconclusive artifact。
        exit_code = 2
        summary = {
            "protocol_id": PROTOCOL_ID,
            "candidate_screen_status": "inconclusive",
            "formal_h1_status": "NOT_TESTED_O0_COUNTERFACTUAL_REQUIRED",
            "h1_pass_claim_allowed": False,
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
        }
    summary["run_id"] = run_id
    summary["repo"] = {
        "root": str(repo_root),
        "branch": git_text(repo_root, "branch", "--show-current"),
        "head": git_text(repo_root, "rev-parse", "HEAD"),
        "status_short": git_text(repo_root, "status", "--short"),
    }
    summary_path = output_dir / "summary.json"
    write_json_create(summary_path, summary)
    checksums = write_checksums(repo_root, input_dir, output_dir, protected_before)
    if not checksums["protected_unchanged"]:
        exit_code = 3
    print(f"candidate_screen_status={summary['candidate_screen_status']}")
    print(f"formal_h1_status={summary['formal_h1_status']}")
    print(f"output_dir={output_dir}")
    print(f"protected_unchanged={checksums['protected_unchanged']}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
