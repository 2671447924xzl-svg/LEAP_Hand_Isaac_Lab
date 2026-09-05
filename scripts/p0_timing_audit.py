"""LEAP Hand P0-1 commanded/applied field timing audit。"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

from isaaclab.app import AppLauncher


TASK_ID = "Isaac-Reorient-Cube-Leap"  # 冻结的科研任务。
CSV_FIELDS = [
    "run_id",
    "parent_run_id",
    "protocol_id",
    "protocol_phase",
    "control_step",
    "physics_step_global",
    "physics_step_relative",
    "physics_substep",
    "time_relative",
    "joint_name",
    "joint_index",
    "action_index",
    "raw_commanded_action",
    "processed_action",
    "task_commanded_target",
    "task_target_increment",
    "joint_pos_target_buffer",
    "physx_position_target",
    "applied_effort",
    "q_pre",
    "qdot_pre",
    "q_post",
    "qdot_post",
    "delta_q",
    "delta_qdot",
    "tracking_residual",
    "reset_flag",
    "command_transition_flag",
    "action_processing_calls",
    "event_sequence",
]


def build_parser() -> argparse.ArgumentParser:  # 定义独立审计入口，不加载 PPO 参数。
    parser = argparse.ArgumentParser(description="Run the LEAP Hand P0-1 timing audit.")
    parser.add_argument("--task", type=str, default=TASK_ID)
    parser.add_argument("--joint_name", type=str, default="a_0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--parent_run_id", type=str, default="")
    parser.add_argument("--output_dir", type=Path, default=Path("logs/p0_timing_audit"))
    AppLauncher.add_app_launcher_args(parser)
    return parser


ARGS = build_parser().parse_args()
APP_LAUNCHER = AppLauncher(ARGS)
SIMULATION_APP = APP_LAUNCHER.app


import gymnasium as gym  # noqa: E402
import matplotlib  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import warp as wp  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import isaaclab  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from isaacsim.core.version import get_version as get_isaac_sim_version  # noqa: E402

import LEAP_Isaaclab.tasks  # noqa: E402, F401


def sha256_bytes(data: bytes) -> str:  # 返回证据内容的 SHA256。
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:  # 流式计算文件 SHA256。
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_bytes(repo_root: Path, *args: str) -> bytes:  # 读取 Git 证据且不改变工作树。
    result = subprocess.run(
        ["git", *args],
        cwd=repo_root,
        check=True,
        capture_output=True,
    )
    return result.stdout


def tensor_scalar(value: Any, env_index: int, joint_index: int) -> float:  # 统一读取 Torch/Warp 二维张量标量。
    if isinstance(value, torch.Tensor):
        tensor = value
    elif hasattr(value, "torch"):
        tensor = value.torch
    else:
        tensor = wp.to_torch(value)
    return float(tensor[env_index, joint_index].detach().cpu().item())


def float_text(value: float) -> str:  # 使用稳定精度保存原始浮点证据。
    if math.isnan(value):
        return "NaN"
    return f"{value:.12e}"


class CsvSink:  # 逐 physics step 落盘，异常时也保留已生成证据。
    def __init__(self, path: Path):
        self.path = path
        self.stream = path.open("x", newline="", encoding="utf-8")
        self.writer = csv.DictWriter(self.stream, fieldnames=CSV_FIELDS)
        self.writer.writeheader()
        self.stream.flush()

    def write(self, row: dict[str, Any]) -> None:  # 原始行只追加一次，不回写。
        self.writer.writerow(row)
        self.stream.flush()

    def close(self) -> None:  # 完成原始数据文件。
        if not self.stream.closed:
            self.stream.close()


class TimingTracer:  # 仅包装当前实例方法，保持真实 env.step 控制语义。
    def __init__(self, core_env: Any, joint_name: str, sink: CsvSink):
        self.env = core_env
        self.joint_name = joint_name
        self.joint_index = core_env.hand.joint_names.index(joint_name)
        self.action_index = core_env.actuated_dof_indices.index(self.joint_index)
        self.sink = sink
        self.decimation = int(core_env.cfg.decimation)
        self.physics_dt = float(core_env.physics_dt)
        self.enabled = False
        self.active_control = False
        self.run_id = ""
        self.parent_run_id = ARGS.parent_run_id
        self.protocol_id = ""
        self.protocol_phase = ""
        self.control_step = -1
        self.audit_start_physics_step = 0
        self.run_start_steps: dict[str, int] = {}
        self.previous_raw: float | None = None
        self.raw_action = 0.0
        self.processed_action = 0.0
        self.reset_flag = False
        self.command_transition_flag = False
        self.physics_substep = 0
        self.action_processing_calls = 0
        self.pending: dict[str, Any] | None = None
        self.rows_written = 0
        self._originals: dict[str, Any] = {}

    def install(self) -> None:  # 包装五个真实调用点，不复制 DirectRLEnv.step。
        self._originals = {
            "pre": self.env._pre_physics_step,
            "apply": self.env._apply_action,
            "write": self.env.scene.write_data_to_sim,
            "sim_step": self.env.sim.step,
            "update": self.env.scene.update,
        }

        def traced_pre(actions: torch.Tensor) -> Any:
            if not self.enabled:
                return self._originals["pre"](actions)
            if self.active_control:
                raise RuntimeError("上一 control step 尚未结束。")
            self.active_control = True
            self.control_step += 1
            self.physics_substep = 0
            self.action_processing_calls = 1
            self.raw_action = tensor_scalar(actions, 0, self.action_index)
            self.command_transition_flag = self.previous_raw is None or not math.isclose(
                self.raw_action, self.previous_raw, rel_tol=0.0, abs_tol=1.0e-12
            )
            result = self._originals["pre"](actions)
            self.processed_action = tensor_scalar(self.env.actions, 0, self.action_index)
            return result

        def traced_apply() -> Any:
            if not self.enabled or not self.active_control:
                return self._originals["apply"]()
            if self.pending is not None:
                raise RuntimeError("上一个 physics substep 尚未完成。")
            q_pre = tensor_scalar(self.env.hand.data.joint_pos, 0, self.joint_index)
            qdot_pre = tensor_scalar(self.env.hand.data.joint_vel, 0, self.joint_index)
            target_before = tensor_scalar(self.env.cur_targets, 0, self.joint_index)
            result = self._originals["apply"]()
            task_target = tensor_scalar(self.env.cur_targets, 0, self.joint_index)
            buffer_target = tensor_scalar(self.env.hand.data.joint_pos_target, 0, self.joint_index)
            self.pending = {
                "q_pre": q_pre,
                "qdot_pre": qdot_pre,
                "task_target": task_target,
                "target_increment": task_target - target_before,
                "buffer_target": buffer_target,
                "events": ["apply_action"],
            }
            return result

        def traced_write() -> Any:
            result = self._originals["write"]()
            if self.enabled and self.active_control and self.pending is not None:
                self.pending["physx_target"] = tensor_scalar(
                    self.env.hand.root_view.get_dof_position_targets(), 0, self.joint_index
                )
                self.pending["events"].append("write_data_to_sim")
            return result

        def traced_sim_step(*args: Any, **kwargs: Any) -> Any:
            if self.enabled and self.active_control and self.pending is not None:
                if "physx_target" not in self.pending:
                    raise RuntimeError("sim.step 前未读取到 PhysX position target。")
                self.pending["events"].append("sim_step")
            return self._originals["sim_step"](*args, **kwargs)

        def traced_update(dt: float) -> Any:
            result = self._originals["update"](dt)
            if self.enabled and self.active_control and self.pending is not None:
                self.pending["events"].append("scene_update")
                self._finalize_substep()
            return result

        self.env._pre_physics_step = traced_pre
        self.env._apply_action = traced_apply
        self.env.scene.write_data_to_sim = traced_write
        self.env.sim.step = traced_sim_step
        self.env.scene.update = traced_update

    def restore(self) -> None:  # 恢复运行时实例方法。
        if not self._originals:
            return
        self.env._pre_physics_step = self._originals["pre"]
        self.env._apply_action = self._originals["apply"]
        self.env.scene.write_data_to_sim = self._originals["write"]
        self.env.sim.step = self._originals["sim_step"]
        self.env.scene.update = self._originals["update"]

    def start_run(self, run_id: str, protocol_id: str) -> None:  # reset 完成后开启一个独立证据 run。
        self.run_id = run_id
        self.protocol_id = protocol_id
        self.protocol_phase = ""
        self.control_step = -1
        self.previous_raw = None
        self.audit_start_physics_step = int(self.env.sim.get_physics_step_count())
        self.run_start_steps[run_id] = self.audit_start_physics_step
        self.reset_flag = True
        self.enabled = True

    def mark_reset_complete(self) -> None:  # 标记下一 control step 为显式 reset 后首步。
        self.reset_flag = True

    def set_phase(self, phase: str) -> None:  # 记录 command protocol 的冻结阶段。
        self.protocol_phase = phase

    def end_control(self) -> None:  # 校验 env.step 确实产生完整四子步。
        if not self.active_control:
            raise RuntimeError("env.step 未进入 _pre_physics_step。")
        if self.physics_substep != self.decimation:
            raise RuntimeError(
                f"control step 仅记录 {self.physics_substep} 个 physics substeps，预期 {self.decimation}。"
            )
        if self.pending is not None:
            raise RuntimeError("control step 结束时仍有未完成的 physics substep。")
        self.previous_raw = self.raw_action
        self.reset_flag = False
        self.active_control = False

    def _finalize_substep(self) -> None:  # 在 sim.step 与 scene.update 后采集 post-step 状态。
        assert self.pending is not None
        q_post = tensor_scalar(self.env.hand.data.joint_pos, 0, self.joint_index)
        qdot_post = tensor_scalar(self.env.hand.data.joint_vel, 0, self.joint_index)
        physics_global = int(self.env.sim.get_physics_step_count())
        physics_relative = physics_global - self.audit_start_physics_step
        physx_target = float(self.pending["physx_target"])
        event_sequence = ">".join(self.pending["events"])
        row = {
            "run_id": self.run_id,
            "parent_run_id": self.parent_run_id,
            "protocol_id": self.protocol_id,
            "protocol_phase": self.protocol_phase,
            "control_step": self.control_step,
            "physics_step_global": physics_global,
            "physics_step_relative": physics_relative,
            "physics_substep": self.physics_substep,
            "time_relative": float_text(physics_relative * self.physics_dt),
            "joint_name": self.joint_name,
            "joint_index": self.joint_index,
            "action_index": self.action_index,
            "raw_commanded_action": float_text(self.raw_action),
            "processed_action": float_text(self.processed_action),
            "task_commanded_target": float_text(self.pending["task_target"]),
            "task_target_increment": float_text(self.pending["target_increment"]),
            "joint_pos_target_buffer": float_text(self.pending["buffer_target"]),
            "physx_position_target": float_text(physx_target),
            "applied_effort": "NaN",
            "q_pre": float_text(self.pending["q_pre"]),
            "qdot_pre": float_text(self.pending["qdot_pre"]),
            "q_post": float_text(q_post),
            "qdot_post": float_text(qdot_post),
            "delta_q": float_text(q_post - self.pending["q_pre"]),
            "delta_qdot": float_text(qdot_post - self.pending["qdot_pre"]),
            "tracking_residual": float_text(physx_target - q_post),
            "reset_flag": int(self.reset_flag),
            "command_transition_flag": int(self.command_transition_flag),
            "action_processing_calls": self.action_processing_calls,
            "event_sequence": event_sequence,
        }
        self.sink.write(row)
        self.rows_written += 1
        self.physics_substep += 1
        self.pending = None


def fixed_reset(env: gym.Env, tracer: TimingTracer, seed: int) -> None:  # reset 期间暂停 hook 记录但不移除 hook。
    tracer.enabled = False
    tracer.active_control = False
    tracer.pending = None
    env.reset(seed=seed)


def run_action(env: gym.Env, tracer: TimingTracer, value: float, phase: str) -> None:  # Primary path 始终调用真实 env.step。
    action = torch.zeros((1, int(tracer.env.cfg.action_space)), device=tracer.env.device)
    action[0, tracer.action_index] = value
    tracer.set_phase(phase)
    env.step(action)
    tracer.end_control()


def run_step_pulse_protocol(env: gym.Env, tracer: TimingTracer, seed: int, repetition: int) -> None:  # 执行冻结的正负阶跃协议。
    fixed_reset(env, tracer, seed)
    tracer.start_run(f"step_pulse_r{repetition:02d}", "step_pulse_timing")
    schedule = [
        (0.0, 8, "settle_zero"),
        (0.3, 3, "positive_step"),
        (0.0, 2, "positive_hold"),
        (-0.3, 3, "negative_step"),
        (0.0, 2, "negative_hold"),
    ]
    for value, count, phase in schedule:
        for _ in range(count):
            run_action(env, tracer, value, phase)


def run_reset_protocol(env: gym.Env, tracer: TimingTracer, seed: int, repetition: int) -> None:  # 检查旧 +0.8 action 是否跨 reset 泄漏。
    fixed_reset(env, tracer, seed)
    tracer.start_run(f"reset_contamination_r{repetition:02d}", "reset_contamination")
    run_action(env, tracer, 0.8, "pre_reset_pulse")
    fixed_reset(env, tracer, seed)
    tracer.enabled = True
    tracer.mark_reset_complete()
    run_action(env, tracer, 0.0, "post_reset_zero")


def read_rows(path: Path) -> list[dict[str, str]]:  # 后续评估和绘图只读取已保存 raw CSV。
    with path.open("r", newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def as_float(row: dict[str, str], key: str) -> float:  # 解析 raw CSV 数值。
    return float(row[key])


def result(status: bool, evidence: str) -> dict[str, str]:  # 统一保存 PASS/FAIL 证据。
    return {"status": "PASS" if status else "FAIL", "evidence": evidence}


def evaluate(rows: list[dict[str, str]], decimation: int, moving_average: float) -> dict[str, dict[str, str]]:  # 仅从 raw CSV 计算 P0-1 判据。
    controls: dict[tuple[str, int], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        controls[(row["run_id"], int(row["control_step"]))].append(row)

    pass_1 = all(
        len(group) == decimation and {int(row["action_processing_calls"]) for row in group} == {1}
        for group in controls.values()
    )
    pass_2_error = max(
        abs(as_float(row, "processed_action") - float(np.clip(as_float(row, "raw_commanded_action"), -1.0, 1.0)))
        for row in rows
    )
    pass_3_error = max(
        abs(as_float(row, "task_target_increment") - moving_average * as_float(row, "processed_action"))
        for row in rows
    )
    nonzero_controls = [
        group for group in controls.values() if abs(as_float(group[0], "processed_action")) > 1.0e-12
    ]
    pass_4 = all(
        len(group) == decimation
        and all(abs(as_float(row, "task_target_increment")) > 1.0e-12 for row in group)
        for group in nonzero_controls
    )
    target_error = max(
        max(
            abs(as_float(row, "task_commanded_target") - as_float(row, "joint_pos_target_buffer")),
            abs(as_float(row, "joint_pos_target_buffer") - as_float(row, "physx_position_target")),
        )
        for row in rows
    )
    expected_sequence = "apply_action>write_data_to_sim>sim_step>scene_update"
    event_order_ok = all(row["event_sequence"] == expected_sequence for row in rows)

    transition_groups = [
        group
        for group in controls.values()
        if int(group[0]["command_transition_flag"]) == 1
        and abs(as_float(group[0], "processed_action")) > 1.0e-12
    ]
    response_onsets: list[int | None] = []
    for group in transition_groups:
        onset = next(
            (
                int(row["physics_substep"])
                for row in group
                if abs(as_float(row, "delta_q")) > 1.0e-9 or abs(as_float(row, "delta_qdot")) > 1.0e-9
            ),
            None,
        )
        response_onsets.append(onset)
    pass_6 = event_order_ok and bool(response_onsets) and all(onset == 0 for onset in response_onsets)
    pass_7 = event_order_ok and all(
        math.isfinite(as_float(row, field)) for row in rows for field in ("q_pre", "qdot_pre", "q_post", "qdot_post")
    )

    reset_rows = [row for row in rows if row["protocol_phase"] == "post_reset_zero"]
    reset_controls: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in reset_rows:
        reset_controls[row["run_id"]].append(row)
    pass_8 = len(reset_controls) >= 3 and all(
        len(group) == decimation
        and all(
            int(row["reset_flag"]) == 1
            and abs(as_float(row, "raw_commanded_action")) <= 1.0e-12
            and abs(as_float(row, "processed_action")) <= 1.0e-12
            and abs(as_float(row, "task_target_increment")) <= 1.0e-7
            for row in group
        )
        for group in reset_controls.values()
    )

    repeat_sequences: list[list[tuple[Any, ...]]] = []
    for run_id in sorted({row["run_id"] for row in rows if row["protocol_id"] == "step_pulse_timing"}):
        run_rows = [row for row in rows if row["run_id"] == run_id]
        repeat_sequences.append(
            [
                (
                    int(row["control_step"]),
                    int(row["physics_substep"]),
                    row["raw_commanded_action"],
                    row["processed_action"],
                    row["task_commanded_target"],
                    row["joint_pos_target_buffer"],
                    row["physx_position_target"],
                )
                for row in run_rows
            ]
        )
    pass_9 = len(repeat_sequences) >= 3 and all(sequence == repeat_sequences[0] for sequence in repeat_sequences[1:])

    return {
        "PASS-1": result(pass_1, f"{len(controls)} control steps; each has one preprocessing call and {decimation} substeps."),
        "PASS-2": result(pass_2_error <= 1.0e-7, f"max |processed-clamp(raw)|={pass_2_error:.3e}."),
        "PASS-3": result(pass_3_error <= 1.0e-6, f"max target-increment error={pass_3_error:.3e} rad."),
        "PASS-4": result(pass_4, f"{len(nonzero_controls)} nonzero controls each produced {decimation} measured target updates."),
        "PASS-5": result(target_error <= 1.0e-6, f"max task/buffer/PhysX target disagreement={target_error:.3e} rad."),
        "PASS-6": result(pass_6, f"event order valid={event_order_ok}; transition response onsets={response_onsets}."),
        "PASS-7": result(pass_7, "q/qdot pre and post are finite and captured after the recorded scene_update order."),
        "PASS-8": result(pass_8, f"{len(reset_controls)} reset-zero controls contain no measured old-action increment."),
        "PASS-9": result(pass_9, f"{len(repeat_sequences)} repeated target traces compared byte-for-byte after CSV serialization."),
    }


def plot_trace(rows: list[dict[str, str]], output_path: Path) -> None:  # 只从 raw CSV 生成论文式 timing trace。
    selected_runs = ["step_pulse_r01", "reset_contamination_r01"]
    figure, axes = plt.subplots(3, 2, figsize=(14, 9), sharex="col", constrained_layout=True)
    for column, run_id in enumerate(selected_runs):
        run_rows = [row for row in rows if row["run_id"] == run_id]
        time = np.array([as_float(row, "time_relative") for row in run_rows])
        raw = np.array([as_float(row, "raw_commanded_action") for row in run_rows])
        processed = np.array([as_float(row, "processed_action") for row in run_rows])
        task_target = np.array([as_float(row, "task_commanded_target") for row in run_rows])
        physx_target = np.array([as_float(row, "physx_position_target") for row in run_rows])
        q_post = np.array([as_float(row, "q_post") for row in run_rows])
        qdot_post = np.array([as_float(row, "qdot_post") for row in run_rows])

        axes[0, column].step(time, raw, where="post", label="raw command", linewidth=1.7)
        axes[0, column].step(time, processed, where="post", label="processed", linestyle="--")
        axes[1, column].plot(time, task_target, label="task target", linewidth=1.7)
        axes[1, column].plot(time, physx_target, label="PhysX target", linestyle="--")
        axes[1, column].plot(time, q_post, label="q post", alpha=0.85)
        axes[2, column].plot(time, qdot_post, label="qdot post", color="tab:purple")

        control_boundaries = [
            as_float(row, "time_relative") for row in run_rows if int(row["physics_substep"]) == 0
        ]
        transitions = [
            as_float(row, "time_relative")
            for row in run_rows
            if int(row["physics_substep"]) == 0 and int(row["command_transition_flag"]) == 1
        ]
        resets = [
            as_float(row, "time_relative")
            for row in run_rows
            if int(row["physics_substep"]) == 0 and int(row["reset_flag"]) == 1
        ]
        for axis in axes[:, column]:
            for boundary in control_boundaries:
                axis.axvline(boundary, color="0.85", linewidth=0.5)
            for transition in transitions:
                axis.axvline(transition, color="tab:orange", linestyle=":", linewidth=1.0)
            for reset in resets:
                axis.axvline(reset, color="tab:red", linestyle="--", linewidth=1.0)
            axis.grid(True, alpha=0.2)

        for row in run_rows:
            axes[0, column].text(
                as_float(row, "time_relative"),
                0.98,
                row["physics_substep"],
                fontsize=6,
                ha="center",
                va="top",
                color="0.35",
                transform=axes[0, column].get_xaxis_transform(),
                clip_on=True,
            )
        axes[0, column].set_title(run_id)
        axes[0, column].set_ylabel("action")
        axes[1, column].set_ylabel("position [rad]")
        axes[2, column].set_ylabel("velocity [rad/s]")
        axes[2, column].set_xlabel("relative simulation time [s]")
        for axis in axes[:, column]:
            axis.legend(loc="best", fontsize=8)

    figure.suptitle("P0-1 Timing Trace: command → target → PhysX → state")
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def configure_environment() -> Any:  # 仅关闭随机扰动，保留正常 reset lifecycle。
    env_cfg = parse_env_cfg(ARGS.task, device=ARGS.device, num_envs=1)
    env_cfg.seed = ARGS.seed
    env_cfg.enable_adr = False
    env_cfg.min_episode_length_s = env_cfg.episode_length_s
    if env_cfg.action_noise_model is not None:
        raise RuntimeError("P0-1 要求 action_noise_model=None。")
    scale_event = env_cfg.events.object_scale_size
    scale_event.params["scale_range"] = (1.0, 1.0)
    return env_cfg


def main() -> int:  # 执行 P0-1，不推进其他科研阶段。
    if ARGS.repetitions < 3:
        raise ValueError("PASS-9 要求 repetitions >= 3。")

    repo_root = Path(__file__).resolve().parents[1]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = ARGS.output_dir.resolve() / timestamp
    output_dir.mkdir(parents=True, exist_ok=False)
    csv_path = output_dir / "p0_1_timing_raw.csv"
    metadata_path = output_dir / "p0_1_metadata.json"
    plot_path = output_dir / "p0_1_timing_trace.png"
    checksum_path = output_dir / "p0_1_checksums.sha256"

    sink = CsvSink(csv_path)
    env = None
    tracer = None
    failure: str | None = None
    try:
        torch.manual_seed(ARGS.seed)
        np.random.seed(ARGS.seed)
        env_cfg = configure_environment()
        env = gym.make(ARGS.task, cfg=env_cfg)
        core_env = env.unwrapped
        if core_env._physics_handles_decimation:
            raise RuntimeError("当前 backend 隐藏 decimation，无法执行本 P0-1 hook 协议。")
        tracer = TimingTracer(core_env, ARGS.joint_name, sink)
        tracer.install()
        for repetition in range(1, ARGS.repetitions + 1):
            run_step_pulse_protocol(env, tracer, ARGS.seed, repetition)
        for repetition in range(1, ARGS.repetitions + 1):
            run_reset_protocol(env, tracer, ARGS.seed, repetition)
    except Exception as exc:
        failure = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        if tracer is not None:
            tracer.restore()
        sink.close()
        if env is not None:
            env.close()

        csv_hash = sha256_file(csv_path)
        rows = read_rows(csv_path)
        pass_results: dict[str, dict[str, str]] = {}
        if failure is None and rows and tracer is not None:
            pass_results = evaluate(rows, tracer.decimation, float(tracer.env.cfg.act_moving_average))
            plot_trace(rows, plot_path)

        git_commit = git_bytes(repo_root, "rev-parse", "HEAD").decode().strip()
        git_status = git_bytes(repo_root, "status", "--porcelain=v1").decode().splitlines()
        dirty_diff = git_bytes(repo_root, "diff", "--binary", "HEAD")
        sim_version = get_isaac_sim_version()[0]
        metadata = {
            "protocol_id": "p0_1_combined_primary",
            "protocols": ["step_pulse_timing", "reset_contamination"],
            "git_commit": git_commit,
            "git_status": git_status,
            "dirty_diff_hash": sha256_bytes(dirty_diff),
            "audit_script_sha256": sha256_file(Path(__file__).resolve()),
            "isaac_lab_version": getattr(isaaclab, "__version__", "unknown"),
            "isaac_sim_version": sim_version,
            "physx_backend": "PhysX",
            "task_id": ARGS.task,
            "physics_dt": None if tracer is None else tracer.physics_dt,
            "decimation": None if tracer is None else tracer.decimation,
            "control_frequency": None if tracer is None else 1.0 / (tracer.physics_dt * tracer.decimation),
            "action_mode": None if tracer is None else tracer.env.cfg.action_type,
            "act_moving_average": None if tracer is None else float(tracer.env.cfg.act_moving_average),
            "ADR_enabled": False,
            "random_event_overrides": {"object_scale_size.scale_range": [1.0, 1.0]},
            "retained_reset_events": [
                "robot_physics_material",
                "robot_joint_stiffness_and_damping",
                "object_physics_material",
                "object_scale_mass",
            ],
            "joint_name": ARGS.joint_name,
            "joint_index": None if tracer is None else tracer.joint_index,
            "action_index": None if tracer is None else tracer.action_index,
            "parent_run_id": ARGS.parent_run_id,
            "seed": ARGS.seed,
            "repetitions": ARGS.repetitions,
            "audit_start_physics_step": None if tracer is None else min(tracer.run_start_steps.values()),
            "run_audit_start_physics_step": {} if tracer is None else tracer.run_start_steps,
            "applied_effort_available": False,
            "applied_effort_encoding": "NaN; PhysX implicit-drive instantaneous motor torque is unavailable.",
            "tracking_residual_definition": "physx_position_target - q_post",
            "q_pre_definition": "before current _apply_action and sim.step",
            "q_post_definition": "after current sim.step and scene.update",
            "reset_flag_definition": "all four substeps of the first control step after explicit reset",
            "environment_modified": False,
            "runtime_instrumentation": True,
            "ppo_used": False,
            "checkpoint_used": False,
            "primary_audit_path": "env.step(action)",
            "supporting_whitebox_trace": False,
            "raw_csv": str(csv_path),
            "raw_csv_sha256": csv_hash,
            "timing_plot": str(plot_path) if plot_path.exists() else None,
            "rows_recorded": len(rows),
            "pass_results": pass_results,
            "overall_status": (
                "ERROR"
                if failure is not None
                else "PASS"
                if pass_results and all(item["status"] == "PASS" for item in pass_results.values())
                else "FAIL"
            ),
            "execution_error": failure,
        }
        with metadata_path.open("x", encoding="utf-8") as stream:
            json.dump(metadata, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
        json_hash = sha256_file(metadata_path)
        with checksum_path.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(f"{csv_hash}  {csv_path.name}\n")
            stream.write(f"{json_hash}  {metadata_path.name}\n")

        print(json.dumps({
            "overall_status": metadata["overall_status"],
            "csv": str(csv_path),
            "metadata": str(metadata_path),
            "plot": str(plot_path) if plot_path.exists() else None,
            "checksums": str(checksum_path),
            "pass_results": pass_results,
        }, ensure_ascii=False, indent=2))

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    finally:
        SIMULATION_APP.close()
