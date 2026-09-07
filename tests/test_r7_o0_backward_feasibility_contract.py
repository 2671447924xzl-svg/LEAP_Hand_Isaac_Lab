"""R7-O0 Phase 4.2b backward feasibility gate 的非 Isaac 合约测试。"""

from __future__ import annotations

import importlib.util
import hashlib
import inspect
import json
import tempfile
import unittest
from argparse import Namespace
from contextlib import nullcontext
from pathlib import Path

import psutil


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/r7_o0_backward_feasibility.py"
SPEC = importlib.util.spec_from_file_location("r7_o0_backward_feasibility", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
feasibility = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(feasibility)


def controlled_invocation() -> dict[str, object]:  # 构造 Windows launch ownership 合同。
    return {
        "script_path": str(SCRIPT),
        "repo_root": str(SCRIPT.parents[1]),
        "output_root": r"C:\evidence\fresh_r1",
        "branch": "history_only",
        "seed": 7100,
    }


def launch_record(
    pid: int,
    ppid: int,
    name: str,
    mode: str,
    create_time: float,
    ancestors: list[int],
    *,
    project_related: bool = True,
    current: bool = False,
) -> dict[str, object]:  # 构造含 ancestry 与 invocation identity 的进程证据。
    identity = controlled_invocation()
    script = identity["script_path"]
    common = (
        f"--branch {identity['branch']} --seed {identity['seed']} "
        f"--repo-root {identity['repo_root']} --output-root {identity['output_root']}"
    )
    if mode == "parent":
        command = f"python scripts\\{Path(str(script)).name} --launch-branch --branch {identity['branch']}"
    elif mode == "cli":
        command = f"cmd /c isaaclab.bat -p {script} --worker {common}"
    elif mode == "kit":
        command = f'kit.exe -c "from isaaclab.cli import cli; cli()" -p {script} --worker {common}'
    elif mode == "venv_cli":
        command = f'python.exe -c "from isaaclab.cli import cli; cli()" -p {script} --worker {common}'
    elif mode == "worker":
        command = f"python {script} --worker {common}"
    else:
        command = f"python {identity['repo_root']}\\child.py"
    return {
        "inspection_available": True,
        "pid": pid,
        "ppid": ppid,
        "name": name,
        "command_line": command,
        "executable": name,
        "working_directory": str(identity["repo_root"]),
        "create_time": create_time,
        "ancestor_pids": ancestors,
        "is_current_worker": current,
        "project_related": project_related,
    }


def valid_launch_records() -> list[dict[str, object]]:  # parent→CLI→kit→worker 的有效链。
    return [
        launch_record(100, 50, "python.exe", "parent", 1000.0, [50]),
        launch_record(101, 100, "cmd.exe", "cli", 1000.1, [100, 50]),
        launch_record(102, 101, "kit.exe", "kit", 1000.2, [101, 100, 50]),
        launch_record(103, 102, "python.exe", "worker", 1000.3, [102, 101, 100, 50], current=True),
    ]


def audit_records(records: list[dict[str, object]]) -> dict[str, object]:  # 统一调用待实现的纯 ownership 审计。
    return feasibility.audit_controlled_launch(
        records,
        expected_parent_pid=100,
        expected_parent_create_time=1000.0,
        current_worker_pid=103,
        current_worker_ancestor_pids={102, 101, 100, 50},
        launch_started_at_epoch=1000.05,
        invocation=controlled_invocation(),
    )


def passing_result(branch: str, process_id: int) -> dict[str, object]:  # 构造最小公平性 PASS 记录。
    checkpoint = {
        "source": {"sha256": feasibility.formal.EXPECTED_SOURCE_SHA256},
        "expanded": {"sha256": feasibility.formal.EXPECTED_EXPANDED_SHA256},
        "tracked_diff": [feasibility.formal.AGENT_YAML_RELATIVE.as_posix()],
        "only_preexisting_yaml_diff": True,
    }
    return {
        "protocol_id": feasibility.PROTOCOL_ID,
        "fairness_schema": feasibility.FAIRNESS_SCHEMA,
        "feasibility_id": feasibility.FEASIBILITY_ID,
        "status": "PASS",
        "branch": branch,
        "seed": 7100,
        "git_commit": "abc",
        "process_id": process_id,
        "run_id": "a" * 32,
        "num_envs": 6144,
        "factor_counts": {"000": 2048, "010": 2048, "100": 2048},
        "model_hash_before": "initial",
        "model_hash_after": f"after-{branch}",
        "scientific_config_sha256": feasibility.scientific_config_sha256(),
        "epochs_completed": 1,
        "frames_completed": 196608,
        "optimizer_state_entries_before": 0,
        "optimizer_state_entries_after": 17,
        "trainable_parameter_count": feasibility.formal.EXPECTED_TRAINABLE_PARAMETERS,
        "state_tensor_element_count": feasibility.formal.EXPECTED_STATE_TENSOR_ELEMENTS,
        "checkpoint_hashes_before": checkpoint,
        "checkpoint_hashes_after": checkpoint,
        "config_source": {
            "committed_sha256": "config",
            "working_tree_consumed_as_config": False,
        },
        "context_contract": {
            "pass": True,
            "source_fields": list(feasibility.formal.CONTEXT_SOURCE_FIELDS),
            "prohibited_fields_consumed": [],
            "requested_values_used": False,
            "factor_codes_used": False,
            "input_shape": [6144, 99],
            "branch": branch,
            "observed_digests_by_code": {"000": f"digest-{branch}"},
        },
        "training_counters": {
            "forward_calls": 1,
            "forward_99d_calls": 1,
            "finite_gradient_calls": 1,
            "nonzero_gradient_calls": 1,
            "optimizer_step_count": 1,
            "context_column_nonzero_gradient_calls": 0 if branch == "history_only" else 1,
        },
        "frozen_contract": feasibility.frozen_contract_record(),
        "no_trained_checkpoint_retained": True,
        "failure_signatures": {"oom": False, "pinned_memory": False, "articulation": False},
        "console_evidence_finalized": True,
        "checks": {"runtime": True, "console_evidence_finalized": True},
        "scientific_conclusion_generated": False,
    }


def write_governed_history(root: Path, record: dict[str, object]) -> None:  # 写入可验证的 History 与 checksum。
    token_digest = hashlib.sha256(b"test-token").hexdigest()
    console_path = root / "history_only" / "console.log"
    console_path.parent.mkdir(parents=True)
    console_path.write_text(
        f"R7_PHASE42B_PARENT_CAPTURE_START token_sha256={token_digest}\n"
        "clean exit\n"
        f"R7_PHASE42B_PARENT_CAPTURE_COMPLETE token_sha256={token_digest} exit_code=0\n",
        encoding="utf-8",
    )
    console_digest = hashlib.sha256(console_path.read_bytes()).hexdigest()
    record["console_sha256"] = console_digest
    record["process_exit_code"] = 0
    record["console_capture"] = {
        "owner": "phase42b_parent_launcher",
        "canonical_path": str(console_path.resolve()),
        "token_sha256": token_digest,
        "start_sentinel_present": True,
        "completion_sentinel_present": True,
    }
    history_path = root / "history_only" / "result.json"
    payload = json.dumps(record, indent=2, sort_keys=True) + "\n"
    history_path.write_text(payload, encoding="utf-8")
    digest = hashlib.sha256(history_path.read_bytes()).hexdigest()
    preflight_path = root / "history_only" / "preflight_system_state.json"
    preflight_path.write_text(
        json.dumps(
            {
                "console_capture_admission": {
                    "owner": "phase42b_parent_launcher",
                    "parent_pid": 700,
                    "token_sha256": token_digest,
                    "token_persisted_in_command_line": False,
                },
                "run_id": record["run_id"],
                "clean_process_gate": {
                    "pass": True,
                    "expected_parent_pid": 700,
                    "controlled_parent_count": 1,
                    "controlled_parent": {"pid": 700, "is_controlled_capture_parent": True},
                },
            }
        ),
        encoding="utf-8",
    )
    preflight_digest = hashlib.sha256(preflight_path.read_bytes()).hexdigest()
    (root / "artifact_checksums.json").write_text(
        json.dumps(
            {
                "artifacts": {
                    "history_only/result.json": digest,
                    "history_only/console.log": console_digest,
                    "history_only/preflight_system_state.json": preflight_digest,
                }
            }
        ),
        encoding="utf-8",
    )


class BackwardFeasibilityContractTest(unittest.TestCase):
    def test_frozen_runtime_contract_is_exact(self) -> None:
        contract = feasibility.frozen_contract_record()
        self.assertEqual(contract["num_envs"], 6144)
        self.assertEqual(contract["factor_counts"], {"000": 2048, "010": 2048, "100": 2048})
        self.assertEqual(contract["horizon_length"], 32)
        self.assertEqual(contract["minibatch_size"], 32768)
        self.assertEqual(contract["mini_epochs"], 5)
        self.assertEqual(contract["seq_length"], 4)
        self.assertEqual(contract["policy_input_dim"], 99)

    def test_resource_boundaries_are_exact(self) -> None:
        self.assertEqual(
            feasibility.BACKWARD_STAGES,
            {
                "B0": "before_rollout",
                "B1": "after_rollout",
                "B2": "before_first_backward",
                "B3": "immediately_after_first_backward",
                "B4": "after_first_optimizer_step",
            },
        )

    def test_only_seed_7100_and_known_branches_are_legal(self) -> None:
        feasibility.validate_cli_contract(Namespace(branch="history_only", seed=7100))
        feasibility.validate_cli_contract(Namespace(branch="oracle", seed=7100))
        with self.assertRaises(ValueError):
            feasibility.validate_cli_contract(Namespace(branch="oracle", seed=7101))
        with self.assertRaises(ValueError):
            feasibility.validate_cli_contract(Namespace(branch="other", seed=7100))

    def test_branch_runner_executes_exactly_one_epoch_without_train_or_save(self) -> None:
        source = inspect.getsource(feasibility.run_branch_epoch)
        self.assertEqual(source.count("agent.train_epoch()"), 1)
        self.assertEqual(source.count("agent.update_epoch()"), 1)
        self.assertNotIn("agent.train()", source)
        self.assertNotIn("agent.save(", source)

    def test_capacity_failure_classification_is_stage_and_signature_bound(self) -> None:
        self.assertEqual(
            feasibility.classify_failure("B3", "CUDA out of memory"),
            ["FORMAL_6144_BACKWARD_CAPACITY_BLOCKED", "HARDWARE_OR_PROTOCOL_REVISION_REQUIRED"],
        )
        self.assertEqual(
            feasibility.classify_failure("B2", "Failed to create articulation"),
            ["FORMAL_TRAINING_PIPELINE_BLOCKED"],
        )
        self.assertEqual(
            feasibility.classify_failure(None, "CUDA out of memory before AppLauncher"),
            ["FORMAL_TRAINING_PIPELINE_BLOCKED"],
        )

    def test_resource_signatures_cover_common_cuda_host_and_physx_forms(self) -> None:
        for message in (
            "CUBLAS_STATUS_ALLOC_FAILED",
            "cudaErrorMemoryAllocation",
            "std::bad_alloc",
            "PhysX GPU heap failed to allocate",
        ):
            self.assertEqual(
                feasibility.classify_failure("B3", message),
                ["FORMAL_6144_BACKWARD_CAPACITY_BLOCKED", "HARDWARE_OR_PROTOCOL_REVISION_REQUIRED"],
            )

    def test_oracle_requires_passed_history_from_same_commit(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            write_governed_history(root, passing_result("history_only", 12))
            feasibility.validate_oracle_precondition(root, "abc", current_pid=34)
            with self.assertRaises(RuntimeError):
                feasibility.validate_oracle_precondition(root, "different", current_pid=34)

    def test_oracle_is_blocked_after_history_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            record = passing_result("history_only", 12)
            record["status"] = "BLOCKED"
            write_governed_history(root, record)
            with self.assertRaises(RuntimeError):
                feasibility.validate_oracle_precondition(root, "abc", current_pid=34)

    def test_oracle_rejects_stale_or_underspecified_history_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            record = passing_result("history_only", 12)
            record["feasibility_id"] = "stale"
            write_governed_history(root, record)
            with self.assertRaises(RuntimeError):
                feasibility.validate_oracle_precondition(root, "abc", current_pid=34)

    def test_oracle_rejects_missing_or_modified_history_console(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            write_governed_history(root, passing_result("history_only", 12))
            console = root / "history_only" / "console.log"
            console.write_text("tampered\n", encoding="utf-8")
            with self.assertRaises(RuntimeError):
                feasibility.validate_oracle_precondition(root, "abc", current_pid=34)

    def test_oracle_rejects_invalid_console_capture_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            record = passing_result("history_only", 12)
            write_governed_history(root, record)
            history_path = root / "history_only" / "result.json"
            history = json.loads(history_path.read_text(encoding="utf-8"))
            history["console_capture"]["owner"] = "manual"  # type: ignore[index]
            payload = json.dumps(history, indent=2, sort_keys=True) + "\n"
            history_path.write_text(payload, encoding="utf-8")
            checksums = json.loads((root / "artifact_checksums.json").read_text(encoding="utf-8"))
            checksums["artifacts"]["history_only/result.json"] = hashlib.sha256(history_path.read_bytes()).hexdigest()
            (root / "artifact_checksums.json").write_text(json.dumps(checksums), encoding="utf-8")
            with self.assertRaises(RuntimeError):
                feasibility.validate_oracle_precondition(root, "abc", current_pid=34)

    def test_oracle_reparses_sentinels_instead_of_trusting_result_booleans(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            write_governed_history(root, passing_result("history_only", 12))
            console_path = root / "history_only" / "console.log"
            console_path.write_text("clean exit without capture sentinels\n", encoding="utf-8")
            history_path = root / "history_only" / "result.json"
            history = json.loads(history_path.read_text(encoding="utf-8"))
            history["console_sha256"] = hashlib.sha256(console_path.read_bytes()).hexdigest()
            history_path.write_text(json.dumps(history, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            checksums_path = root / "artifact_checksums.json"
            checksums = json.loads(checksums_path.read_text(encoding="utf-8"))
            checksums["artifacts"]["history_only/console.log"] = history["console_sha256"]
            checksums["artifacts"]["history_only/result.json"] = hashlib.sha256(history_path.read_bytes()).hexdigest()
            checksums_path.write_text(json.dumps(checksums), encoding="utf-8")
            with self.assertRaises(RuntimeError):
                feasibility.validate_oracle_precondition(root, "abc", current_pid=34)

    def test_timeline_uses_active_boundary_instead_of_first_incomplete_boundary(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            timeline = feasibility.BackwardResourceTimeline(
                Path(temp_dir) / "timeline.json",
                "history_only",
                snapshot_provider=lambda: {"resource": "snapshot"},
            )
            timeline.begin("B3")
            timeline.fail(RuntimeError("CUDA out of memory"))
            self.assertEqual(timeline.stages[-1]["stage_id"], "B3")

    def test_completed_stage_cannot_be_reactivated_by_later_minibatches(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            timeline = feasibility.BackwardResourceTimeline(
                Path(temp_dir) / "timeline.json",
                "history_only",
                snapshot_provider=lambda: {"resource": "snapshot"},
            )
            timeline.begin("B4")
            timeline.record("B4")
            timeline.begin("B4")
            self.assertIsNone(timeline.active_stage_id)
            self.assertEqual(len(timeline.stages), 1)

    def test_console_finalization_can_block_native_failure_text(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            branch_dir = root / "history_only"
            branch_dir.mkdir(parents=True)
            record = passing_result("history_only", 12)
            record["status"] = "PENDING_CONSOLE_FINALIZATION"
            record["checks"] = {"runtime": True}
            (branch_dir / "result.json").write_text(json.dumps(record), encoding="utf-8")
            console = branch_dir / "console.log"
            token = "test-token"
            digest = hashlib.sha256(token.encode("utf-8")).hexdigest()
            console.write_text(
                f"R7_PHASE42B_PARENT_CAPTURE_START token_sha256={digest}\n"
                "PhysX GPU heap failed to allocate\n"
                f"R7_PHASE42B_PARENT_CAPTURE_COMPLETE token_sha256={digest} exit_code=0\n",
                encoding="utf-8",
            )
            finalized = feasibility.finalize_console_evidence(root, "history_only", console, exit_code=0, capture_token=token)
            self.assertEqual(finalized["status"], "BLOCKED")
            self.assertTrue(finalized["failure_signatures"]["resource_allocation"])

    def test_default_output_is_derived_from_formal_artifact_root(self) -> None:
        self.assertEqual(
            feasibility.DEFAULT_OUTPUT_ROOT,
            feasibility.formal.DEFAULT_OUTPUT_ROOT / "backward_feasibility",
        )

    def test_manual_console_finalizer_cli_is_not_exposed(self) -> None:
        self.assertNotIn("--finalize-console", inspect.getsource(feasibility.main))

    def test_finalizer_rejects_decoy_or_noncanonical_console(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            branch_dir = root / "history_only"
            branch_dir.mkdir(parents=True)
            (branch_dir / "result.json").write_text(json.dumps(passing_result("history_only", 12)), encoding="utf-8")
            decoy = root / "decoy.log"
            decoy.write_text("clean exit\n", encoding="utf-8")
            with self.assertRaises(RuntimeError):
                feasibility.finalize_console_evidence(root, "history_only", decoy, exit_code=0, capture_token="x")

    def test_parent_launcher_owns_stdout_stderr_and_finalization(self) -> None:
        source = inspect.getsource(feasibility.launch_branch_process)
        self.assertIn("stdout=console_stream", source)
        self.assertIn("stderr=subprocess.STDOUT", source)
        self.assertIn("finalize_console_evidence", source)
        self.assertNotIn('"--capture-token"', source)
        self.assertIn("R7_PHASE42B_CAPTURE_PARENT_PID", source)

    def test_worker_cli_does_not_accept_capture_secret(self) -> None:
        self.assertNotIn("capture-token", inspect.getsource(feasibility.build_parser))

    def test_controlled_parent_launcher_is_allowed(self) -> None:
        audit = audit_records(valid_launch_records())
        self.assertTrue(audit["pass"])
        self.assertEqual(audit["owned_roles"]["controlled_parent"], [100])

    def test_controlled_parent_may_use_default_or_relative_arguments(self) -> None:
        parent = valid_launch_records()[0]
        self.assertNotIn("--repo-root", parent["command_line"])
        self.assertNotIn("--output-root", parent["command_line"])
        self.assertTrue(audit_records(valid_launch_records())["parent_identity_pass"])

    def test_isaaclab_cli_child_is_allowed(self) -> None:
        audit = audit_records(valid_launch_records())
        self.assertEqual(audit["owned_roles"]["cli_shim"], [101])

    def test_kit_controlled_child_is_allowed(self) -> None:
        audit = audit_records(valid_launch_records())
        self.assertEqual(audit["owned_roles"]["kit_launcher"], [102])

    def test_virtualenv_python_isaac_cli_launcher_is_allowed(self) -> None:
        records = valid_launch_records()
        records[2] = launch_record(102, 101, "python.exe", "venv_cli", 1000.2, [101, 100, 50])
        audit = audit_records(records)
        self.assertTrue(audit["pass"])
        self.assertEqual(audit["owned_roles"]["kit_launcher"], [102])

    def test_python_worker_is_allowed(self) -> None:
        audit = audit_records(valid_launch_records())
        self.assertEqual(audit["owned_roles"]["current_worker"], [103])

    def test_same_launch_tree_grandchild_is_allowed(self) -> None:
        records = valid_launch_records()
        records.append(launch_record(104, 103, "python.exe", "grandchild", 1000.4, [103, 102, 101, 100, 50]))
        audit = audit_records(records)
        self.assertTrue(audit["pass"])
        self.assertNotIn("--branch", records[-1]["command_line"])
        self.assertEqual(audit["owned_roles"]["controlled_descendant"], [104])

    def test_unrelated_old_kit_is_rejected(self) -> None:
        records = valid_launch_records()
        records.append(launch_record(200, 9, "kit.exe", "kit", 900.0, [9]))
        audit = audit_records(records)
        self.assertFalse(audit["pass"])
        self.assertEqual([item["pid"] for item in audit["residuals"]], [200])

    def test_unrelated_project_python_residual_is_rejected(self) -> None:
        records = valid_launch_records()
        records.append(launch_record(201, 9, "python.exe", "grandchild", 1000.4, [9]))
        audit = audit_records(records)
        self.assertFalse(audit["pass"])
        self.assertEqual([item["pid"] for item in audit["residuals"]], [201])

    def test_non_project_python_is_ignored(self) -> None:
        records = valid_launch_records()
        record = launch_record(202, 9, "python.exe", "grandchild", 1000.4, [9], project_related=False)
        record["command_line"] = "python unrelated_tool.py"
        records.append(record)
        audit = audit_records(records)
        self.assertTrue(audit["pass"])
        self.assertIn(202, audit["ignored_pids"])

    def test_uninspectable_python_kit_or_isaac_candidate_fails_closed(self) -> None:
        records = valid_launch_records()
        records.append(
            {
                "inspection_available": False,
                "inspection_error_kind": "AccessDenied",
                "pid": 203,
                "name": "kit.exe",
            }
        )
        audit = audit_records(records)
        self.assertFalse(audit["pass"])
        self.assertEqual(audit["uncertain_process_pids"], [203])

    def test_cwd_access_denied_propagates_to_fail_closed_census(self) -> None:
        class FakeProcess:
            pid = 205

            def oneshot(self):
                return nullcontext()

            def cmdline(self):
                return ["python", "scripts\\project_worker.py"]

            def exe(self):
                return r"D:\Python\python.exe"

            def cwd(self):
                raise psutil.AccessDenied(pid=self.pid)

            def parents(self):
                return []

            def ppid(self):
                return 1

            def name(self):
                return "python.exe"

            def create_time(self):
                return 1000.0

        with self.assertRaises(psutil.AccessDenied):
            feasibility._live_process_record(
                FakeProcess(),
                Path(r"D:\Research\LEAP\LEAP_Hand_Isaac_Lab"),
                Path(r"C:\evidence\fresh_r1"),
            )

    def test_process_that_vanished_during_census_does_not_block(self) -> None:
        records = valid_launch_records()
        records.append(
            {
                "inspection_available": False,
                "inspection_error_kind": "NoSuchProcess",
                "pid": 204,
                "name": "python.exe",
            }
        )
        audit = audit_records(records)
        self.assertTrue(audit["pass"])
        self.assertEqual(audit["vanished_process_pids"], [204])

    def test_pid_reuse_does_not_grant_parent_ownership(self) -> None:
        records = valid_launch_records()
        records[0]["create_time"] = 800.0
        audit = audit_records(records)
        self.assertFalse(audit["pass"])
        self.assertFalse(audit["parent_identity_pass"])

    def test_creation_time_mismatch_does_not_allow_controlled_kit(self) -> None:
        records = valid_launch_records()
        records[2]["create_time"] = 900.0
        audit = audit_records(records)
        self.assertFalse(audit["pass"])
        self.assertEqual([item["pid"] for item in audit["residuals"]], [102])

    def test_wrong_working_directory_does_not_grant_parent_or_worker_ownership(self) -> None:
        records = valid_launch_records()
        records[0]["working_directory"] = r"C:\unrelated"
        records[3]["working_directory"] = r"C:\unrelated"
        audit = audit_records(records)
        self.assertFalse(audit["pass"])
        self.assertFalse(audit["parent_identity_pass"])
        self.assertFalse(audit["worker_identity_pass"])

    def test_fake_kit_command_on_same_spine_is_rejected(self) -> None:
        records = valid_launch_records()
        records[2]["command_line"] = str(records[2]["command_line"]).replace(
            'from isaaclab.cli import cli; cli()',
            "print('not the Isaac Lab CLI')",
        )
        audit = audit_records(records)
        self.assertFalse(audit["pass"])
        self.assertEqual([item["pid"] for item in audit["residuals"]], [102])

    def test_complete_parent_cli_kit_worker_chain_is_required(self) -> None:
        without_cli = [record for record in valid_launch_records() if record["pid"] != 101]
        without_kit = [record for record in valid_launch_records() if record["pid"] != 102]
        self.assertFalse(audit_records(without_cli)["role_completeness_pass"])
        self.assertFalse(audit_records(without_kit)["role_completeness_pass"])

    def test_formal_5000_epoch_authorization_path_remains_unreachable(self) -> None:
        self.assertNotIn("authorization", inspect.getsource(feasibility.build_launch_parser))
        self.assertNotIn("agent.train()", inspect.getsource(feasibility.run_branch_epoch))
        self.assertEqual(inspect.getsource(feasibility.run_branch_epoch).count("agent.train_epoch()"), 1)

    def test_history_requires_a_completely_fresh_output_root(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "fresh_run"
            feasibility.validate_output_root_freshness(root, "history_only")
            root.mkdir()
            (root / "oracle").mkdir()
            (root / "oracle" / "result.json").write_text("{}", encoding="utf-8")
            with self.assertRaises(RuntimeError):
                feasibility.validate_output_root_freshness(root, "history_only")

    def test_oracle_requires_existing_root_and_empty_oracle_slot(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "run"
            with self.assertRaises(RuntimeError):
                feasibility.validate_output_root_freshness(root, "oracle")
            root.mkdir()
            feasibility.validate_output_root_freshness(root, "oracle")
            (root / "oracle").mkdir()
            (root / "oracle" / "console.log").write_text("old", encoding="utf-8")
            with self.assertRaises(RuntimeError):
                feasibility.validate_output_root_freshness(root, "oracle")

    def test_fairness_pass_requires_two_real_updates_and_oracle_context_gradient(self) -> None:
        results = {
            "history_only": passing_result("history_only", 12),
            "oracle": passing_result("oracle", 34),
        }
        report = feasibility.build_fairness_report(results)
        self.assertTrue(report["overall_pass"])
        results["oracle"]["training_counters"]["context_column_nonzero_gradient_calls"] = 0  # type: ignore[index]
        self.assertFalse(feasibility.build_fairness_report(results)["overall_pass"])

    def test_fairness_rejects_cross_run_branch_mix(self) -> None:
        results = {
            "history_only": passing_result("history_only", 12),
            "oracle": passing_result("oracle", 34),
        }
        results["oracle"]["run_id"] = "b" * 32
        report = feasibility.build_fairness_report(results)
        self.assertFalse(report["overall_pass"])
        self.assertFalse(report["checks"]["same_run_id"])

    def test_fairness_requires_lowercase_hex_run_id_and_records_it(self) -> None:
        results = {
            "history_only": passing_result("history_only", 12),
            "oracle": passing_result("oracle", 34),
        }
        results["history_only"]["run_id"] = results["oracle"]["run_id"] = "z" * 32
        report = feasibility.build_fairness_report(results)
        self.assertFalse(report["checks"]["same_run_id"])
        self.assertEqual(report["run_id"], "z" * 32)

    def test_finalizer_rejects_result_preflight_session_mixing(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            branch_dir = root / "history_only"
            branch_dir.mkdir(parents=True)
            record = passing_result("history_only", 12)
            record["status"] = "PENDING_CONSOLE_FINALIZATION"
            (branch_dir / "result.json").write_text(json.dumps(record), encoding="utf-8")
            token = "capture-token"
            token_digest = hashlib.sha256(token.encode("utf-8")).hexdigest()
            (branch_dir / "preflight_system_state.json").write_text(
                json.dumps(
                    {
                        "run_id": "b" * 32,
                        "console_capture_admission": {"token_sha256": token_digest},
                        "clean_process_gate": {"current_worker_pid": 12},
                    }
                ),
                encoding="utf-8",
            )
            console = branch_dir / "console.log"
            console.write_text(
                f"R7_PHASE42B_PARENT_CAPTURE_START token_sha256={token_digest}\n"
                f"R7_PHASE42B_PARENT_CAPTURE_COMPLETE token_sha256={token_digest} exit_code=0\n",
                encoding="utf-8",
            )
            finalized = feasibility.finalize_console_evidence(root, "history_only", console, 0, token)
            self.assertEqual(finalized["status"], "BLOCKED")
            self.assertFalse(finalized["checks"]["result_preflight_lineage"])

    def test_later_minibatch_oom_uses_backward_capacity_stage_hint(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            timeline = feasibility.BackwardResourceTimeline(
                Path(temp_dir) / "timeline.json",
                "history_only",
                snapshot_provider=lambda: {"resource": "snapshot"},
            )
            for stage_id in feasibility.BACKWARD_STAGES:
                timeline.record(stage_id)
            timeline.set_failure_hint("B3")
            timeline.fail(RuntimeError("CUDA out of memory in later minibatch"))
            self.assertEqual(timeline.failed_stage_id, "B3")
            self.assertEqual(
                feasibility.classify_failure(timeline.failed_stage_id, "CUDA out of memory"),
                ["FORMAL_6144_BACKWARD_CAPACITY_BLOCKED", "HARDWARE_OR_PROTOCOL_REVISION_REQUIRED"],
            )

    def test_pass_decision_requires_both_branches(self) -> None:
        results = {
            "history_only": passing_result("history_only", 12),
            "oracle": passing_result("oracle", 34),
        }
        self.assertEqual(
            feasibility.final_decision(results),
            ["FORMAL_6144_BACKWARD_PASS", "READY_FOR_FORMAL_O0_TRAINING"],
        )
        del results["oracle"]
        self.assertEqual(feasibility.final_decision(results), ["FORMAL_TRAINING_PIPELINE_BLOCKED"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
