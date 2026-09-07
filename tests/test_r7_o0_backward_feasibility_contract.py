"""R7-O0 Phase 4.2b backward feasibility gate 的非 Isaac 合约测试。"""

from __future__ import annotations

import importlib.util
import hashlib
import inspect
import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/r7_o0_backward_feasibility.py"
SPEC = importlib.util.spec_from_file_location("r7_o0_backward_feasibility", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
feasibility = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(feasibility)


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

    def test_clean_process_gate_requires_exact_controlled_parent_pid(self) -> None:
        records = [
            {
                "pid": 41,
                "command_line": f"python {feasibility.SCRIPT_PATH if hasattr(feasibility, 'SCRIPT_PATH') else feasibility.__file__} --launch-branch",
                "inspection_available": True,
                "project_related": True,
                "is_current_worker": False,
                "is_current_launch_ancestor": False,
            }
        ]
        accepted = feasibility.audit_controlled_parent(records, expected_parent_pid=41, ancestor_pids={41})
        self.assertTrue(accepted["pass"])
        self.assertFalse(feasibility.audit_controlled_parent(records, expected_parent_pid=99, ancestor_pids={41})["pass"])

    def test_fairness_pass_requires_two_real_updates_and_oracle_context_gradient(self) -> None:
        results = {
            "history_only": passing_result("history_only", 12),
            "oracle": passing_result("oracle", 34),
        }
        report = feasibility.build_fairness_report(results)
        self.assertTrue(report["overall_pass"])
        results["oracle"]["training_counters"]["context_column_nonzero_gradient_calls"] = 0  # type: ignore[index]
        self.assertFalse(feasibility.build_fairness_report(results)["overall_pass"])

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
