"""Non-Isaac contract tests for the R7-O0 Phase 4.1 smoke driver."""

from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/r7_o0_smoke_training.py"
SPEC = importlib.util.spec_from_file_location("r7_o0_smoke_training", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
smoke = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(smoke)


class SmokeContractTest(unittest.TestCase):
    def test_frozen_constants(self) -> None:
        self.assertEqual(smoke.NUM_ENVS * smoke.HORIZON_LENGTH, smoke.MINIBATCH_SIZE)
        self.assertEqual((smoke.OBSERVATION_DIM, smoke.CONTEXT_DIM, smoke.POLICY_INPUT_DIM), (96, 3, 99))
        self.assertEqual(smoke.ACTION_DIM, 16)
        self.assertEqual(smoke.SEED, 7100)
        self.assertEqual(smoke.DEFAULT_OUTPUT_ROOT.name, "r7_o0_smoke_phase41c")
        self.assertEqual(smoke.SMOKE_ID, "R7_O0_PHASE41C_CLEAN_SMOKE_V1")

    def test_allocation_stage_contract(self) -> None:
        self.assertEqual(tuple(smoke.ALLOCATION_STAGES), tuple(range(13)))
        self.assertEqual(smoke.ALLOCATION_STAGES[0], "before_app_launcher")
        self.assertEqual(smoke.ALLOCATION_STAGES[12], "after_first_optimizer_step")

    def test_incremental_timeline_is_atomic_and_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "allocation_timeline.json"
            recorder = smoke.AllocationTimeline(
                path,
                branch="history_only",
                snapshot_provider=lambda: {"gpu": {"available": False}},
            )
            recorder.record(0)
            recorder.record(0)
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(len(payload["stages"]), 1)
            self.assertEqual(payload["stages"][0]["stage_id"], 0)
            self.assertEqual(payload["stages"][0]["status"], "COMPLETE")

    def test_failure_classification_uses_first_incomplete_boundary(self) -> None:
        self.assertEqual(
            smoke.classify_runtime_failure({0, 1}, RuntimeError("Failed to create articulation")),
            "TRANSIENT_ENVIRONMENT_LIFECYCLE_FAILURE_REPRODUCED",
        )
        self.assertEqual(
            smoke.classify_runtime_failure({0, 1, 2, 3, 4, 5, 6, 7}, RuntimeError("CUDA out of memory")),
            "PPO_BUFFER_ALLOCATION_LIMIT_IDENTIFIED",
        )
        self.assertEqual(
            smoke.classify_runtime_failure(set(range(9)), RuntimeError("gradient hook missing")),
            "PPO_PIPELINE_CONTRACT_FAILURE",
        )
        self.assertEqual(smoke.classify_runtime_failure({0, 1, 2}, RuntimeError("unknown")), "ROOT_CAUSE_UNRESOLVED")

    def test_clean_gate_excludes_current_launcher_ancestor(self) -> None:
        processes = [
            {"pid": 10, "project_related": True, "is_current_worker": True, "is_current_launch_ancestor": False},
            {"pid": 9, "project_related": True, "is_current_worker": False, "is_current_launch_ancestor": True},
            {"pid": 8, "project_related": True, "is_current_worker": False, "is_current_launch_ancestor": False},
        ]
        self.assertEqual([item["pid"] for item in smoke.confirmed_project_residuals(processes)], [8])

    def test_cli_bounds(self) -> None:
        for branch in smoke.BRANCHES:
            smoke.validate_cli_contract(Namespace(branch=branch, max_iterations=1))
            smoke.validate_cli_contract(Namespace(branch=branch, max_iterations=100))
        for value in (0, 101):
            with self.assertRaises(ValueError):
                smoke.validate_cli_contract(Namespace(branch="history_only", max_iterations=value))

    def test_normalized_contexts(self) -> None:
        self.assertEqual(smoke.normalized_context(1.0, 0.80, 1.0), (-1.0, 1.0, 1.0))
        self.assertEqual(smoke.normalized_context(1.0, 0.45, 1.0), (-1.0, -1.0, 1.0))
        self.assertEqual(smoke.normalized_context(1.35, 0.80, 1.0), (1.0, 1.0, 1.0))
        with self.assertRaises(ValueError):
            smoke.normalized_context(2.0, 0.80, 1.0)

    def test_frozen_config_validation(self) -> None:
        cfg = {
            "params": {
                "algo": {"name": "a2c_continuous"},
                "model": {"name": "continuous_a2c_logstd"},
                "network": {
                    "name": "actor_critic",
                    "separate": False,
                    "space": {"continuous": {"fixed_sigma": True}},
                    "mlp": {"units": [512, 256, 128], "activation": "elu"},
                    "rnn": {
                        "name": "gru",
                        "units": 256,
                        "layers": 1,
                        "before_mlp": True,
                        "concat_input": True,
                        "layer_norm": True,
                    },
                },
                "env": {"clip_observations": 5.0, "clip_actions": 1.0},
                "config": {
                    "env_name": "rlgpu",
                    "ppo": True,
                    "mixed_precision": False,
                    "normalize_input": True,
                    "normalize_value": True,
                    "normalize_advantage": True,
                    "gamma": 0.99,
                    "tau": 0.95,
                    "learning_rate": 5e-4,
                    "lr_schedule": "adaptive",
                    "grad_norm": 1.0,
                    "horizon_length": 32,
                    "minibatch_size": 32768,
                    "mini_epochs": 5,
                    "seq_length": 4,
                },
            }
        }
        self.assertEqual(smoke.validate_frozen_agent_config(cfg)["status"], "PASS")
        cfg["params"]["config"]["learning_rate"] = "5e-4"
        coercions = smoke.normalize_committed_yaml_scalars(cfg)
        self.assertEqual(coercions[0]["runtime_value"], 5e-4)
        self.assertFalse(coercions[0]["semantic_change"])
        self.assertEqual(smoke.validate_frozen_agent_config(cfg)["status"], "PASS")
        cfg["params"]["config"]["minibatch_size"] = 4096
        with self.assertRaises(RuntimeError):
            smoke.validate_frozen_agent_config(cfg)

    def test_aggregate_fairness_contract(self) -> None:
        common = {
            "status": "PASS",
            "seed": 7100,
            "condition": "D0",
            "num_envs": 1024,
            "iterations_requested": 1,
            "physics_dt": 1.0 / 120.0,
            "decimation": 4,
            "input_dimension": 99,
            "legal_observation_dimension": 96,
            "context_dimension": 3,
            "action_dimension": 16,
            "trainable_parameter_count": 572705,
            "state_tensor_element_count": 572907,
            "model_hash_before": "same",
            "process_id": 1001,
            "checkpoint": {"expanded_checkpoint": {"sha256": smoke.EXPECTED_EXPANDED_SHA256}},
            "config_source": {"committed_sha256": "same", "working_tree_consumed_as_config": False},
        }
        history = dict(common)
        history["context_injection"] = {"first_raw_context": [0.0, 0.0, 0.0]}
        oracle = dict(common)
        oracle["process_id"] = 1002
        oracle["context_injection"] = {"first_raw_context": [-1.0, 1.0, 1.0]}
        report = smoke.fairness_from_results({"history_only": history, "oracle": oracle})
        self.assertTrue(report["overall_pass"])
        self.assertEqual(report["status"], "PASS")

    def test_aggregate_fairness_accepts_float32_context_readback(self) -> None:
        common = {
            "status": "PASS", "seed": 7100, "condition": "D0", "num_envs": 1024,
            "iterations_requested": 1, "physics_dt": 1.0 / 120.0, "decimation": 4,
            "input_dimension": 99, "legal_observation_dimension": 96, "context_dimension": 3,
            "action_dimension": 16, "trainable_parameter_count": 572705,
            "state_tensor_element_count": 572907, "model_hash_before": "same", "process_id": 1001,
            "checkpoint": {"expanded_checkpoint": {"sha256": smoke.EXPECTED_EXPANDED_SHA256}},
            "config_source": {"committed_sha256": "same", "working_tree_consumed_as_config": False},
        }
        history = dict(common)
        history["context_injection"] = {"first_raw_context": [0.0, 0.0, 0.0]}
        oracle = dict(common)
        oracle["process_id"] = 1002
        oracle["context_injection"] = {"first_raw_context": [-1.0, 1.0, 0.9999998807907104]}
        report = smoke.fairness_from_results({"history_only": history, "oracle": oracle})
        self.assertTrue(report["overall_pass"])
        self.assertTrue(report["checks"]["oracle_context_d0_actual"])

    def test_import_is_inert(self) -> None:
        self.assertNotIn("torch", smoke.__dict__)
        self.assertNotIn("simulation_app", smoke.__dict__)

    def test_artifact_hashes_cover_json_and_binary_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "branch").mkdir()
            (root / "result.json").write_text("{}", encoding="utf-8")
            (root / "branch" / "events.bin").write_bytes(b"event")
            checksum_path = root / "artifact_checksums.json"
            checksum_path.write_text("self", encoding="utf-8")
            hashes = smoke.collect_artifact_hashes(root, checksum_path)
            self.assertEqual(set(hashes), {"result.json", "branch/events.bin"})


if __name__ == "__main__":
    unittest.main(verbosity=2)
