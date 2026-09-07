"""R7-O0 Phase 4.2 正式训练入口的非 Isaac 合约测试。"""

from __future__ import annotations

import importlib.util
import inspect
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/r7_o0_formal_train.py"
SPEC = importlib.util.spec_from_file_location("r7_o0_formal_train", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
formal = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(formal)


class FormalTrainContractTest(unittest.TestCase):
    def test_formal_environment_total_is_exact(self) -> None:
        self.assertEqual(formal.FORMAL_NUM_ENVS, 6144)

    def test_factor_assignment_is_exactly_balanced(self) -> None:
        codes = formal.build_balanced_factor_codes()
        self.assertEqual(len(codes), 6144)
        self.assertEqual({code: codes.count(code) for code in set(codes)}, {"000": 2048, "010": 2048, "100": 2048})

    def test_factor_assignment_uses_only_o0_support(self) -> None:
        self.assertEqual(set(formal.build_balanced_factor_codes()), {"000", "010", "100"})

    def test_only_legal_training_seeds(self) -> None:
        for seed in formal.TRAINING_SEEDS:
            formal.validate_cli_contract(Namespace(mode="capacity-preflight", branch="oracle", seed=seed))
        with self.assertRaises(ValueError):
            formal.validate_cli_contract(Namespace(mode="capacity-preflight", branch="oracle", seed=7099))

    def test_only_legal_branches(self) -> None:
        with self.assertRaises(ValueError):
            formal.validate_cli_contract(Namespace(mode="capacity-preflight", branch="other", seed=7100))

    def test_policy_input_is_99d(self) -> None:
        self.assertEqual(formal.LEGAL_OBSERVATION_DIM + formal.CONTEXT_DIM, formal.POLICY_INPUT_DIM)
        self.assertEqual(formal.POLICY_INPUT_DIM, 99)

    def test_history_context_is_exact_zero(self) -> None:
        self.assertEqual(formal.policy_context("history_only", "010", 1.0, 0.45, 1.0), (0.0, 0.0, 0.0))

    def test_three_oracle_contexts_are_frozen(self) -> None:
        actual = {
            code: formal.policy_context("oracle", code, *values)
            for code, values in {
                "000": (1.0, 0.80, 1.0),
                "010": (1.0, 0.45, 1.0),
                "100": (1.35, 0.80, 1.0),
            }.items()
        }
        self.assertEqual(actual, formal.EXPECTED_ORACLE_CONTEXTS)

    def test_context_lineage_is_actual_readback_only(self) -> None:
        self.assertEqual(set(formal.CONTEXT_SOURCE_FIELDS), {"actual_object_mass", "actual_object_static_friction", "actual_effort_limit"})
        self.assertFalse(set(formal.CONTEXT_SOURCE_FIELDS) & set(formal.PROHIBITED_POLICY_FIELDS))

    def test_policy_context_normalization_does_not_derive_reference_from_factor_codes(self) -> None:
        source = inspect.getsource(formal.make_formal_adapter_class)
        self.assertIn("mass / NOMINAL_OBJECT_MASS_KG", source)
        self.assertIn("effort.mean(dim=1) / NOMINAL_EFFORT_LIMIT", source)
        self.assertNotIn('mass[code_tensor["000"]].mean()', source)

    def test_checkpoint_selection_is_epoch_5000_only(self) -> None:
        self.assertEqual(formal.MAX_EPOCHS, 5000)
        self.assertEqual(formal.EVALUATION_CHECKPOINT_RULE, "epoch_5000_final_only")

    def test_fresh_optimizer_contract(self) -> None:
        formal.require_fresh_optimizer({})
        with self.assertRaises(RuntimeError):
            formal.require_fresh_optimizer({"state": 1})

    def test_dirty_minibatch_override_is_rejected(self) -> None:
        cfg = formal.expected_frozen_config_fields()
        cfg["params.config.minibatch_size"] = 4096
        with self.assertRaises(RuntimeError):
            formal.validate_flat_config(cfg)

    def test_capacity_path_has_no_train_epoch_call(self) -> None:
        self.assertNotIn("train_epoch(", inspect.getsource(formal.run_capacity_preflight))

    def test_capacity_path_has_no_backward_call(self) -> None:
        self.assertNotIn(".backward(", inspect.getsource(formal.run_capacity_preflight))

    def test_capacity_path_has_no_optimizer_step_call(self) -> None:
        self.assertNotIn("optimizer.step(", inspect.getsource(formal.run_capacity_preflight))

    def test_train_mode_is_blocked_without_authorization(self) -> None:
        with self.assertRaisesRegex(PermissionError, "FORMAL_TRAINING_NOT_AUTHORIZED"):
            formal.validate_training_authorization(None, "oracle", 7100, "abc")

    def test_artifact_schema_contains_all_future_runs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            paths = formal.initialize_artifact_schema(Path(temp_dir))
            expected = {f"{branch}/seed_{seed}" for branch in formal.BRANCHES for seed in formal.TRAINING_SEEDS}
            self.assertTrue(expected.issubset(set(paths)))
            self.assertTrue((Path(temp_dir) / "capacity_preflight").is_dir())
            self.assertTrue((Path(temp_dir) / "mixed_factor_integrity").is_dir())

    def test_mixed_factor_count_audit(self) -> None:
        report = formal.audit_factor_counts(formal.build_balanced_factor_codes())
        self.assertTrue(report["pass"])
        self.assertEqual(report["counts"], formal.EXPECTED_FACTOR_COUNTS)

    def test_mixed_material_identity_is_an_explicit_runtime_gate(self) -> None:
        source = inspect.getsource(formal.audit_mixed_snapshot)
        self.assertIn("material_prim_identity_pass", source)
        self.assertIn("material_state_separation", source)

    def test_mixed_integrity_gate_cannot_be_bypassed(self) -> None:
        self.assertEqual(
            formal.final_decision(capacity_pass=True, mixed_integrity_pass=False),
            ["FORMAL_6144_CAPACITY_PASS", "MIXED_FACTOR_INTEGRITY_BLOCKED", "SCIENTIFIC_PROTOCOL_DECISION_REQUIRED"],
        )

    def test_capacity_failure_requires_hardware_decision(self) -> None:
        self.assertEqual(
            formal.final_decision(capacity_pass=False, mixed_integrity_pass=False),
            ["FORMAL_6144_CAPACITY_BLOCKED", "HARDWARE_DECISION_REQUIRED"],
        )

    def test_capacity_stages_are_exact(self) -> None:
        self.assertEqual(tuple(formal.FORMAL_STAGES), tuple(range(8)))
        self.assertEqual(formal.FORMAL_STAGES[7], "after_checkpoint_load_and_rollout_allocation")

    def test_formal_run_artifact_fields_are_complete(self) -> None:
        required = {
            "protocol_id", "branch", "seed", "git_commit", "checkpoint_sha256", "realized_config_sha256",
            "factor_counts", "actual_context_distribution", "optimizer_fresh_state", "epoch_count", "frame_count",
            "final_checkpoint_sha256", "resource_telemetry", "completion_status",
        }
        self.assertTrue(required.issubset(set(formal.FORMAL_RUN_REQUIRED_FIELDS)))


if __name__ == "__main__":
    unittest.main(verbosity=2)
