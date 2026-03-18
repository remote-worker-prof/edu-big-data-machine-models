import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]
MODULE_PATH = BASE_DIR / "lab_utils.py"
SPEC = importlib.util.spec_from_file_location("lab03_utils", MODULE_PATH)
lab = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(lab)


class Lab03UtilsTestCase(unittest.TestCase):
    def make_generalization_audit(self, dataset_name: str, model_name: str, rows):
        records = []
        for feature_set, train_f1, valid_f1 in rows:
            records.extend(
                [
                    {
                        "dataset": dataset_name,
                        "feature_set": feature_set,
                        "model": model_name,
                        "split": "train",
                        "accuracy": train_f1,
                        "f1": train_f1,
                        "roc_auc": train_f1,
                        "fit_time_sec": 0.1,
                    },
                    {
                        "dataset": dataset_name,
                        "feature_set": feature_set,
                        "model": model_name,
                        "split": "validation",
                        "accuracy": valid_f1,
                        "f1": valid_f1,
                        "roc_auc": valid_f1,
                        "fit_time_sec": 0.1,
                    },
                ]
            )
        return pd.DataFrame(records)

    def make_model_feature_set_decisions(self):
        return pd.DataFrame(
            [
                {
                    "dataset": "finance",
                    "model": "LogisticRegression",
                    "selected_feature_set": "set_A_wrapper",
                    "train_f1": 0.61,
                    "validation_f1": 0.60,
                    "f1_gap": 0.01,
                    "abs_f1_gap": 0.01,
                    "tie_break_reason": "best validation_f1",
                },
                {
                    "dataset": "finance",
                    "model": "RandomForest",
                    "selected_feature_set": "set_B_tree",
                    "train_f1": 0.70,
                    "validation_f1": 0.58,
                    "f1_gap": 0.12,
                    "abs_f1_gap": 0.12,
                    "tie_break_reason": "best validation_f1",
                },
                {
                    "dataset": "medical",
                    "model": "LogisticRegression",
                    "selected_feature_set": "full",
                    "train_f1": 0.55,
                    "validation_f1": 0.56,
                    "f1_gap": -0.01,
                    "abs_f1_gap": 0.01,
                    "tie_break_reason": "best validation_f1",
                },
                {
                    "dataset": "medical",
                    "model": "RandomForest",
                    "selected_feature_set": "set_C_hybrid",
                    "train_f1": 0.74,
                    "validation_f1": 0.57,
                    "f1_gap": 0.17,
                    "abs_f1_gap": 0.17,
                    "tie_break_reason": "best validation_f1",
                },
            ],
            columns=lab.MODEL_FEATURE_SET_DECISION_COLUMNS,
        )

    def test_choose_feature_set_for_model_prefers_best_validation_f1(self):
        audit = self.make_generalization_audit(
            dataset_name="medical",
            model_name="LogisticRegression",
            rows=[
                ("full", 0.70, 0.62),
                ("set_A_wrapper", 0.71, 0.66),
                ("set_B_tree", 0.73, 0.61),
            ],
        )
        winner = lab.choose_feature_set_for_model(audit, "medical", "LogisticRegression")
        self.assertEqual(winner, "set_A_wrapper")

    def test_choose_feature_set_for_model_uses_abs_gap_tie_break(self):
        audit = self.make_generalization_audit(
            dataset_name="medical",
            model_name="RandomForest",
            rows=[
                ("full", 0.90, 0.70),
                ("set_A_wrapper", 0.72, 0.70),
            ],
        )
        winner = lab.choose_feature_set_for_model(audit, "medical", "RandomForest")
        self.assertEqual(winner, "set_A_wrapper")

    def test_choose_feature_set_for_model_prefers_non_full_on_tie(self):
        audit = self.make_generalization_audit(
            dataset_name="finance",
            model_name="LogisticRegression",
            rows=[
                ("full", 0.70, 0.65),
                ("set_A_wrapper", 0.60, 0.65),
            ],
        )
        winner = lab.choose_feature_set_for_model(audit, "finance", "LogisticRegression")
        self.assertEqual(winner, "set_A_wrapper")

    def test_choose_feature_set_for_model_uses_lexicographic_tie_break(self):
        audit = self.make_generalization_audit(
            dataset_name="finance",
            model_name="RandomForest",
            rows=[
                ("set_B_tree", 0.70, 0.65),
                ("set_A_wrapper", 0.70, 0.65),
            ],
        )
        winner = lab.choose_feature_set_for_model(audit, "finance", "RandomForest")
        self.assertEqual(winner, "set_A_wrapper")

    def test_explain_feature_set_tie_break_variants(self):
        cases = [
            (
                pd.DataFrame(
                    [
                        {"feature_set": "full", "validation_f1": 0.66, "abs_f1_gap": 0.05, "full_penalty": 1},
                        {"feature_set": "set_A_wrapper", "validation_f1": 0.62, "abs_f1_gap": 0.01, "full_penalty": 0},
                    ]
                ),
                "best validation_f1",
            ),
            (
                pd.DataFrame(
                    [
                        {"feature_set": "full", "validation_f1": 0.66, "abs_f1_gap": 0.05, "full_penalty": 1},
                        {"feature_set": "set_A_wrapper", "validation_f1": 0.66, "abs_f1_gap": 0.01, "full_penalty": 0},
                    ]
                ),
                "tie on validation_f1 -> min abs_f1_gap",
            ),
            (
                pd.DataFrame(
                    [
                        {"feature_set": "full", "validation_f1": 0.66, "abs_f1_gap": 0.01, "full_penalty": 1},
                        {"feature_set": "set_A_wrapper", "validation_f1": 0.66, "abs_f1_gap": 0.01, "full_penalty": 0},
                    ]
                ),
                "tie on validation_f1 and abs_f1_gap -> prefer non-full",
            ),
            (
                pd.DataFrame(
                    [
                        {"feature_set": "set_B_tree", "validation_f1": 0.66, "abs_f1_gap": 0.01, "full_penalty": 0},
                        {"feature_set": "set_A_wrapper", "validation_f1": 0.66, "abs_f1_gap": 0.01, "full_penalty": 0},
                    ]
                ),
                "tie on validation_f1, abs_f1_gap and full/non-full -> lexicographic order",
            ),
        ]

        for feature_rows, expected in cases:
            with self.subTest(expected=expected):
                self.assertEqual(lab.explain_feature_set_tie_break(feature_rows), expected)

    def test_choose_validation_winner_prefers_logistic_regression_on_tie(self):
        validation_summary = pd.DataFrame(
            [
                {
                    "dataset": "medical",
                    "feature_set": "full",
                    "model": "RandomForest",
                    "validation_f1": 0.72,
                    "validation_roc_auc": 0.80,
                    "best_params_json": "{}",
                },
                {
                    "dataset": "medical",
                    "feature_set": "set_A_wrapper",
                    "model": "LogisticRegression",
                    "validation_f1": 0.72,
                    "validation_roc_auc": 0.80,
                    "best_params_json": "{}",
                },
            ]
        )
        winner = lab.choose_validation_winner(validation_summary, "medical")
        self.assertEqual(winner["model"], "LogisticRegression")

    def test_load_model_feature_set_decisions_validates_successfully(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "model_feature_set_decisions.csv"
            self.make_model_feature_set_decisions().to_csv(path, index=False)
            loaded = lab.load_model_feature_set_decisions(
                path=path,
                expected_datasets=["finance", "medical"],
                expected_models=["LogisticRegression", "RandomForest"],
            )

        self.assertEqual(list(loaded.columns), lab.MODEL_FEATURE_SET_DECISION_COLUMNS)
        self.assertEqual(len(loaded), 4)

    def test_load_model_feature_set_decisions_rejects_missing_column(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "model_feature_set_decisions.csv"
            self.make_model_feature_set_decisions().drop(columns=["tie_break_reason"]).to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "неверные колонки"):
                lab.load_model_feature_set_decisions(path=path)

    def test_load_model_feature_set_decisions_rejects_duplicate_pairs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "model_feature_set_decisions.csv"
            decisions = self.make_model_feature_set_decisions()
            duplicate_row = decisions.iloc[[0]].copy()
            pd.concat([decisions, duplicate_row], ignore_index=True).to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "уникальные пары dataset \\+ model"):
                lab.load_model_feature_set_decisions(path=path)

    def test_load_model_feature_set_decisions_rejects_missing_pair(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "model_feature_set_decisions.csv"
            decisions = self.make_model_feature_set_decisions()
            decisions = decisions[
                ~(
                    (decisions["dataset"] == "finance")
                    & (decisions["model"] == "RandomForest")
                )
            ]
            decisions.to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "неполон: отсутствуют строки"):
                lab.load_model_feature_set_decisions(path=path)

    def test_get_model_feature_set_decision_returns_single_row(self):
        decisions = self.make_model_feature_set_decisions()
        row = lab.get_model_feature_set_decision(
            decisions=decisions,
            dataset_name="medical",
            model_name="RandomForest",
        )
        self.assertEqual(row["selected_feature_set"], "set_C_hybrid")

    def test_preprocessed_feature_selector_zero_fills_missing_features(self):
        x_train = pd.DataFrame(
            {
                "age": [20, 30, 40],
                "color": ["red", "red", "red"],
            }
        )
        selector = lab.PreprocessedFeatureSelector(selected_features=["cat__color_blue"]).fit(x_train)
        transformed = selector.transform(pd.DataFrame({"age": [25, 35], "color": ["blue", "red"]}))
        self.assertEqual(transformed.shape, (2, 1))
        self.assertTrue(np.allclose(transformed, 0.0))

    def test_feature_set_helpers_cover_basic_cases(self):
        feature_sets = {
            "medical": {
                "set_B_tree": ["num__age"],
                "set_A_wrapper": ["num__bp"],
            }
        }
        self.assertEqual(
            lab.list_feature_set_names(feature_sets, "medical"),
            ["full", "set_A_wrapper", "set_B_tree"],
        )
        self.assertIsNone(lab.get_feature_set_features(feature_sets, "medical", "full"))
        self.assertEqual(
            lab.get_feature_set_features(feature_sets, "medical", "set_A_wrapper"),
            ["num__bp"],
        )


if __name__ == "__main__":
    unittest.main()
