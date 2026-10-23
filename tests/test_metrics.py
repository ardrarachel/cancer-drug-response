"""
Unit tests for regression and correlation metrics edge cases.
"""

import numpy as np
import pytest
import torch

from src.evaluate import compute_metrics


class TestMetricsCalculation:
    """Test suite for evaluate.py metrics and stability guards."""

    def test_perfect_correlation(self):
        """Verify metric values when predictions match ground truth identically."""
        y_true = np.array([1.2, -0.5, 3.4, 2.1, 0.0])
        y_pred = y_true.copy()

        metrics = compute_metrics(y_true, y_pred)
        assert pytest.approx(metrics["rmse"], abs=1e-6) == 0.0
        assert pytest.approx(metrics["mae"], abs=1e-6) == 0.0
        assert pytest.approx(metrics["pcc"], abs=1e-6) == 1.0
        assert pytest.approx(metrics["scc"], abs=1e-6) == 1.0
        assert pytest.approx(metrics["r2"], abs=1e-6) == 1.0
        assert metrics["pcc_pvalue"] == 0.0

    def test_inverted_correlation(self):
        """Verify metric values for perfectly inverted predictions."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = -y_true

        metrics = compute_metrics(y_true, y_pred)
        assert pytest.approx(metrics["pcc"], abs=1e-6) == -1.0
        assert pytest.approx(metrics["scc"], abs=1e-6) == -1.0

    def test_zero_variance_guard(self):
        """Verify that constant predictions (variance = 0) do not trigger division by zero."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([2.5, 2.5, 2.5, 2.5, 2.5])  # Zero variance

        metrics = compute_metrics(y_true, y_pred)
        assert not np.isnan(metrics["rmse"])
        assert not np.isnan(metrics["mae"])
        assert metrics["pcc"] == 0.0
        assert metrics["scc"] == 0.0
        assert metrics["pcc_pvalue"] == 1.0

    def test_torch_tensor_inputs(self):
        """Verify support for PyTorch Tensor inputs directly."""
        y_true = torch.tensor([1.0, 2.0, 3.0, 4.0])
        y_pred = torch.tensor([1.1, 1.9, 3.2, 3.9])

        metrics = compute_metrics(y_true, y_pred)
        assert metrics["rmse"] > 0.0
        assert metrics["pcc"] > 0.9

    def test_empty_and_single_sample_guards(self):
        """Verify robustness when passed empty or single-element arrays."""
        empty_res = compute_metrics(np.array([]), np.array([]))
        assert np.isnan(empty_res["rmse"])

        single_res = compute_metrics(np.array([1.0]), np.array([1.0]))
        assert single_res["rmse"] == 0.0
        assert single_res["pcc"] == 0.0
