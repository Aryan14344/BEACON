"""
Evaluation harness for PhishBP.

Implements all metrics from the paper:
  - ROC-AUC, PR-AUC
  - TPR @ 0.1% FPR (operational metric — the key number for security)
  - TPR @ 0.5% FPR
  - Expected Calibration Error (ECE)
  - Runtime and peak GPU memory

Also handles:
  - Multiple seeds + temporal windows → mean ± std
  - Mann-Whitney U significance test across windows
  - Weights & Biases logging

Usage
-----
>>> from src.eval.metrics import evaluate, compare_methods
>>> results = evaluate(y_true, y_score, method_name="H3")
>>> compare_methods(all_results)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

try:
    from sklearn.metrics import (
        roc_auc_score, average_precision_score,
        roc_curve, precision_recall_curve,
        brier_score_loss,
    )
    from scipy.stats import mannwhitneyu
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


# ─────────────────────────────────────────────────────────────────────────────
# Result container
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class EvalResult:
    method:    str
    window:    int
    seed:      int
    roc_auc:   float = 0.0
    pr_auc:    float = 0.0
    tpr_01fpr: float = 0.0   # TPR @ FPR=0.1%
    tpr_05fpr: float = 0.0   # TPR @ FPR=0.5%
    ece:       float = 0.0   # Expected Calibration Error
    brier:     float = 0.0   # Brier score (calibration)
    runtime_s: float = 0.0   # Wall-clock seconds
    gpu_mb:    float = 0.0   # Peak GPU memory in MB
    n_test:    int   = 0
    n_phish:   int   = 0
    extra:     dict  = field(default_factory=dict)


# ─────────────────────────────────────────────────────────────────────────────
# Core metric computation
# ─────────────────────────────────────────────────────────────────────────────

def tpr_at_fpr(y_true: np.ndarray, y_score: np.ndarray, target_fpr: float) -> float:
    """
    Compute TPR at a specific FPR operating point.

    This is the key metric for phishing detection:
    "What fraction of phishing sites do we catch while only
     mislabelling X% of benign sites?"

    Parameters
    ----------
    target_fpr : e.g. 0.001 for 0.1% FPR
    """
    if not HAS_SKLEARN:
        return 0.0
    fpr, tpr, _ = roc_curve(y_true, y_score)
    # Find the threshold where FPR is closest to (but does not exceed) target
    valid = fpr <= target_fpr
    if not valid.any():
        return 0.0
    return float(tpr[valid][-1])  # highest TPR where FPR constraint satisfied


def expected_calibration_error(
    y_true: np.ndarray,
    y_score: np.ndarray,
    n_bins: int = 15,
) -> float:
    """
    ECE: measures how well predicted probabilities match empirical frequencies.

    Lower is better (0 = perfectly calibrated).
    Important for security: uncalibrated models produce misleading confidence.
    """
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (y_score >= lo) & (y_score < hi)
        if mask.sum() == 0:
            continue
        bin_conf = y_score[mask].mean()
        bin_acc  = y_true[mask].mean()
        ece += (mask.sum() / n) * abs(bin_conf - bin_acc)
    return float(ece)


def evaluate(
    y_true: np.ndarray,
    y_score: np.ndarray,
    method: str = "unknown",
    window: int = 0,
    seed: int = 0,
    runtime_s: float = 0.0,
    gpu_mb: float = 0.0,
) -> EvalResult:
    """
    Compute all metrics for a single (method, window, seed) run.

    Parameters
    ----------
    y_true  : (N,) binary ground truth (0=benign, 1=phishing)
    y_score : (N,) predicted phishing probability ∈ [0,1]
    """
    if not HAS_SKLEARN:
        raise ImportError("scikit-learn required: pip install scikit-learn")

    y_true  = np.asarray(y_true,  dtype=int)
    y_score = np.asarray(y_score, dtype=float)

    # Guard: at least one positive and one negative sample
    if y_true.sum() == 0 or (1 - y_true).sum() == 0:
        return EvalResult(method=method, window=window, seed=seed,
                          n_test=len(y_true), n_phish=int(y_true.sum()))

    result = EvalResult(
        method    = method,
        window    = window,
        seed      = seed,
        roc_auc   = float(roc_auc_score(y_true, y_score)),
        pr_auc    = float(average_precision_score(y_true, y_score)),
        tpr_01fpr = tpr_at_fpr(y_true, y_score, 0.001),
        tpr_05fpr = tpr_at_fpr(y_true, y_score, 0.005),
        ece       = expected_calibration_error(y_true, y_score),
        brier     = float(brier_score_loss(y_true, y_score)),
        runtime_s = runtime_s,
        gpu_mb    = gpu_mb,
        n_test    = len(y_true),
        n_phish   = int(y_true.sum()),
    )
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Aggregation across seeds and windows
# ─────────────────────────────────────────────────────────────────────────────

def aggregate_results(results: list[EvalResult]) -> dict:
    """
    Compute mean ± std for each metric across seeds and windows.

    Returns dict of {metric_name: {"mean": float, "std": float}}.
    """
    metrics = ["roc_auc", "pr_auc", "tpr_01fpr", "tpr_05fpr", "ece", "brier", "runtime_s"]
    agg = {}
    for m in metrics:
        vals = [getattr(r, m) for r in results]
        agg[m] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
    return agg


def compare_methods(
    all_results: dict[str, list[EvalResult]],
    primary_metric: str = "roc_auc",
    significance_level: float = 0.05,
) -> None:
    """
    Print a comparison table of all methods and run Mann-Whitney U tests.

    Parameters
    ----------
    all_results : dict mapping method_name → list of EvalResult
    """
    print("\n" + "=" * 100)
    print(f"{'Method':<25}  {'ROC-AUC':>10}  {'PR-AUC':>10}  {'TPR@0.1%FPR':>12}  {'TPR@0.5%FPR':>12}  {'ECE':>8}  {'Time(s)':>9}")
    print("-" * 100)

    for method, results in all_results.items():
        agg = aggregate_results(results)
        def fmt(key): return f"{agg[key]['mean']:.4f}±{agg[key]['std']:.4f}"
        print(
            f"  {method:<23}  {fmt('roc_auc'):>10}  {fmt('pr_auc'):>10}  "
            f"{fmt('tpr_01fpr'):>12}  {fmt('tpr_05fpr'):>12}  "
            f"{agg['ece']['mean']:.4f}  {agg['runtime_s']['mean']:>8.2f}s"
        )

    # Mann-Whitney U significance tests (pairwise vs best method)
    if not HAS_SKLEARN:
        return

    methods = list(all_results.keys())
    best_method = max(
        methods,
        key=lambda m: aggregate_results(all_results[m])[primary_metric]["mean"]
    )
    print(f"\n  Significance tests vs best method ({best_method}) on {primary_metric}:")

    best_vals = [getattr(r, primary_metric) for r in all_results[best_method]]
    for method, results in all_results.items():
        if method == best_method:
            continue
        vals = [getattr(r, primary_metric) for r in results]
        stat, p = mannwhitneyu(best_vals, vals, alternative="greater")
        sig = "✓ significant" if p < significance_level else "✗ not significant"
        print(f"    {best_method} > {method}: p={p:.4f}  {sig}")
    print("=" * 100)


# ─────────────────────────────────────────────────────────────────────────────
# GPU memory tracker
# ─────────────────────────────────────────────────────────────────────────────

class GPUMemoryTracker:
    """Context manager to track peak GPU memory usage."""

    def __init__(self, device: int = 0):
        self.device = device
        self.peak_mb = 0.0

    def __enter__(self):
        if HAS_TORCH and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(self.device)
        return self

    def __exit__(self, *args):
        if HAS_TORCH and torch.cuda.is_available():
            self.peak_mb = torch.cuda.max_memory_allocated(self.device) / 1e6

    @property
    def mb(self) -> float:
        return self.peak_mb


# ─────────────────────────────────────────────────────────────────────────────
# WandB logging helper
# ─────────────────────────────────────────────────────────────────────────────

def log_to_wandb(result: EvalResult, step: Optional[int] = None) -> None:
    """Log an EvalResult to Weights & Biases (no-op if wandb not configured)."""
    try:
        import wandb
        if wandb.run is None:
            return
        log_dict = {
            f"{result.method}/roc_auc":   result.roc_auc,
            f"{result.method}/pr_auc":    result.pr_auc,
            f"{result.method}/tpr_01fpr": result.tpr_01fpr,
            f"{result.method}/tpr_05fpr": result.tpr_05fpr,
            f"{result.method}/ece":       result.ece,
            f"{result.method}/brier":     result.brier,
            f"{result.method}/runtime_s": result.runtime_s,
            "window": result.window,
            "seed":   result.seed,
        }
        wandb.log(log_dict, step=step)
    except ImportError:
        pass
