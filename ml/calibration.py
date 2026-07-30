"""Probability calibration for the XGBoost classifier (Stage 4).

The classifier is trained with ``scale_pos_weight`` ~ 3.75 to buy recall on a
~1-in-5 positive rate. That inflates the positive-class scores: the raw
``predict_proba`` output ranks flights well but is NOT a calibrated frequency
(on the held-out test the raw ECE is 0.227 — a flight scored 0.25 is delayed
~9% of the time). This module remaps those scores onto the probability scale
so ``delay_probability`` means what it says, WITHOUT retraining and WITHOUT
disturbing the ranking.

Two maps are fit and BOTH persisted; serving uses ONE:

* **Platt** (a sigmoid of the logit, ``SERVING_METHOD``) is STRICTLY monotonic,
  so it preserves the complete score ordering — on the shipped run ROC-AUC and
  PR-AUC are BIT-UNCHANGED on the held-out test (Δ = 0). It is the shipped
  serving map, and ``build_calibration`` HARD-GATES on that preservation.
* **Isotonic** is a step function; it calibrates marginally on some tails but
  its ties coarsen the ranking, moving test PR-AUC by ~3e-3 (a mathematical
  property of ``average_precision`` under ties, not a leak). It is persisted
  for OFFLINE analysis only and is deliberately NOT the serving map, precisely
  because it does not preserve AUC.

On the held-out test both maps cut Brier ~0.191 -> ~0.135 and ECE ~0.227 ->
~0.017; Platt edges isotonic on both here (it transfers better across the
val->test base-rate gap, 0.260 -> 0.197), so shipping the invariant-safe map
costs nothing in calibration quality.

Fit discipline (CLAUDE.md §9): the map is fit on a time-based VALIDATION slice
carved from INSIDE the training window (the last 8 weeks — the same slice
ml.tuning uses), NEVER on the held-out test set. That slice is in-sample for
the shipped full-window classifier, but the in-sample optimism was measured
(an out-of-sample fit on a fit-set-only model moved test Brier by 6e-5 / ECE by
5e-3) and is negligible; keeping the full-window scores also makes the
calibration input distribution identical to what the shipped model produces.

TreeSHAP / margin attribution note: SHAP explains the RAW XGBoost margin
(log-odds), which sits UPSTREAM of this map. SHAP values do NOT sum to the
calibrated ``delay_probability`` — attribute the margin, then read the
calibrated probability as the reported output.

The persisted ``Calibrator`` is PURE NUMPY (two Platt scalars + the isotonic
threshold arrays) — no sklearn object is pickled, so the artifact is
version-independent and byte-deterministic to persist.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

# the serving map: strictly monotonic -> exact ROC/PR-AUC preservation
SERVING_METHOD = "platt"
# Platt preserves AUC to float roundoff (Δ = 0 on the shipped run); 1e-6 flags a
# real break (a leaked or retrained fit, or a non-monotonic map) — a real break
# moves AUC by >=1e-3 — without tripping on roundoff.
AUC_PRESERVE_TOL = 1e-6
ECE_BINS = 10
_EPS = 1e-6  # logit clip, matching the exploration


class CalibrationError(RuntimeError):
    """The serving calibrator failed the AUC-preservation invariant."""


def logit(p: np.ndarray) -> np.ndarray:
    # Clip keeps the logit finite. It can in principle collapse two DISTINCT
    # near-saturated scores (both >= 1 - _EPS) into one calibrated value — a tie
    # absent from the raw scores that could nudge average_precision. That never
    # occurred on the shipped run (test PR-AUC Δ = 0 exactly), and if it ever
    # did the AUC-preservation gate in build_calibration would FAIL THE BUILD
    # rather than ship a coarsened ranking — so do not loosen that tolerance.
    p = np.clip(np.asarray(p, dtype=float), _EPS, 1.0 - _EPS)
    return np.log(p / (1.0 - p))


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


@dataclass
class Calibrator:
    """A fitted probability calibrator, stored as plain numbers/arrays.

    ``transform`` applies the SERVING method (Platt) by default; the isotonic
    map is carried for offline comparison and is reachable via
    ``transform(scores, method="isotonic")``.
    """

    method: str  # the serving default (SERVING_METHOD)
    platt_a: float  # sigmoid(a * logit(p) + b)
    platt_b: float
    iso_x: np.ndarray  # isotonic interpolation thresholds (ascending)
    iso_y: np.ndarray
    val_window: tuple[str, str]  # (first, last) date of the fit slice
    n_val: int

    def _platt(self, scores: np.ndarray) -> np.ndarray:
        return _sigmoid(self.platt_a * logit(scores) + self.platt_b)

    def _isotonic(self, scores: np.ndarray) -> np.ndarray:
        # np.interp reproduces IsotonicRegression(out_of_bounds="clip").predict:
        # linear interpolation between thresholds, clipped to the endpoints
        # outside the fitted range (asserted equal at fit time).
        return np.interp(np.asarray(scores, dtype=float), self.iso_x, self.iso_y)

    def transform(self, scores: np.ndarray, method: str | None = None) -> np.ndarray:
        method = method or self.method
        if method == "platt":
            return self._platt(scores)
        if method == "isotonic":
            return self._isotonic(scores)
        raise ValueError(f"unknown calibration method {method!r}")


def fit_calibrator(
    p_val: np.ndarray,
    y_val: np.ndarray,
    val_window: tuple[str, str],
    method: str = SERVING_METHOD,
) -> Calibrator:
    """Fit both maps on validation-slice (score, label) pairs and pack the
    parameters into a pure-numpy Calibrator (serving method = ``method``)."""
    p_val = np.asarray(p_val, dtype=float)
    y_val = np.asarray(y_val)

    # Platt: LogisticRegression on the logit reproduces predict_proba[:, 1] =
    # sigmoid(coef * logit(p) + intercept). C=1e6 -> effectively unregularized.
    lr = LogisticRegression(C=1e6).fit(logit(p_val).reshape(-1, 1), y_val)

    iso = IsotonicRegression(out_of_bounds="clip").fit(p_val, y_val)
    iso_x = np.asarray(iso.X_thresholds_, dtype=float)
    iso_y = np.asarray(iso.y_thresholds_, dtype=float)
    # guard the np.interp equivalence: if a future sklearn changes threshold
    # semantics, the serving-side reproduction would silently drift.
    if not np.allclose(iso.predict(p_val), np.interp(p_val, iso_x, iso_y), atol=1e-9):
        raise CalibrationError("np.interp does not reproduce IsotonicRegression.predict")

    return Calibrator(
        method=method,
        platt_a=float(lr.coef_[0, 0]),
        platt_b=float(lr.intercept_[0]),
        iso_x=iso_x,
        iso_y=iso_y,
        val_window=(str(val_window[0]), str(val_window[1])),
        n_val=int(len(y_val)),
    )


def calibration_metrics(y: np.ndarray, p: np.ndarray, n_bins: int = ECE_BINS) -> dict:
    """Brier, ROC/PR-AUC, ECE/MCE and the equal-width reliability table
    (per bin: lo, hi, n, mean_pred, frac_pos)."""
    y = np.asarray(y)
    p = np.asarray(p, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, n_bins - 1)
    ece = mce = 0.0
    table = []
    for b in range(n_bins):
        sel = idx == b
        n = int(sel.sum())
        if n == 0:
            continue
        mp = float(p[sel].mean())
        fp = float(y[sel].mean())
        gap = abs(mp - fp)
        ece += (n / len(p)) * gap
        mce = max(mce, gap)
        table.append(
            {
                "lo": round(float(edges[b]), 2),
                "hi": round(float(edges[b + 1]), 2),
                "n": n,
                "mean_pred": round(mp, 4),
                "frac_pos": round(fp, 4),
            }
        )
    return {
        "brier": float(brier_score_loss(y, p)),
        "roc_auc": float(roc_auc_score(y, p)),
        "pr_auc": float(average_precision_score(y, p)),
        "ece": float(ece),
        "mce": float(mce),
        "reliability": table,
    }


def build_calibration(
    p_val: np.ndarray,
    y_val: np.ndarray,
    raw_test: np.ndarray,
    y_test: np.ndarray,
    val_window: tuple[str, str],
    method: str = SERVING_METHOD,
) -> tuple[Calibrator, dict]:
    """Fit the calibrator on the validation slice, evaluate raw / Platt /
    isotonic on the held-out TEST set, and HARD-GATE that the ACTUAL serving
    map (``method`` — whatever ``ml.serving`` applies) preserves ROC and PR-AUC.
    The test set is READ for reporting and the gate only — never fit on.
    Returns (calibrator, report)."""
    cal = fit_calibrator(p_val, y_val, val_window, method=method)

    raw_m = calibration_metrics(y_test, raw_test)
    platt_test = cal.transform(raw_test, method="platt")
    iso_test = cal.transform(raw_test, method="isotonic")
    platt_m = calibration_metrics(y_test, platt_test)
    iso_m = calibration_metrics(y_test, iso_test)

    # THE INVARIANT: calibration is a monotonic remap, so the SERVING map must
    # leave ROC/PR-AUC bit-unchanged. Gate the map serving ACTUALLY applies —
    # cal.transform(raw_test) dispatches on cal.method, the exact call
    # ml.serving.predict makes — NOT a hardcoded method, so the guarantee
    # protects whatever ships. With Platt active this equals platt_m (Δ=0); if
    # the serving method were isotonic, its tie-induced PR-AUC move (~3e-3)
    # would trip this gate instead of being silently validated as Platt. A move
    # beyond float roundoff means the map is non-monotonic, or the fit leaked,
    # or the base scores were retrained — fail loudly rather than ship a
    # silently-degraded ranking.
    served_m = calibration_metrics(y_test, cal.transform(raw_test))
    roc_delta = abs(served_m["roc_auc"] - raw_m["roc_auc"])
    pr_delta = abs(served_m["pr_auc"] - raw_m["pr_auc"])
    if roc_delta > AUC_PRESERVE_TOL or pr_delta > AUC_PRESERVE_TOL:
        raise CalibrationError(
            f"serving calibrator ({cal.method}) moved AUC beyond tol "
            f"{AUC_PRESERVE_TOL}: roc_delta={roc_delta:.3e} pr_delta={pr_delta:.3e} "
            f"(raw roc={raw_m['roc_auc']:.10f} pr={raw_m['pr_auc']:.10f})"
        )

    report = {
        "serving_method": cal.method,
        "val_window": list(cal.val_window),
        "n_val": cal.n_val,
        "platt_params": {"a": cal.platt_a, "b": cal.platt_b},
        "auc_preserved": {
            # deltas are for the SERVING map (cal.method), not a fixed method
            "method": cal.method,
            "roc_delta": roc_delta,
            "pr_delta": pr_delta,
            "tol": AUC_PRESERVE_TOL,
            "passed": True,
        },
        # isotonic is reported for the record but is NOT shipped; its AUC move
        # is expected tie-coarsening, quantified here.
        "isotonic_auc_move": {
            "roc_delta": abs(iso_m["roc_auc"] - raw_m["roc_auc"]),
            "pr_delta": abs(iso_m["pr_auc"] - raw_m["pr_auc"]),
        },
        "test": {"raw": raw_m, "platt": platt_m, "isotonic": iso_m},
    }
    return cal, report
