"""Evaluation protocol metrics (Section 5 of the plan).

All functions take:  score g (higher = more reliable), correct y in {0,1}.
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


def risk_coverage_curve(score: np.ndarray, correct: np.ndarray):
    """Answer the top-c fraction by score; risk = error rate among answered."""
    order = np.argsort(-score)                      # descending score
    c_sorted = correct[order]
    n = len(c_sorted)
    coverage = np.arange(1, n + 1) / n
    risk = np.cumsum(1 - c_sorted) / np.arange(1, n + 1)
    return coverage, risk


def aurc(score: np.ndarray, correct: np.ndarray) -> float:
    cov, risk = risk_coverage_curve(score, correct)
    trapezoid = getattr(np, "trapezoid", None) or np.trapz  # numpy 2.x / 1.x
    return float(trapezoid(risk, cov))


def oracle_aurc(correct: np.ndarray) -> float:
    """Rank by true correctness (ties broken to put errors last)."""
    return aurc(correct.astype(float), correct)


def selective_accuracy(score: np.ndarray, correct: np.ndarray, coverage: float) -> float:
    n = len(score)
    k = max(1, int(round(coverage * n)))
    idx = np.argsort(-score)[:k]
    return float(correct[idx].mean())


def auroc(score: np.ndarray, correct: np.ndarray) -> float:
    """AUROC that survives NaNs / infs in the score: rows with non-finite
    scores are dropped (with a warning if it's a large fraction), and a
    single-class y is reported as NaN rather than crashing."""
    score = np.asarray(score, dtype=float)
    correct = np.asarray(correct)
    ok = np.isfinite(score)
    dropped = int((~ok).sum())
    if dropped:
        frac = dropped / len(score)
        if frac > 0.10:
            print(f"  [auroc] warning: dropped {dropped}/{len(score)} "
                  f"non-finite scores ({frac:.1%})")
        score, correct = score[ok], correct[ok]
    if len(correct) == 0 or correct.min() == correct.max():
        return float("nan")
    return float(roc_auc_score(correct, score))


def ece(conf: np.ndarray, correct: np.ndarray, n_bins: int = 15) -> float:
    """Expected calibration error of the VLM's letter confidence — tests whether
    the miscalibration-under-shift premise (Guo'17 / Ovadia'19) transfers."""
    bins = np.linspace(0, 1, n_bins + 1)
    e, n = 0.0, len(conf)
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.sum():
            e += m.sum() / n * abs(correct[m].mean() - conf[m].mean())
    return float(e)


def gain_beyond_confidence(conf: np.ndarray, g: np.ndarray, correct: np.ndarray,
                           conf_fit=None, both_fit=None):
    """Requirement (3): Delta = AUROC(sigma(w1*conf + w2*g)) - AUROC(sigma(w1*conf)).

    Fit the logistic combiners on a fit split, report Delta on the given
    (held-out) arrays. Pass pre-fitted models via conf_fit/both_fit when the fit
    split differs from the eval split (it should!).
    """
    Xc = conf.reshape(-1, 1)
    Xb = np.stack([conf, g], axis=1)
    if conf_fit is None:
        conf_fit = LogisticRegression(max_iter=1000).fit(Xc, correct)
    if both_fit is None:
        both_fit = LogisticRegression(max_iter=1000).fit(Xb, correct)
    a_conf = auroc(conf_fit.predict_proba(Xc)[:, 1], correct)
    a_both = auroc(both_fit.predict_proba(Xb)[:, 1], correct)
    return a_both - a_conf, conf_fit, both_fit


def bootstrap_ci(metric_fn, score: np.ndarray, correct: np.ndarray,
                 n_boot: int = 1000, alpha: float = 0.05, seed: int = 0):
    """Percentile bootstrap CI over QUESTIONS (resample inputs, not
    predictions). Returns (point, lo, hi). Use for AUROC/AURC error bars."""
    rng = np.random.RandomState(seed)
    point = metric_fn(score, correct)
    n = len(score)
    stats = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        try:
            stats.append(metric_fn(score[idx], correct[idx]))
        except ValueError:
            continue
    lo, hi = np.percentile(stats, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(point), float(lo), float(hi)
