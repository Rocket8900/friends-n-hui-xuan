from __future__ import annotations
from flask import Blueprint, request, jsonify
import math
import numpy as np
from typing import List, Optional

blankety_bp = Blueprint("blankety_bp", __name__)

def _to_float_array(seq: List[Optional[float]]) -> np.ndarray:
    arr = np.array(seq, dtype=float)
    return arr


def _linear_interp_edges(y: np.ndarray) -> np.ndarray:
    x = np.arange(len(y))
    mask = ~np.isnan(y)
    if mask.sum() == 0:
        return np.zeros_like(y)
    z = y.copy()
    z[~mask] = np.interp(x[~mask], x[mask], z[mask])
    # ensure edges filled by nearest valid (np.interp already does linear extrapolation using endpoints)
    first = np.argmax(mask)
    last = len(mask) - 1 - np.argmax(mask[::-1])
    z[:first] = z[first]
    z[last + 1 :] = z[last]
    return z


def _fit_poly_trend(y: np.ndarray, mask_obs: np.ndarray) -> np.ndarray:
    x = np.arange(len(y))
    xo = x[mask_obs]
    yo = y[mask_obs]
    if len(xo) < 3:
        # too few points, fallback to constant mean
        mu = float(np.nanmean(yo)) if yo.size else 0.0
        return np.full_like(y, mu)

    best_yhat = None
    best_bic = None
    for deg in (1, 2):
        try:
            coef = np.polyfit(xo, yo, deg)
            yhat = np.polyval(coef, x)
            resid = yo - np.polyval(coef, xo)
            rss = float((resid ** 2).sum())
            k = deg + 1
            bic = _bic(len(xo), rss, k)
            if (best_bic is None) or (bic < best_bic):
                best_bic = bic
                best_yhat = yhat
        except Exception:
            continue
    if best_yhat is None:
        mu = float(np.nanmean(yo)) if yo.size else 0.0
        return np.full_like(y, mu)
    return best_yhat


def _bic(n: int, rss: float, k: int) -> float:
    rss = max(rss, 1e-12)
    return n * math.log(rss / n) + k * math.log(n)


def _dominant_k(y: np.ndarray, max_k: int = 2) -> List[int]:
    n = len(y)
    t = np.arange(n)
    # detrend
    try:
        p = np.polyfit(t, y, 1)
        ydt = y - np.polyval(p, t)
    except Exception:
        ydt = y - np.nanmean(y)

    fft = np.fft.rfft(ydt)
    power = np.abs(fft)
    k = np.arange(len(power))
    valid = k > 0
    # limit to reasonable periods
    period = np.where(k == 0, np.inf, n / np.maximum(k, 1))
    valid &= (period >= 6) & (period <= n / 2)
    if not valid.any():
        return []
    idxs = np.argsort(power[valid])[::-1]
    kvals = k[valid][idxs]
    return list(map(int, kvals[:max_k]))


def _fit_seasonal(y: np.ndarray, ks: List[int]) -> np.ndarray:
    n = len(y)
    if not ks:
        return np.zeros(n)
    t = np.arange(n)
    cols = []
    for k in ks:
        omega = 2 * np.pi * k / n
        cols.append(np.sin(omega * t))
        cols.append(np.cos(omega * t))
    X = np.vstack(cols).T
    try:
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        return X.dot(coef)
    except Exception:
        return np.zeros(n)


def _fit_ar1_phi(resid: np.ndarray) -> float:
    if resid.size < 3:
        return 0.0
    r0 = float(np.dot(resid[:-1], resid[:-1]))
    if r0 <= 1e-12:
        return 0.0
    phi = float(np.dot(resid[1:], resid[:-1]) / r0)
    return float(np.clip(phi, -0.98, 0.98))


def _smooth(resid: np.ndarray, w: int = 7) -> np.ndarray:
    if resid.size < 3:
        return resid
    w = max(3, min(w, resid.size // 2 * 2 + 1))
    kernel = np.ones(w) / w
    return np.convolve(resid, kernel, mode="same")


def impute_series_one(series: List[Optional[float]]) -> List[float]:
    y = _to_float_array(series)
    n = len(y)
    if n == 0:
        return []
    mask_obs = ~np.isnan(y)
    if mask_obs.sum() == 0:
        # no observations -> zeros
        return [0.0] * n

    if mask_obs.sum() <= 5:
        # very sparse: simple forward/backward fill then mean
        filled = _linear_interp_edges(y)
        filled[np.isnan(filled)] = np.nanmean(y[mask_obs])
        filled[np.isfinite(filled) == False] = 0.0
        return [float(v) for v in filled]

    # initial fill with linear interp
    filled = _linear_interp_edges(y)

    # refine using trend + seasonal + residual smoothing
    for _ in range(3):
        trend = _fit_poly_trend(filled, mask_obs)
        ks = _dominant_k(filled, max_k=2)
        seasonal = _fit_seasonal(filled - trend, ks)
        baseline = trend + seasonal
        resid = filled - baseline
        phi = _fit_ar1_phi(resid[mask_obs]) if mask_obs.any() else 0.0

        # Predict residuals via AR(1) initialized from observed residuals
        resid_pred = np.zeros_like(resid)
        # seed with first observed residual
        first_obs_idx = np.argmax(mask_obs)
        resid_pred[first_obs_idx] = resid[first_obs_idx]
        for i in range(first_obs_idx + 1, n):
            resid_pred[i] = phi * resid_pred[i - 1]

        resid_sm = 0.6 * resid_pred + 0.4 * _smooth(resid, w=9)
        model = baseline + resid_sm

        # Update only missing positions; keep observed values exact
        filled[~mask_obs] = model[~mask_obs]

        # quick stability clamp per-iteration
        finite = filled[np.isfinite(filled)]
        if finite.size:
            lo, hi = np.percentile(finite, [0.5, 99.5])
            rng = max(1e-6, hi - lo)
            filled = np.clip(filled, lo - 0.5 * rng, hi + 0.5 * rng)

    # final safety: put observed back exactly and ensure no NaNs
    filled[mask_obs] = y[mask_obs]
    filled[~np.isfinite(filled)] = 0.0
    return [float(v) for v in filled]


@blankety_bp.post("/blankety")
def blankety():
    try:
        payload = request.get_json(force=True, silent=False)
        if not isinstance(payload, dict) or "series" not in payload:
            return jsonify({"error": "Expected JSON body with key 'series'"}), 400

        series_all = payload["series"]
        if not isinstance(series_all, list) or len(series_all) != 100:
            return jsonify({"error": "'series' must be a list of exactly 100 lists"}), 400

        # Validate lengths quickly
        for i, s in enumerate(series_all):
            if not isinstance(s, list) or len(s) != 1000:
                return jsonify({"error": f"Each series must be a list of length 1000 (bad at index {i})"}), 400

        # Impute each series. Keep memory low by streaming.
        answers = []
        for s in series_all:
            imputed = impute_series_one(s)
            answers.append(imputed)

        # Final validation: ensure no nulls
        for i, row in enumerate(answers):
            if any([v is None or (isinstance(v, float) and (math.isinf(v) or math.isnan(v))) for v in row]):
                return jsonify({"error": f"Invalid numeric produced for series {i}"}), 500

        return jsonify({"answer": answers})
    except Exception as e:
        return jsonify({"error": str(e)}), 500