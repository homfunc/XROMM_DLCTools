from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .conversion import dlc_to_xma
from .models import TemporalOptimizationConfig


def _load_dlc_dataframe(data: str | pd.DataFrame) -> pd.DataFrame:
    if isinstance(data, pd.DataFrame):
        return data.copy()
    if not isinstance(data, str):
        raise TypeError("Expected path or pandas.DataFrame for DLC data input")
    if data.endswith(".h5"):
        return pd.read_hdf(data)
    if data.endswith(".csv"):
        return pd.read_csv(data, header=[0, 1, 2], index_col=0)
    raise ValueError("DLC input must be .h5 or .csv")


def _to_float(values: pd.Series) -> np.ndarray:
    return pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)


def _fill_nan(values: np.ndarray) -> np.ndarray:
    values = values.astype(float, copy=True)
    n = values.shape[0]
    if n == 0:
        return values
    idx = np.arange(n)
    valid = np.isfinite(values)
    if not valid.any():
        return np.zeros(n, dtype=float)
    if valid.sum() == 1:
        return np.full(n, values[valid][0], dtype=float)
    values[~valid] = np.interp(idx[~valid], idx[valid], values[valid])
    return values


def _rolling_median(values: np.ndarray, window: int = 5) -> np.ndarray:
    return (
        pd.Series(values, dtype=float)
        .rolling(window=window, center=True, min_periods=1)
        .median()
        .to_numpy(dtype=float)
    )


def _viterbi_temporal_path(
    x_obs: np.ndarray,
    y_obs: np.ndarray,
    confidence: np.ndarray,
    transition_weight: float,
) -> tuple[np.ndarray, np.ndarray]:
    n = x_obs.shape[0]
    if n == 0:
        return x_obs, y_obs

    x_filled = _fill_nan(x_obs)
    y_filled = _fill_nan(y_obs)
    conf = np.clip(np.nan_to_num(confidence, nan=0.0), 0.0, 1.0)

    x_trend = np.empty(n, dtype=float)
    y_trend = np.empty(n, dtype=float)
    x_trend[0] = x_filled[0]
    y_trend[0] = y_filled[0]
    if n > 1:
        x_trend[1] = x_filled[1]
        y_trend[1] = y_filled[1]
    for i in range(2, n):
        x_trend[i] = 2.0 * x_filled[i - 1] - x_filled[i - 2]
        y_trend[i] = 2.0 * y_filled[i - 1] - y_filled[i - 2]

    x_med = _rolling_median(x_filled, window=5)
    y_med = _rolling_median(y_filled, window=5)

    cand_x = np.column_stack([x_filled, x_trend, x_med])
    cand_y = np.column_stack([y_filled, y_trend, y_med])
    n_states = cand_x.shape[1]

    alt_penalty = conf + 0.05
    obs_penalty = 1.0 - conf
    obs_cost = np.column_stack([obs_penalty, alt_penalty, alt_penalty])

    cost = np.full((n, n_states), np.inf, dtype=float)
    back = np.zeros((n, n_states), dtype=int)
    cost[0] = obs_cost[0]

    for t in range(1, n):
        for k in range(n_states):
            dx = cand_x[t, k] - cand_x[t - 1]
            dy = cand_y[t, k] - cand_y[t - 1]
            trans = transition_weight * (dx * dx + dy * dy)
            prev_cost = cost[t - 1] + trans
            best_prev = int(np.argmin(prev_cost))
            cost[t, k] = obs_cost[t, k] + prev_cost[best_prev]
            back[t, k] = best_prev

    states = np.zeros(n, dtype=int)
    states[-1] = int(np.argmin(cost[-1]))
    for t in range(n - 2, -1, -1):
        states[t] = back[t + 1, states[t + 1]]

    return cand_x[np.arange(n), states], cand_y[np.arange(n), states]


def _kalman_smooth_1d(
    observations: np.ndarray,
    confidence: np.ndarray,
    process_noise: float,
) -> np.ndarray:
    n = observations.shape[0]
    if n == 0:
        return observations
    if n == 1:
        return observations.copy()

    z = _fill_nan(observations)
    conf = np.clip(np.nan_to_num(confidence, nan=0.0), 0.0, 1.0)

    f = np.array([[1.0, 1.0], [0.0, 1.0]], dtype=float)
    h = np.array([[1.0, 0.0]], dtype=float)
    q = process_noise * np.array([[0.25, 0.5], [0.5, 1.0]], dtype=float)
    i2 = np.eye(2, dtype=float)

    x_filt = np.zeros((n, 2), dtype=float)
    p_filt = np.zeros((n, 2, 2), dtype=float)
    x_pred = np.zeros((n, 2), dtype=float)
    p_pred = np.zeros((n, 2, 2), dtype=float)

    x_filt[0] = np.array([z[0], 0.0], dtype=float)
    p_filt[0] = np.eye(2, dtype=float)
    x_pred[0] = x_filt[0]
    p_pred[0] = p_filt[0]

    for t in range(1, n):
        x_prior = f @ x_filt[t - 1]
        p_prior = f @ p_filt[t - 1] @ f.T + q

        r = max((1.0 - conf[t]) + 1e-3, 1e-3)
        s = h @ p_prior @ h.T + r
        k = (p_prior @ h.T) / s[0, 0]
        innovation = z[t] - (h @ x_prior)[0]
        x_post = x_prior + (k[:, 0] * innovation)
        p_post = (i2 - (k @ h)) @ p_prior

        x_pred[t] = x_prior
        p_pred[t] = p_prior
        x_filt[t] = x_post
        p_filt[t] = p_post

    x_smooth = x_filt.copy()
    p_smooth = p_filt.copy()
    for t in range(n - 2, -1, -1):
        gain = p_filt[t] @ f.T @ np.linalg.pinv(p_pred[t + 1])
        x_smooth[t] = x_filt[t] + gain @ (x_smooth[t + 1] - x_pred[t + 1])
        p_smooth[t] = p_filt[t] + gain @ (p_smooth[t + 1] - p_pred[t + 1]) @ gain.T

    return x_smooth[:, 0]


def _coordinate_label(columns: pd.MultiIndex, name: str) -> str:
    coord_values = columns.get_level_values("coords")
    for value in coord_values:
        if str(value).lower() == name:
            return value
    raise KeyError(f"Could not find coord '{name}' in DLC columns")


def optimize_dlc_dataframe(
    data: pd.DataFrame,
    config: TemporalOptimizationConfig,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    if not isinstance(data.columns, pd.MultiIndex):
        raise ValueError("Expected DLC dataframe with MultiIndex columns")

    scorer = data.columns.get_level_values("scorer")[0]
    bodyparts = list(data.columns.get_level_values("bodyparts").unique())
    x_label = _coordinate_label(data.columns, "x")
    y_label = _coordinate_label(data.columns, "y")
    l_label = _coordinate_label(data.columns, "likelihood")

    optimized = data.copy()
    residual_columns: list[np.ndarray] = []
    confidence_columns: list[np.ndarray] = []

    for bodypart in bodyparts:
        x_obs = _to_float(data[(scorer, bodypart, x_label)])
        y_obs = _to_float(data[(scorer, bodypart, y_label)])
        conf = np.clip(_to_float(data[(scorer, bodypart, l_label)]), 0.0, 1.0)

        x_vit, y_vit = _viterbi_temporal_path(
            x_obs, y_obs, conf, config.transition_weight
        )
        x_smooth = _kalman_smooth_1d(x_vit, conf, config.process_noise)
        y_smooth = _kalman_smooth_1d(y_vit, conf, config.process_noise)

        x_ref = _fill_nan(x_obs)
        y_ref = _fill_nan(y_obs)
        blend = conf
        x_final = blend * x_ref + (1.0 - blend) * x_smooth
        y_final = blend * y_ref + (1.0 - blend) * y_smooth

        optimized[(scorer, bodypart, x_label)] = x_final
        optimized[(scorer, bodypart, y_label)] = y_final

        residual = np.sqrt((x_final - x_ref) ** 2 + (y_final - y_ref) ** 2)
        residual_columns.append(residual)
        confidence_columns.append(conf)

    residual_matrix = np.column_stack(residual_columns) if residual_columns else np.empty((0, 0))
    confidence_matrix = (
        np.column_stack(confidence_columns) if confidence_columns else np.empty((0, 0))
    )
    return optimized, residual_matrix, confidence_matrix


def _build_review_frames(
    frame_residual: np.ndarray,
    frame_confidence: np.ndarray,
    top_k: int,
) -> list[dict]:
    if frame_residual.size == 0:
        return []
    review_score = frame_residual * (2.0 - frame_confidence)
    ranked = np.argsort(-review_score)[: max(int(top_k), 0)]
    return [
        {
            "frame_index": int(idx),
            "review_score": round(float(review_score[idx]), 6),
            "mean_residual_px": round(float(frame_residual[idx]), 6),
            "mean_confidence": round(float(frame_confidence[idx]), 6),
        }
        for idx in ranked
    ]


def optimize_dlc_predictions(
    cam1data: str | pd.DataFrame,
    cam2data: str | pd.DataFrame,
    trialname: str,
    savepath: str,
    config: TemporalOptimizationConfig | None = None,
) -> dict:
    config = config or TemporalOptimizationConfig()
    cam1 = _load_dlc_dataframe(cam1data)
    cam2 = _load_dlc_dataframe(cam2data)

    cam1_opt, cam1_residual, cam1_conf = optimize_dlc_dataframe(cam1, config)
    cam2_opt, cam2_residual, cam2_conf = optimize_dlc_dataframe(cam2, config)

    frame_count = min(cam1_opt.shape[0], cam2_opt.shape[0])
    if frame_count == 0:
        frame_residual = np.array([], dtype=float)
        frame_confidence = np.array([], dtype=float)
    else:
        cam1_frame_res = cam1_residual[:frame_count].mean(axis=1) if cam1_residual.size else np.zeros(frame_count)
        cam2_frame_res = cam2_residual[:frame_count].mean(axis=1) if cam2_residual.size else np.zeros(frame_count)
        frame_residual = (cam1_frame_res + cam2_frame_res) / 2.0

        cam1_frame_conf = cam1_conf[:frame_count].mean(axis=1) if cam1_conf.size else np.ones(frame_count)
        cam2_frame_conf = cam2_conf[:frame_count].mean(axis=1) if cam2_conf.size else np.ones(frame_count)
        frame_confidence = (cam1_frame_conf + cam2_frame_conf) / 2.0

    optimized_trialname = f"{trialname}-Optimized"
    dlc_to_xma(cam1_opt, cam2_opt, optimized_trialname, savepath)

    out_dir = Path(savepath)
    report_path = out_dir / f"{trialname}-OptimizationReport.json"
    report = {
        "trial_name": trialname,
        "optimized_trial_name": optimized_trialname,
        "frame_count": int(frame_count),
        "mean_residual_px": round(float(frame_residual.mean()), 6)
        if frame_residual.size
        else 0.0,
        "mean_confidence": round(float(frame_confidence.mean()), 6)
        if frame_confidence.size
        else 0.0,
        "config": {
            "transition_weight": config.transition_weight,
            "process_noise": config.process_noise,
            "top_k_review_frames": config.top_k_review_frames,
        },
        "review_frames": _build_review_frames(
            frame_residual, frame_confidence, config.top_k_review_frames
        ),
        "outputs": {
            "csv": str(out_dir / f"{optimized_trialname}-Predicted2DPoints.csv"),
            "h5": str(out_dir / f"{optimized_trialname}-Predicted2DPoints.h5"),
            "report_json": str(report_path),
        },
    }
    report_path.write_text(json.dumps(report, indent=2))
    return report
