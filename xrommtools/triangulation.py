from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .models import StereoTriangulationConfig


def _load_dlc_dataframe(data: str | pd.DataFrame) -> pd.DataFrame:
    if isinstance(data, pd.DataFrame):
        return data.copy()
    if not isinstance(data, str):
        raise TypeError("Expected path or pandas.DataFrame for DLC input")
    if data.endswith(".h5"):
        return pd.read_hdf(data)
    if data.endswith(".csv"):
        return pd.read_csv(data, header=[0, 1, 2], index_col=0)
    raise ValueError("DLC input must be .h5 or .csv")


def _coordinate_label(columns: pd.MultiIndex, name: str) -> str:
    coord_values = columns.get_level_values("coords")
    for value in coord_values:
        if str(value).lower() == name:
            return value
    raise KeyError(f"Could not find coord '{name}' in DLC columns")


def _to_float(values: pd.Series) -> np.ndarray:
    return pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)


def _load_projection_matrix(path: str) -> np.ndarray:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Projection matrix file not found: {path}")

    if p.suffix == ".npy":
        matrix = np.load(p)
    elif p.suffix == ".json":
        payload = json.loads(p.read_text())
        if isinstance(payload, dict):
            if "projection_matrix" in payload:
                payload = payload["projection_matrix"]
            elif "P" in payload:
                payload = payload["P"]
        matrix = np.asarray(payload, dtype=float)
    else:
        raise ValueError("Projection matrix must be .npy or .json")

    if matrix.shape != (3, 4):
        raise ValueError(f"Projection matrix must be shape (3, 4), got {matrix.shape}")
    return matrix.astype(float)


def _load_rigid_constraints(path: str | None) -> list[tuple[str, str, float]]:
    if path is None:
        return []
    payload = json.loads(Path(path).read_text())
    pairs = payload["pairs"] if isinstance(payload, dict) and "pairs" in payload else payload
    constraints: list[tuple[str, str, float]] = []

    if isinstance(pairs, dict):
        for key, value in pairs.items():
            if "-" in key:
                marker_a, marker_b = key.split("-", 1)
            elif "," in key:
                marker_a, marker_b = key.split(",", 1)
            else:
                continue
            constraints.append((marker_a.strip(), marker_b.strip(), float(value)))
        return constraints

    if not isinstance(pairs, list):
        raise ValueError("Rigid constraints JSON must contain a list or mapping of pairs")

    for item in pairs:
        if not isinstance(item, dict):
            continue
        marker_a = item.get("a") or item.get("marker_a")
        marker_b = item.get("b") or item.get("marker_b")
        distance = item.get("distance")
        if marker_a is None or marker_b is None or distance is None:
            continue
        constraints.append((str(marker_a), str(marker_b), float(distance)))
    return constraints


def _project_point(P: np.ndarray, X: np.ndarray) -> tuple[float, float]:
    x_h = P @ np.array([X[0], X[1], X[2], 1.0], dtype=float)
    if abs(x_h[2]) < 1e-12:
        raise ValueError("Degenerate projection during reprojection")
    return float(x_h[0] / x_h[2]), float(x_h[1] / x_h[2])


def _triangulate_point(
    P1: np.ndarray,
    P2: np.ndarray,
    u1: float,
    v1: float,
    u2: float,
    v2: float,
) -> tuple[np.ndarray, float]:
    A = np.array(
        [
            u1 * P1[2] - P1[0],
            v1 * P1[2] - P1[1],
            u2 * P2[2] - P2[0],
            v2 * P2[2] - P2[1],
        ],
        dtype=float,
    )
    _, _, vh = np.linalg.svd(A)
    Xh = vh[-1]
    if abs(Xh[3]) < 1e-12:
        raise ValueError("Degenerate homogeneous solution in triangulation")
    X = Xh[:3] / Xh[3]

    u1_hat, v1_hat = _project_point(P1, X)
    u2_hat, v2_hat = _project_point(P2, X)
    err1 = np.sqrt((u1 - u1_hat) ** 2 + (v1 - v1_hat) ** 2)
    err2 = np.sqrt((u2 - u2_hat) ** 2 + (v2 - v2_hat) ** 2)
    return X, float((err1 + err2) / 2.0)


def _build_review_frames(
    frame_reproj: np.ndarray,
    frame_conf: np.ndarray,
    frame_counts: np.ndarray,
    top_k: int,
) -> list[dict]:
    if frame_reproj.size == 0:
        return []
    score = frame_reproj * (2.0 - frame_conf)
    invalid = ~np.isfinite(score)
    score[invalid] = -np.inf
    ranked = np.argsort(-score)[: max(int(top_k), 0)]
    return [
        {
            "frame_index": int(i),
            "review_score": round(float(score[i]), 6),
            "mean_reprojection_error_px": round(float(frame_reproj[i]), 6),
            "mean_confidence": round(float(frame_conf[i]), 6),
            "triangulated_points": int(frame_counts[i]),
        }
        for i in ranked
        if np.isfinite(score[i])
    ]


def _safe_nanmean(values: np.ndarray, axis: int | None = None) -> np.ndarray | float:
    if values.size == 0:
        if axis is None:
            return float("nan")
        return np.array([], dtype=float)
    with np.errstate(invalid="ignore"):
        return np.nanmean(values, axis=axis)


def _compute_reprojection_errors(
    xyz: np.ndarray,
    P1: np.ndarray,
    P2: np.ndarray,
    u1_obs: np.ndarray,
    v1_obs: np.ndarray,
    u2_obs: np.ndarray,
    v2_obs: np.ndarray,
    conf: np.ndarray,
    min_confidence: float,
) -> np.ndarray:
    n_frames, n_markers, _ = xyz.shape
    reproj = np.full((n_frames, n_markers), np.nan, dtype=float)
    for i in range(n_frames):
        for j in range(n_markers):
            if conf[i, j] < min_confidence:
                continue
            if not np.isfinite(xyz[i, j]).all():
                continue
            if not np.isfinite(u1_obs[i, j]) or not np.isfinite(v1_obs[i, j]):
                continue
            if not np.isfinite(u2_obs[i, j]) or not np.isfinite(v2_obs[i, j]):
                continue
            try:
                u1_hat, v1_hat = _project_point(P1, xyz[i, j])
                u2_hat, v2_hat = _project_point(P2, xyz[i, j])
            except Exception:
                continue
            err1 = np.sqrt((u1_obs[i, j] - u1_hat) ** 2 + (v1_obs[i, j] - v1_hat) ** 2)
            err2 = np.sqrt((u2_obs[i, j] - u2_hat) ** 2 + (v2_obs[i, j] - v2_hat) ** 2)
            reproj[i, j] = float((err1 + err2) / 2.0)
    return reproj


def _optimize_rigid_constraints(
    xyz: np.ndarray,
    xyz_anchor: np.ndarray,
    conf: np.ndarray,
    constraints: list[tuple[str, str, float]],
    marker_to_idx: dict[str, int],
    *,
    iterations: int,
    step_size: float,
    reprojection_anchor_weight: float,
) -> np.ndarray:
    if iterations <= 0 or step_size <= 0.0 or not constraints:
        return xyz
    optimized = xyz.copy()
    anchor_weight = float(np.clip(reprojection_anchor_weight, 0.0, 1.0))
    for _ in range(iterations):
        for i in range(optimized.shape[0]):
            for marker_a, marker_b, expected_distance in constraints:
                if marker_a not in marker_to_idx or marker_b not in marker_to_idx:
                    continue
                ia = marker_to_idx[marker_a]
                ib = marker_to_idx[marker_b]
                pa = optimized[i, ia]
                pb = optimized[i, ib]
                if not np.isfinite(pa).all() or not np.isfinite(pb).all():
                    continue
                diff = pb - pa
                dist = float(np.linalg.norm(diff))
                if dist < 1e-12:
                    continue
                direction = diff / dist
                error = dist - expected_distance
                correction = step_size * error * direction

                conf_a = float(np.nan_to_num(conf[i, ia], nan=0.0))
                conf_b = float(np.nan_to_num(conf[i, ib], nan=0.0))
                move_a = max(1e-6, 1.0 - np.clip(conf_a, 0.0, 1.0))
                move_b = max(1e-6, 1.0 - np.clip(conf_b, 0.0, 1.0))
                norm = move_a + move_b
                wa = move_a / norm
                wb = move_b / norm
                updated_a = pa + (wa * correction)
                updated_b = pb - (wb * correction)

                anchor_a = xyz_anchor[i, ia]
                anchor_b = xyz_anchor[i, ib]
                anchor_gain_a = anchor_weight * np.clip(conf_a, 0.0, 1.0)
                anchor_gain_b = anchor_weight * np.clip(conf_b, 0.0, 1.0)
                if np.isfinite(anchor_a).all():
                    updated_a = ((1.0 - anchor_gain_a) * updated_a) + (anchor_gain_a * anchor_a)
                if np.isfinite(anchor_b).all():
                    updated_b = ((1.0 - anchor_gain_b) * updated_b) + (anchor_gain_b * anchor_b)

                optimized[i, ia] = updated_a
                optimized[i, ib] = updated_b
                optimized[i, ib] = pb - (wb * correction)
    return optimized


def _summarize_rigid_constraints(
    xyz: np.ndarray,
    constraints: list[tuple[str, str, float]],
    marker_to_idx: dict[str, int],
) -> tuple[list[dict], float | None, float | None]:
    summaries: list[dict] = []
    residual_pool: list[np.ndarray] = []
    for marker_a, marker_b, expected_distance in constraints:
        if marker_a not in marker_to_idx or marker_b not in marker_to_idx:
            continue
        ia = marker_to_idx[marker_a]
        ib = marker_to_idx[marker_b]
        d = np.linalg.norm(xyz[:, ia] - xyz[:, ib], axis=1)
        residual = np.abs(d - expected_distance)
        valid = np.isfinite(residual)
        if not valid.any():
            continue
        valid_residual = residual[valid]
        valid_dist = d[valid]
        residual_pool.append(valid_residual)
        summaries.append(
            {
                "marker_a": marker_a,
                "marker_b": marker_b,
                "expected_distance": expected_distance,
                "valid_frames": int(valid.sum()),
                "mean_distance": round(float(np.mean(valid_dist)), 6),
                "mean_abs_residual": round(float(np.mean(valid_residual)), 6),
                "p95_abs_residual": round(float(np.percentile(valid_residual, 95)), 6),
            }
        )

    if not residual_pool:
        return summaries, None, None
    combined = np.concatenate(residual_pool)
    return (
        summaries,
        round(float(np.mean(combined)), 6),
        round(float(np.percentile(combined, 95)), 6),
    )


def triangulate_dlc_predictions(
    cam1data: str | pd.DataFrame,
    cam2data: str | pd.DataFrame,
    trialname: str,
    savepath: str,
    cam1_projection: str,
    cam2_projection: str,
    config: StereoTriangulationConfig | None = None,
    constraints_json: str | None = None,
) -> dict:
    config = config or StereoTriangulationConfig()
    P1 = _load_projection_matrix(cam1_projection)
    P2 = _load_projection_matrix(cam2_projection)
    constraints = _load_rigid_constraints(constraints_json)

    cam1 = _load_dlc_dataframe(cam1data)
    cam2 = _load_dlc_dataframe(cam2data)
    if not isinstance(cam1.columns, pd.MultiIndex) or not isinstance(cam2.columns, pd.MultiIndex):
        raise ValueError("Expected DLC predictions with MultiIndex columns")

    scorer1 = cam1.columns.get_level_values("scorer")[0]
    scorer2 = cam2.columns.get_level_values("scorer")[0]
    bodyparts1 = list(cam1.columns.get_level_values("bodyparts").unique())
    bodyparts2 = list(cam2.columns.get_level_values("bodyparts").unique())
    bodyparts = [bp for bp in bodyparts1 if bp in set(bodyparts2)]
    if not bodyparts:
        raise ValueError("No shared bodyparts found between camera predictions")

    x1_label = _coordinate_label(cam1.columns, "x")
    y1_label = _coordinate_label(cam1.columns, "y")
    l1_label = _coordinate_label(cam1.columns, "likelihood")
    x2_label = _coordinate_label(cam2.columns, "x")
    y2_label = _coordinate_label(cam2.columns, "y")
    l2_label = _coordinate_label(cam2.columns, "likelihood")

    n_frames = min(cam1.shape[0], cam2.shape[0])
    n_markers = len(bodyparts)
    xyz = np.full((n_frames, n_markers, 3), np.nan, dtype=float)
    conf = np.full((n_frames, n_markers), np.nan, dtype=float)
    u1_obs = np.full((n_frames, n_markers), np.nan, dtype=float)
    v1_obs = np.full((n_frames, n_markers), np.nan, dtype=float)
    u2_obs = np.full((n_frames, n_markers), np.nan, dtype=float)
    v2_obs = np.full((n_frames, n_markers), np.nan, dtype=float)

    for j, marker in enumerate(bodyparts):
        u1 = _to_float(cam1[(scorer1, marker, x1_label)])[:n_frames]
        v1 = _to_float(cam1[(scorer1, marker, y1_label)])[:n_frames]
        c1 = _to_float(cam1[(scorer1, marker, l1_label)])[:n_frames]
        u2 = _to_float(cam2[(scorer2, marker, x2_label)])[:n_frames]
        v2 = _to_float(cam2[(scorer2, marker, y2_label)])[:n_frames]
        c2 = _to_float(cam2[(scorer2, marker, l2_label)])[:n_frames]
        u1_obs[:, j] = u1
        v1_obs[:, j] = v1
        u2_obs[:, j] = u2
        v2_obs[:, j] = v2

        pair_conf = np.clip(np.minimum(c1, c2), 0.0, 1.0)
        conf[:, j] = pair_conf
        for i in range(n_frames):
            if pair_conf[i] < config.min_confidence:
                continue
            if not np.isfinite(u1[i]) or not np.isfinite(v1[i]):
                continue
            if not np.isfinite(u2[i]) or not np.isfinite(v2[i]):
                continue
            try:
                point3d, _ = _triangulate_point(P1, P2, u1[i], v1[i], u2[i], v2[i])
            except Exception:
                continue
            xyz[i, j] = point3d

    marker_to_idx = {marker: idx for idx, marker in enumerate(bodyparts)}
    xyz_before = xyz.copy()
    reproj_before = _compute_reprojection_errors(
        xyz_before,
        P1,
        P2,
        u1_obs,
        v1_obs,
        u2_obs,
        v2_obs,
        conf,
        config.min_confidence,
    )
    _, rigid_before_mean, rigid_before_p95 = _summarize_rigid_constraints(
        xyz_before, constraints, marker_to_idx
    )

    xyz = _optimize_rigid_constraints(
        xyz,
        xyz_before,
        conf,
        constraints,
        marker_to_idx,
        iterations=max(int(config.rigid_iterations), 0),
        step_size=max(float(config.rigid_step_size), 0.0),
        reprojection_anchor_weight=float(config.reprojection_anchor_weight),
    )
    reproj = _compute_reprojection_errors(
        xyz,
        P1,
        P2,
        u1_obs,
        v1_obs,
        u2_obs,
        v2_obs,
        conf,
        config.min_confidence,
    )
    rigid_summaries, rigid_after_mean, rigid_after_p95 = _summarize_rigid_constraints(
        xyz, constraints, marker_to_idx
    )

    out_dir = Path(savepath)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{trialname}-Triangulated3DPoints.csv"
    h5_path = out_dir / f"{trialname}-Triangulated3DPoints.h5"
    report_path = out_dir / f"{trialname}-TriangulationReport.json"

    flat_cols: list[str] = []
    flat_arrays: list[np.ndarray] = []
    for j, marker in enumerate(bodyparts):
        flat_cols.extend([f"{marker}_X", f"{marker}_Y", f"{marker}_Z"])
        flat_arrays.extend([xyz[:, j, 0], xyz[:, j, 1], xyz[:, j, 2]])
    if flat_arrays:
        arr = np.column_stack(flat_arrays)
        xyz_df = pd.DataFrame(arr, columns=flat_cols)
    else:
        xyz_df = pd.DataFrame()
    xyz_df.to_hdf(h5_path, key="df_with_missing", mode="w")
    xyz_df.to_csv(csv_path, index=False, na_rep="NaN")

    frame_reproj = _safe_nanmean(reproj, axis=1)
    frame_conf = _safe_nanmean(conf, axis=1)
    frame_counts = np.sum(np.isfinite(reproj), axis=1)
    triangulated_fraction = (
        float(np.isfinite(reproj).sum()) / float(max(n_frames * n_markers, 1))
    )


    report = {
        "trial_name": trialname,
        "frame_count": int(n_frames),
        "bodypart_count": int(n_markers),
        "triangulated_fraction": round(triangulated_fraction, 6),
        "mean_reprojection_error_px": round(float(_safe_nanmean(reproj)), 6)
        if np.isfinite(_safe_nanmean(reproj))
        else None,
        "mean_confidence": round(float(_safe_nanmean(conf)), 6)
        if np.isfinite(_safe_nanmean(conf))
        else None,
        "config": {
            "min_confidence": config.min_confidence,
            "top_k_review_frames": config.top_k_review_frames,
            "rigid_iterations": config.rigid_iterations,
            "rigid_step_size": config.rigid_step_size,
            "reprojection_anchor_weight": config.reprojection_anchor_weight,
        },
        "projection_inputs": {
            "cam1_projection": cam1_projection,
            "cam2_projection": cam2_projection,
        },
        "rigid_constraints": rigid_summaries,
        "rigid_optimization": {
            "enabled": bool(constraints) and config.rigid_iterations > 0,
            "constraint_count": len(constraints),
            "reprojection_anchor_weight": config.reprojection_anchor_weight,
            "mean_abs_residual_before": rigid_before_mean,
            "mean_abs_residual_after": rigid_after_mean,
            "p95_abs_residual_before": rigid_before_p95,
            "p95_abs_residual_after": rigid_after_p95,
            "mean_reprojection_error_before_px": round(float(_safe_nanmean(reproj_before)), 6)
            if np.isfinite(_safe_nanmean(reproj_before))
            else None,
            "mean_reprojection_error_after_px": round(float(_safe_nanmean(reproj)), 6)
            if np.isfinite(_safe_nanmean(reproj))
            else None,
        },
        "review_frames": _build_review_frames(
            frame_reproj.copy(),
            frame_conf.copy(),
            frame_counts,
            config.top_k_review_frames,
        ),
        "outputs": {
            "csv": str(csv_path),
            "h5": str(h5_path),
            "report_json": str(report_path),
        },
    }
    report_path.write_text(json.dumps(report, indent=2))
    return report
