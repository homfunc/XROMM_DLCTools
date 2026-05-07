from __future__ import annotations

import argparse
from pathlib import Path

from .api import (
    run_add_frames,
    run_analyze_xromm_videos,
    run_dlc_to_xma,
    run_optimize_dlc_predictions,
    run_triangulate_dlc_predictions,
    run_xma_to_dlc,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="XROMM Tools CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    p_ingest = sub.add_parser("xma-to-dlc", help="Convert XMALab training data into DLC format")
    p_ingest.add_argument("--config", required=True)
    p_ingest.add_argument("--data-path", required=True)
    p_ingest.add_argument("--dataset-name", required=True)
    p_ingest.add_argument("--scorer", required=True)
    p_ingest.add_argument("--nframes", type=int, required=True)
    p_ingest.add_argument("--nnetworks", type=int, default=1, choices=[1, 2])
    p_ingest.add_argument("--config-cam2", default=[])
    p_ingest.add_argument("--metadata-out", default=None)

    p_convert = sub.add_parser("dlc-to-xma", help="Convert DLC output into XMALab format")
    p_convert.add_argument("--cam1-data", required=True)
    p_convert.add_argument("--cam2-data", required=True)
    p_convert.add_argument("--trial-name", required=True)
    p_convert.add_argument("--save-path", required=True)
    p_convert.add_argument("--metadata-out", default=None)

    p_predict = sub.add_parser(
        "analyze-xromm-videos", help="Run DLC prediction and convert to XMALab format"
    )
    p_predict.add_argument("--config", required=True)
    p_predict.add_argument("--data-path", required=True)
    p_predict.add_argument("--iteration", required=True, type=int)
    p_predict.add_argument("--nnetworks", type=int, default=1, choices=[1, 2])
    p_predict.add_argument("--config-cam2", default=[])
    p_predict.add_argument("--metadata-out", default=None)

    p_add = sub.add_parser("add-frames", help="Add corrected frames into existing DLC dataset")
    p_add.add_argument("--config", required=True)
    p_add.add_argument("--data-path", required=True)
    p_add.add_argument("--iteration", required=True, type=int)
    p_add.add_argument("--frames-csv", required=True)
    p_add.add_argument("--nnetworks", type=int, default=1, choices=[1, 2])
    p_add.add_argument("--config-cam2", default="enterpathofcam2config")
    p_add.add_argument("--metadata-out", default=None)

    p_opt = sub.add_parser(
        "optimize-dlc-predictions",
        help="Run temporal optimization on DLC predictions and emit review-frame report",
    )
    p_opt.add_argument("--cam1-data", required=True)
    p_opt.add_argument("--cam2-data", required=True)
    p_opt.add_argument("--trial-name", required=True)
    p_opt.add_argument("--save-path", required=True)
    p_opt.add_argument("--transition-weight", type=float, default=0.02)
    p_opt.add_argument("--process-noise", type=float, default=0.01)
    p_opt.add_argument("--top-k-review-frames", type=int, default=25)
    p_opt.add_argument("--metadata-out", default=None)

    p_tri = sub.add_parser(
        "triangulate-dlc-predictions",
        help="Triangulate stereo DLC predictions into 3D points and score review frames",
    )
    p_tri.add_argument("--cam1-data", required=True)
    p_tri.add_argument("--cam2-data", required=True)
    p_tri.add_argument("--trial-name", required=True)
    p_tri.add_argument("--save-path", required=True)
    p_tri.add_argument("--cam1-projection", required=True)
    p_tri.add_argument("--cam2-projection", required=True)
    p_tri.add_argument("--min-confidence", type=float, default=0.1)
    p_tri.add_argument("--constraints-json", default=None)
    p_tri.add_argument("--top-k-review-frames", type=int, default=25)
    p_tri.add_argument("--rigid-iterations", type=int, default=0)
    p_tri.add_argument("--rigid-step-size", type=float, default=0.05)
    p_tri.add_argument("--reprojection-anchor-weight", type=float, default=0.4)
    p_tri.add_argument("--metadata-out", default=None)

    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    if args.command == "xma-to-dlc":
        run_xma_to_dlc(
            args.config,
            args.data_path,
            args.dataset_name,
            args.scorer,
            args.nframes,
            nnetworks=args.nnetworks,
            path_config_file_cam2=args.config_cam2,
            metadata_out=args.metadata_out,
        )
        return 0

    if args.command == "dlc-to-xma":
        cam1 = args.cam1_data
        cam2 = args.cam2_data
        if Path(cam1).suffix not in {".csv", ".h5"}:
            raise ValueError("cam1-data must be path to .csv or .h5 file")
        if Path(cam2).suffix not in {".csv", ".h5"}:
            raise ValueError("cam2-data must be path to .csv or .h5 file")
        run_dlc_to_xma(
            cam1,
            cam2,
            args.trial_name,
            args.save_path,
            metadata_out=args.metadata_out,
        )
        return 0

    if args.command == "analyze-xromm-videos":
        run_analyze_xromm_videos(
            args.config,
            args.data_path,
            args.iteration,
            nnetworks=args.nnetworks,
            path_config_file_cam2=args.config_cam2,
            metadata_out=args.metadata_out,
        )
        return 0

    if args.command == "add-frames":
        run_add_frames(
            args.config,
            args.data_path,
            args.iteration,
            args.frames_csv,
            nnetworks=args.nnetworks,
            path_config_file_cam2=args.config_cam2,
            metadata_out=args.metadata_out,
        )
        return 0

    if args.command == "optimize-dlc-predictions":
        cam1 = args.cam1_data
        cam2 = args.cam2_data
        if Path(cam1).suffix not in {".csv", ".h5"}:
            raise ValueError("cam1-data must be path to .csv or .h5 file")
        if Path(cam2).suffix not in {".csv", ".h5"}:
            raise ValueError("cam2-data must be path to .csv or .h5 file")
        run_optimize_dlc_predictions(
            cam1,
            cam2,
            args.trial_name,
            args.save_path,
            transition_weight=args.transition_weight,
            process_noise=args.process_noise,
            top_k_review_frames=args.top_k_review_frames,
            metadata_out=args.metadata_out,
        )
        return 0

    if args.command == "triangulate-dlc-predictions":
        cam1 = args.cam1_data
        cam2 = args.cam2_data
        if Path(cam1).suffix not in {".csv", ".h5"}:
            raise ValueError("cam1-data must be path to .csv or .h5 file")
        if Path(cam2).suffix not in {".csv", ".h5"}:
            raise ValueError("cam2-data must be path to .csv or .h5 file")
        run_triangulate_dlc_predictions(
            cam1,
            cam2,
            args.trial_name,
            args.save_path,
            args.cam1_projection,
            args.cam2_projection,
            min_confidence=args.min_confidence,
            top_k_review_frames=args.top_k_review_frames,
            rigid_iterations=args.rigid_iterations,
            rigid_step_size=args.rigid_step_size,
            reprojection_anchor_weight=args.reprojection_anchor_weight,
            constraints_json=args.constraints_json,
            metadata_out=args.metadata_out,
        )
        return 0

    raise RuntimeError(f"Unknown command: {args.command}")
