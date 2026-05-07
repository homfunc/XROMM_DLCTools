from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from .api import (
    run_add_frames,
    run_analyze_xromm_videos,
    run_dlc_to_xma,
    run_optimize_dlc_predictions,
    run_triangulate_dlc_predictions,
    run_xma_to_dlc,
)
from .registry import ModelRegistry
from .service import LocalJobStore, LocalPipelineOrchestrator
from .server import serve_api


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

    p_submit = sub.add_parser(
        "submit-job",
        help="Submit a local pipeline job with persisted run state",
    )
    p_submit.add_argument("--operation", required=True)
    p_submit.add_argument("--params-json", required=True)
    p_submit.add_argument("--jobs-root", default=".xrommtools/jobs")
    p_submit.add_argument("--scheduler", default="local")
    p_submit.add_argument("--max-retries", type=int, default=0)
    p_submit.add_argument("--defer-run", action="store_true")
    p_submit.add_argument("--metadata-out", default=None)

    p_run = sub.add_parser("run-job", help="Execute a previously submitted job")
    p_run.add_argument("--job-id", required=True)
    p_run.add_argument("--jobs-root", default=".xrommtools/jobs")

    p_retry = sub.add_parser("retry-job", help="Retry a failed or canceled job")
    p_retry.add_argument("--job-id", required=True)
    p_retry.add_argument("--jobs-root", default=".xrommtools/jobs")
    p_retry.add_argument("--extra-retries", type=int, default=1)

    p_cancel = sub.add_parser("cancel-job", help="Request cancellation for a pending/running job")
    p_cancel.add_argument("--job-id", required=True)
    p_cancel.add_argument("--jobs-root", default=".xrommtools/jobs")

    p_queue = sub.add_parser(
        "run-queue",
        help="Execute pending/retrying jobs for a scheduler queue",
    )
    p_queue.add_argument("--jobs-root", default=".xrommtools/jobs")
    p_queue.add_argument("--scheduler", default="local")
    p_queue.add_argument("--max-jobs", type=int, default=20)

    p_status = sub.add_parser("job-status", help="Show persisted state for one pipeline job")
    p_status.add_argument("--job-id", required=True)
    p_status.add_argument("--jobs-root", default=".xrommtools/jobs")

    p_list = sub.add_parser("list-jobs", help="List recent persisted pipeline jobs")
    p_list.add_argument("--jobs-root", default=".xrommtools/jobs")
    p_list.add_argument("--limit", type=int, default=20)

    p_register_model = sub.add_parser(
        "register-model-version",
        help="Register a model artifact/version in the local model registry",
    )
    p_register_model.add_argument("--model-name", required=True)
    p_register_model.add_argument("--artifact-path", required=True)
    p_register_model.add_argument("--version", default=None)
    p_register_model.add_argument("--metrics-json", default=None)
    p_register_model.add_argument("--metadata-json", default=None)
    p_register_model.add_argument("--registry-path", default=".xrommtools/model_registry.json")

    p_list_models = sub.add_parser(
        "list-model-versions",
        help="List model versions from the local model registry",
    )
    p_list_models.add_argument("--model-name", default=None)
    p_list_models.add_argument("--registry-path", default=".xrommtools/model_registry.json")

    p_promote_model = sub.add_parser(
        "promote-model-version",
        help="Promote a model version to a stage alias",
    )
    p_promote_model.add_argument("--model-name", required=True)
    p_promote_model.add_argument("--version", required=True)
    p_promote_model.add_argument("--stage", default="production")
    p_promote_model.add_argument("--notes", default=None)
    p_promote_model.add_argument("--registry-path", default=".xrommtools/model_registry.json")

    p_active_model = sub.add_parser(
        "active-model-version",
        help="Show active model version for a stage alias",
    )
    p_active_model.add_argument("--model-name", required=True)
    p_active_model.add_argument("--stage", default="production")
    p_active_model.add_argument("--registry-path", default=".xrommtools/model_registry.json")

    p_compare_models = sub.add_parser(
        "compare-model-versions",
        help="Compare candidate model metrics against a baseline version or active stage alias",
    )
    p_compare_models.add_argument("--model-name", required=True)
    p_compare_models.add_argument("--candidate-version", required=True)
    p_compare_models.add_argument("--baseline-version", default=None)
    p_compare_models.add_argument("--baseline-stage", default="production")
    p_compare_models.add_argument("--higher-is-better-metrics", default=None)
    p_compare_models.add_argument("--registry-path", default=".xrommtools/model_registry.json")

    p_trigger_retrain = sub.add_parser(
        "trigger-retrain",
        help="Conditionally submit a retraining job based on corrected-frame thresholds",
    )
    p_trigger_retrain.add_argument("--operation", required=True)
    p_trigger_retrain.add_argument("--params-json", required=True)
    p_trigger_retrain.add_argument("--corrected-frames-count", type=int, required=True)
    p_trigger_retrain.add_argument("--corrected-frames-threshold", type=int, default=25)
    p_trigger_retrain.add_argument("--force", action="store_true")
    p_trigger_retrain.add_argument("--jobs-root", default=".xrommtools/jobs")
    p_trigger_retrain.add_argument("--scheduler", default="local")
    p_trigger_retrain.add_argument("--max-retries", type=int, default=0)
    p_trigger_retrain.add_argument("--defer-run", action="store_true")
    p_trigger_retrain.add_argument("--metadata-out", default=None)
    p_trigger_retrain.add_argument("--registry-path", default=".xrommtools/model_registry.json")

    p_auto_trigger_retrain = sub.add_parser(
        "auto-trigger-retrain",
        help="Auto trigger retraining from corrected-frame and model-degradation signals",
    )
    p_auto_trigger_retrain.add_argument("--model-name", required=True)
    p_auto_trigger_retrain.add_argument("--candidate-version", required=True)
    p_auto_trigger_retrain.add_argument("--operation", required=True)
    p_auto_trigger_retrain.add_argument("--params-json", required=True)
    p_auto_trigger_retrain.add_argument("--corrected-frames-count", type=int, required=True)
    p_auto_trigger_retrain.add_argument("--corrected-frames-threshold", type=int, default=25)
    p_auto_trigger_retrain.add_argument("--degradation-thresholds-json", default=None)
    p_auto_trigger_retrain.add_argument("--higher-is-better-metrics", default=None)
    p_auto_trigger_retrain.add_argument("--baseline-version", default=None)
    p_auto_trigger_retrain.add_argument("--baseline-stage", default="production")
    p_auto_trigger_retrain.add_argument("--force", action="store_true")
    p_auto_trigger_retrain.add_argument("--jobs-root", default=".xrommtools/jobs")
    p_auto_trigger_retrain.add_argument("--scheduler", default="local")
    p_auto_trigger_retrain.add_argument("--max-retries", type=int, default=0)
    p_auto_trigger_retrain.add_argument("--defer-run", action="store_true")
    p_auto_trigger_retrain.add_argument("--metadata-out", default=None)
    p_auto_trigger_retrain.add_argument("--registry-path", default=".xrommtools/model_registry.json")

    p_serve_api = sub.add_parser(
        "serve-api",
        help="Run a lightweight local JSON API over orchestration primitives",
    )
    p_serve_api.add_argument("--host", default="127.0.0.1")
    p_serve_api.add_argument("--port", type=int, default=8765)
    p_serve_api.add_argument("--jobs-root", default=".xrommtools/jobs")
    p_serve_api.add_argument("--registry-path", default=".xrommtools/model_registry.json")

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

    if args.command == "submit-job":
        params = json.loads(Path(args.params_json).read_text())
        if not isinstance(params, dict):
            raise ValueError("params-json must contain a JSON object of operation kwargs")
        orchestrator = LocalPipelineOrchestrator(LocalJobStore(args.jobs_root))
        record = orchestrator.submit(
            args.operation,
            params,
            metadata_out=args.metadata_out,
            max_retries=args.max_retries,
            run_immediately=not args.defer_run,
            scheduler=args.scheduler,
        )
        print(json.dumps(asdict(record), indent=2))
        return 0 if record.status in {"pending", "queued_remote", "succeeded"} else 1

    if args.command == "run-job":
        orchestrator = LocalPipelineOrchestrator(LocalJobStore(args.jobs_root))
        record = orchestrator.execute_job(args.job_id)
        print(json.dumps(asdict(record), indent=2))
        return 0 if record.status == "succeeded" else 1

    if args.command == "retry-job":
        orchestrator = LocalPipelineOrchestrator(LocalJobStore(args.jobs_root))
        record = orchestrator.retry_job(args.job_id, extra_retries=args.extra_retries)
        print(json.dumps(asdict(record), indent=2))
        return 0 if record.status == "succeeded" else 1

    if args.command == "cancel-job":
        orchestrator = LocalPipelineOrchestrator(LocalJobStore(args.jobs_root))
        record = orchestrator.request_cancel(args.job_id)
        print(json.dumps(asdict(record), indent=2))
        return 0

    if args.command == "run-queue":
        orchestrator = LocalPipelineOrchestrator(LocalJobStore(args.jobs_root))
        records = orchestrator.run_queue(
            scheduler=args.scheduler,
            max_jobs=args.max_jobs,
        )
        print(json.dumps([asdict(r) for r in records], indent=2))
        if any(r.status == "failed" for r in records):
            return 1
        return 0

    if args.command == "job-status":
        store = LocalJobStore(args.jobs_root)
        record = store.load(args.job_id)
        print(json.dumps(asdict(record), indent=2))
        return 0

    if args.command == "list-jobs":
        store = LocalJobStore(args.jobs_root)
        records = store.list_jobs(limit=args.limit)
        print(json.dumps([asdict(r) for r in records], indent=2))
        return 0

    if args.command == "register-model-version":
        metrics = {}
        if args.metrics_json:
            metrics = json.loads(Path(args.metrics_json).read_text())
            if not isinstance(metrics, dict):
                raise ValueError("metrics-json must contain a JSON object")
        metadata = {}
        if args.metadata_json:
            metadata = json.loads(Path(args.metadata_json).read_text())
            if not isinstance(metadata, dict):
                raise ValueError("metadata-json must contain a JSON object")

        orchestrator = LocalPipelineOrchestrator(
            model_registry=ModelRegistry(args.registry_path)
        )
        record = orchestrator.register_model_version(
            args.model_name,
            args.artifact_path,
            version=args.version,
            metrics=metrics,
            metadata=metadata,
        )
        print(json.dumps(asdict(record), indent=2))
        return 0

    if args.command == "list-model-versions":
        orchestrator = LocalPipelineOrchestrator(
            model_registry=ModelRegistry(args.registry_path)
        )
        records = orchestrator.list_model_versions(model_name=args.model_name)
        print(json.dumps([asdict(r) for r in records], indent=2))
        return 0

    if args.command == "promote-model-version":
        orchestrator = LocalPipelineOrchestrator(
            model_registry=ModelRegistry(args.registry_path)
        )
        promotion = orchestrator.promote_model_version(
            args.model_name,
            args.version,
            stage=args.stage,
            notes=args.notes,
        )
        print(json.dumps(asdict(promotion), indent=2))
        return 0

    if args.command == "active-model-version":
        orchestrator = LocalPipelineOrchestrator(
            model_registry=ModelRegistry(args.registry_path)
        )
        active = orchestrator.get_active_model_version(args.model_name, stage=args.stage)
        if active is None:
            print("null")
        else:
            print(json.dumps(asdict(active), indent=2))
        return 0

    if args.command == "compare-model-versions":
        higher_is_better_metrics = [
            item.strip()
            for item in (args.higher_is_better_metrics or "").split(",")
            if item.strip()
        ]
        orchestrator = LocalPipelineOrchestrator(
            model_registry=ModelRegistry(args.registry_path)
        )
        comparison = orchestrator.compare_model_versions(
            args.model_name,
            candidate_version=args.candidate_version,
            baseline_version=args.baseline_version,
            baseline_stage=args.baseline_stage,
            higher_is_better_metrics=higher_is_better_metrics,
        )
        print(json.dumps(asdict(comparison), indent=2))
        return 0

    if args.command == "trigger-retrain":
        params = json.loads(Path(args.params_json).read_text())
        if not isinstance(params, dict):
            raise ValueError("params-json must contain a JSON object of operation kwargs")
        orchestrator = LocalPipelineOrchestrator(
            store=LocalJobStore(args.jobs_root),
            model_registry=ModelRegistry(args.registry_path),
        )
        result = orchestrator.trigger_retraining(
            operation=args.operation,
            parameters=params,
            corrected_frames_count=args.corrected_frames_count,
            corrected_frames_threshold=args.corrected_frames_threshold,
            force=args.force,
            metadata_out=args.metadata_out,
            max_retries=args.max_retries,
            run_immediately=not args.defer_run,
            scheduler=args.scheduler,
        )
        print(json.dumps(asdict(result), indent=2))
        if result.triggered and result.job is not None and result.job.status == "failed":
            return 1
        return 0

    if args.command == "auto-trigger-retrain":
        params = json.loads(Path(args.params_json).read_text())
        if not isinstance(params, dict):
            raise ValueError("params-json must contain a JSON object of operation kwargs")
        degradation_thresholds: dict[str, float] = {}
        if args.degradation_thresholds_json:
            degradation_thresholds = json.loads(Path(args.degradation_thresholds_json).read_text())
            if not isinstance(degradation_thresholds, dict):
                raise ValueError("degradation-thresholds-json must contain a JSON object")
        higher_is_better_metrics = [
            item.strip()
            for item in (args.higher_is_better_metrics or "").split(",")
            if item.strip()
        ]
        orchestrator = LocalPipelineOrchestrator(
            store=LocalJobStore(args.jobs_root),
            model_registry=ModelRegistry(args.registry_path),
        )
        result = orchestrator.auto_trigger_retraining(
            model_name=args.model_name,
            candidate_version=args.candidate_version,
            operation=args.operation,
            parameters=params,
            corrected_frames_count=args.corrected_frames_count,
            corrected_frames_threshold=args.corrected_frames_threshold,
            degradation_thresholds=degradation_thresholds,
            higher_is_better_metrics=higher_is_better_metrics,
            baseline_version=args.baseline_version,
            baseline_stage=args.baseline_stage,
            force=args.force,
            metadata_out=args.metadata_out,
            max_retries=args.max_retries,
            run_immediately=not args.defer_run,
            scheduler=args.scheduler,
        )
        print(json.dumps(asdict(result), indent=2))
        if result.triggered and result.job is not None and result.job.status == "failed":
            return 1
        return 0

    if args.command == "serve-api":
        print(
            f"Serving xrommtools API on http://{args.host}:{args.port} "
            f"(UI: http://{args.host}:{args.port}/, Ctrl+C to stop)."
        )
        try:
            serve_api(
                host=args.host,
                port=args.port,
                jobs_root=args.jobs_root,
                registry_path=args.registry_path,
            )
        except KeyboardInterrupt:
            return 0
        return 0

    raise RuntimeError(f"Unknown command: {args.command}")
