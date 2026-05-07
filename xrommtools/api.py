from __future__ import annotations

from typing import Callable

from .augmentation import add_frames
from .conversion import dlc_to_xma
from .ingestion import xma_to_dlc
from .metadata import create_run_metadata, finalize_run_metadata, write_run_metadata
from .models import (
    OperationRunMetadata,
    OperationSpec,
    StereoTriangulationConfig,
    TemporalOptimizationConfig,
)
from .optimization import optimize_dlc_predictions
from .prediction import analyze_xromm_videos
from .triangulation import triangulate_dlc_predictions


def _execute_operation(
    *,
    spec: OperationSpec,
    declared_outputs: dict,
    runner: Callable[[], None],
    metadata_out: str | None = None,
) -> OperationRunMetadata:
    record = create_run_metadata(spec, declared_outputs)
    try:
        runner()
    except Exception as exc:
        finalize_run_metadata(record, status="failed", error=str(exc))
        if metadata_out:
            write_run_metadata(metadata_out, record)
        raise
    finalize_run_metadata(record, status="succeeded")
    if metadata_out:
        write_run_metadata(metadata_out, record)
    return record


def _declared_xma_to_dlc_outputs(
    path_config_file: str,
    dataset_name: str,
    scorer: str,
    nnetworks: int,
    path_config_file_cam2,
) -> dict:
    if nnetworks == 2:
        cam1_root = path_config_file[:-12] + "/labeled-data/" + dataset_name + "_cam1"
        cam2_root = path_config_file_cam2[:-12] + "/labeled-data/" + dataset_name + "_cam2"
        return {
            "camera_1": {
                "dataset_dir": cam1_root,
                "csv": cam1_root + f"/CollectedData_{scorer}.csv",
                "h5": cam1_root + f"/CollectedData_{scorer}.h5",
            },
            "camera_2": {
                "dataset_dir": cam2_root,
                "csv": cam2_root + f"/CollectedData_{scorer}.csv",
                "h5": cam2_root + f"/CollectedData_{scorer}.h5",
            },
        }

    root = path_config_file[:-12] + "/labeled-data/" + dataset_name
    return {
        "dataset_dir": root,
        "csv": root + f"/CollectedData_{scorer}.csv",
        "h5": root + f"/CollectedData_{scorer}.h5",
    }


def _declared_dlc_to_xma_outputs(trialname: str, savepath: str) -> dict:
    return {
        "csv": savepath + "/" + trialname + "-Predicted2DPoints.csv",
        "h5": savepath + "/" + trialname + "-Predicted2DPoints.h5",
    }


def _declared_analyze_outputs(path_data_to_analyze: str, iteration: int) -> dict:
    return {
        "trials_root": path_data_to_analyze,
        "iteration_subfolder": f"it{iteration}",
    }


def _declared_add_frames_outputs(
    path_config_file: str, nnetworks: int, path_config_file_cam2: str
) -> dict:
    if nnetworks == 2:
        return {
            "camera_1_labeled_data_root": path_config_file[:-12] + "/labeled-data",
            "camera_2_labeled_data_root": path_config_file_cam2[:-12] + "/labeled-data",
        }
    return {"labeled_data_root": path_config_file[:-12] + "/labeled-data"}


def _declared_optimize_outputs(trialname: str, savepath: str) -> dict:
    optimized_trialname = f"{trialname}-Optimized"
    return {
        "optimized_csv": savepath + "/" + optimized_trialname + "-Predicted2DPoints.csv",
        "optimized_h5": savepath + "/" + optimized_trialname + "-Predicted2DPoints.h5",
        "optimization_report_json": savepath + "/" + trialname + "-OptimizationReport.json",
    }


def _declared_triangulation_outputs(trialname: str, savepath: str) -> dict:
    return {
        "triangulated_csv": savepath + "/" + trialname + "-Triangulated3DPoints.csv",
        "triangulated_h5": savepath + "/" + trialname + "-Triangulated3DPoints.h5",
        "triangulation_report_json": savepath + "/" + trialname + "-TriangulationReport.json",
    }


def run_xma_to_dlc(
    path_config_file,
    data_path,
    dataset_name,
    scorer,
    nframes,
    nnetworks=1,
    path_config_file_cam2=[],
    metadata_out: str | None = None,
) -> OperationRunMetadata:
    spec = OperationSpec(
        name="xma_to_dlc",
        parameters={
            "path_config_file": path_config_file,
            "data_path": data_path,
            "dataset_name": dataset_name,
            "scorer": scorer,
            "nframes": nframes,
            "nnetworks": nnetworks,
            "path_config_file_cam2": path_config_file_cam2,
        },
    )
    declared_outputs = _declared_xma_to_dlc_outputs(
        path_config_file,
        dataset_name,
        scorer,
        nnetworks,
        path_config_file_cam2,
    )
    return _execute_operation(
        spec=spec,
        declared_outputs=declared_outputs,
        runner=lambda: xma_to_dlc(
            path_config_file,
            data_path,
            dataset_name,
            scorer,
            nframes,
            nnetworks=nnetworks,
            path_config_file_cam2=path_config_file_cam2,
        ),
        metadata_out=metadata_out,
    )


def run_triangulate_dlc_predictions(
    cam1data,
    cam2data,
    trialname,
    savepath,
    cam1_projection,
    cam2_projection,
    min_confidence: float = 0.1,
    top_k_review_frames: int = 25,
    rigid_iterations: int = 0,
    rigid_step_size: float = 0.05,
    reprojection_anchor_weight: float = 0.4,
    constraints_json: str | None = None,
    metadata_out: str | None = None,
) -> OperationRunMetadata:
    config = StereoTriangulationConfig(
        min_confidence=min_confidence,
        top_k_review_frames=top_k_review_frames,
        rigid_iterations=rigid_iterations,
        rigid_step_size=rigid_step_size,
        reprojection_anchor_weight=reprojection_anchor_weight,
    )
    spec = OperationSpec(
        name="triangulate_dlc_predictions",
        parameters={
            "cam1data": str(cam1data),
            "cam2data": str(cam2data),
            "trialname": trialname,
            "savepath": savepath,
            "cam1_projection": cam1_projection,
            "cam2_projection": cam2_projection,
            "min_confidence": min_confidence,
            "top_k_review_frames": top_k_review_frames,
            "rigid_iterations": rigid_iterations,
            "rigid_step_size": rigid_step_size,
            "reprojection_anchor_weight": reprojection_anchor_weight,
            "constraints_json": constraints_json,
        },
    )
    declared_outputs = _declared_triangulation_outputs(trialname, savepath)
    return _execute_operation(
        spec=spec,
        declared_outputs=declared_outputs,
        runner=lambda: triangulate_dlc_predictions(
            cam1data,
            cam2data,
            trialname,
            savepath,
            cam1_projection,
            cam2_projection,
            config=config,
            constraints_json=constraints_json,
        ),
        metadata_out=metadata_out,
    )


def run_optimize_dlc_predictions(
    cam1data,
    cam2data,
    trialname,
    savepath,
    transition_weight: float = 0.02,
    process_noise: float = 0.01,
    top_k_review_frames: int = 25,
    metadata_out: str | None = None,
) -> OperationRunMetadata:
    config = TemporalOptimizationConfig(
        transition_weight=transition_weight,
        process_noise=process_noise,
        top_k_review_frames=top_k_review_frames,
    )
    spec = OperationSpec(
        name="optimize_dlc_predictions",
        parameters={
            "cam1data": str(cam1data),
            "cam2data": str(cam2data),
            "trialname": trialname,
            "savepath": savepath,
            "transition_weight": transition_weight,
            "process_noise": process_noise,
            "top_k_review_frames": top_k_review_frames,
        },
    )
    declared_outputs = _declared_optimize_outputs(trialname, savepath)
    return _execute_operation(
        spec=spec,
        declared_outputs=declared_outputs,
        runner=lambda: optimize_dlc_predictions(
            cam1data,
            cam2data,
            trialname,
            savepath,
            config=config,
        ),
        metadata_out=metadata_out,
    )


def run_dlc_to_xma(
    cam1data,
    cam2data,
    trialname,
    savepath,
    metadata_out: str | None = None,
) -> OperationRunMetadata:
    spec = OperationSpec(
        name="dlc_to_xma",
        parameters={
            "cam1data": str(cam1data),
            "cam2data": str(cam2data),
            "trialname": trialname,
            "savepath": savepath,
        },
    )
    declared_outputs = _declared_dlc_to_xma_outputs(trialname, savepath)
    return _execute_operation(
        spec=spec,
        declared_outputs=declared_outputs,
        runner=lambda: dlc_to_xma(cam1data, cam2data, trialname, savepath),
        metadata_out=metadata_out,
    )


def run_analyze_xromm_videos(
    path_config_file,
    path_data_to_analyze,
    iteration,
    nnetworks=1,
    path_config_file_cam2=[],
    metadata_out: str | None = None,
) -> OperationRunMetadata:
    spec = OperationSpec(
        name="analyze_xromm_videos",
        parameters={
            "path_config_file": path_config_file,
            "path_data_to_analyze": path_data_to_analyze,
            "iteration": iteration,
            "nnetworks": nnetworks,
            "path_config_file_cam2": path_config_file_cam2,
        },
    )
    declared_outputs = _declared_analyze_outputs(path_data_to_analyze, iteration)
    return _execute_operation(
        spec=spec,
        declared_outputs=declared_outputs,
        runner=lambda: analyze_xromm_videos(
            path_config_file,
            path_data_to_analyze,
            iteration,
            nnetworks=nnetworks,
            path_config_file_cam2=path_config_file_cam2,
        ),
        metadata_out=metadata_out,
    )


def run_add_frames(
    path_config_file,
    data_path,
    iteration,
    frames,
    nnetworks=1,
    path_config_file_cam2="enterpathofcam2config",
    metadata_out: str | None = None,
) -> OperationRunMetadata:
    spec = OperationSpec(
        name="add_frames",
        parameters={
            "path_config_file": path_config_file,
            "data_path": data_path,
            "iteration": iteration,
            "frames": frames,
            "nnetworks": nnetworks,
            "path_config_file_cam2": path_config_file_cam2,
        },
    )
    declared_outputs = _declared_add_frames_outputs(
        path_config_file, nnetworks, path_config_file_cam2
    )
    return _execute_operation(
        spec=spec,
        declared_outputs=declared_outputs,
        runner=lambda: add_frames(
            path_config_file,
            data_path,
            iteration,
            frames,
            nnetworks=nnetworks,
            path_config_file_cam2=path_config_file_cam2,
        ),
        metadata_out=metadata_out,
    )


class PipelineAPI:
    def xma_to_dlc(self, *args, **kwargs) -> OperationRunMetadata:
        return run_xma_to_dlc(*args, **kwargs)

    def dlc_to_xma(self, *args, **kwargs) -> OperationRunMetadata:
        return run_dlc_to_xma(*args, **kwargs)

    def analyze_xromm_videos(self, *args, **kwargs) -> OperationRunMetadata:
        return run_analyze_xromm_videos(*args, **kwargs)

    def add_frames(self, *args, **kwargs) -> OperationRunMetadata:
        return run_add_frames(*args, **kwargs)

    def optimize_dlc_predictions(self, *args, **kwargs) -> OperationRunMetadata:
        return run_optimize_dlc_predictions(*args, **kwargs)

    def triangulate_dlc_predictions(self, *args, **kwargs) -> OperationRunMetadata:
        return run_triangulate_dlc_predictions(*args, **kwargs)
