from __future__ import annotations

from .api import (
    PipelineAPI,
    run_add_frames,
    run_analyze_xromm_videos,
    run_dlc_to_xma,
    run_optimize_dlc_predictions,
    run_triangulate_dlc_predictions,
    run_xma_to_dlc,
)
from .augmentation import add_frames
from .conversion import dlc_to_xma
from .ingestion import xma_to_dlc
from .models import (
    CameraNamePatterns,
    OperationRunMetadata,
    OperationSpec,
    PipelinePaths,
    StereoTriangulationConfig,
    TemporalOptimizationConfig,
)
from .optimization import optimize_dlc_predictions
from .prediction import analyze_xromm_videos
from .registry import ModelPromotionRecord, ModelRegistry, ModelVersionRecord
from .service import (
    AutoRetrainDecisionResult,
    HpcStubSchedulerAdapter,
    LocalApiExecutor,
    LocalJobStore,
    LocalPipelineOrchestrator,
    LocalQueueSchedulerAdapter,
    ModelVersionComparisonResult,
    OperationExecutor,
    PipelineJobRecord,
    RetrainTriggerResult,
    SchedulerAdapter,
)
from .server import OrchestratorHttpServer, create_api_server, serve_api
from .triangulation import triangulate_dlc_predictions

__all__ = [
    "add_frames",
    "analyze_xromm_videos",
    "run_add_frames",
    "run_analyze_xromm_videos",
    "run_dlc_to_xma",
    "run_optimize_dlc_predictions",
    "run_triangulate_dlc_predictions",
    "run_xma_to_dlc",
    "PipelineAPI",
    "dlc_to_xma",
    "xma_to_dlc",
    "CameraNamePatterns",
    "OperationRunMetadata",
    "OperationSpec",
    "PipelinePaths",
    "StereoTriangulationConfig",
    "TemporalOptimizationConfig",
    "optimize_dlc_predictions",
    "triangulate_dlc_predictions",
    "ModelVersionRecord",
    "ModelPromotionRecord",
    "ModelRegistry",
    "PipelineJobRecord",
    "RetrainTriggerResult",
    "ModelVersionComparisonResult",
    "AutoRetrainDecisionResult",
    "LocalJobStore",
    "LocalPipelineOrchestrator",
    "OperationExecutor",
    "LocalApiExecutor",
    "SchedulerAdapter",
    "LocalQueueSchedulerAdapter",
    "HpcStubSchedulerAdapter",
    "OrchestratorHttpServer",
    "create_api_server",
    "serve_api",
]
