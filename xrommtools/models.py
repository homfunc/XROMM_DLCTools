from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class CameraNamePatterns:
    cam1: tuple[str, ...]
    cam2: tuple[str, ...]

    def as_nested_list(self) -> list[list[str]]:
        return [list(self.cam1), list(self.cam2)]


DEFAULT_CAMERA_PATTERNS = CameraNamePatterns(
    cam1=(
        "c01",
        "c1",
        "C01",
        "C1",
        "Cam1",
        "cam1",
        "Cam01",
        "cam01",
        "Camera1",
        "camera1",
    ),
    cam2=(
        "c02",
        "c2",
        "C02",
        "C2",
        "Cam2",
        "cam2",
        "Cam02",
        "cam02",
        "Camera2",
        "camera2",
    ),
)


@dataclass(frozen=True)
class PipelinePaths:
    config_path: str
    data_path: str


@dataclass(frozen=True)
class OperationSpec:
    name: str
    parameters: dict[str, Any]


@dataclass
class OperationRunMetadata:
    run_id: str
    operation: str
    status: str
    started_at_utc: str
    finished_at_utc: str | None
    duration_seconds: float | None
    parameters: dict[str, Any]
    declared_outputs: dict[str, Any]
    error: str | None = None


@dataclass(frozen=True)
class TemporalOptimizationConfig:
    transition_weight: float = 0.02
    process_noise: float = 0.01
    top_k_review_frames: int = 25


@dataclass(frozen=True)
class StereoTriangulationConfig:
    min_confidence: float = 0.1
    top_k_review_frames: int = 25
    rigid_iterations: int = 0
    rigid_step_size: float = 0.05
    reprojection_anchor_weight: float = 0.4
