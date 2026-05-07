from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Protocol
from uuid import uuid4

from .api import (
    run_add_frames,
    run_analyze_xromm_videos,
    run_dlc_to_xma,
    run_optimize_dlc_predictions,
    run_triangulate_dlc_predictions,
    run_xma_to_dlc,
)
from .metadata import utc_now_iso
from .registry import ModelRegistry, ModelVersionRecord


@dataclass
class PipelineJobRecord:
    job_id: str = ""
    operation: str = ""
    status: str = "pending"
    submitted_at_utc: str = ""
    started_at_utc: str | None = None
    finished_at_utc: str | None = None
    parameters: dict[str, Any] = field(default_factory=dict)
    metadata_out: str | None = None
    operation_run: dict[str, Any] | None = None
    error: str | None = None
    attempt_count: int = 0
    max_retries: int = 0
    cancel_requested: bool = False
    last_updated_at_utc: str = ""
    scheduler: str = "local"
    scheduler_job_id: str | None = None


@dataclass
class RetrainTriggerResult:
    triggered: bool
    reason: str
    job: PipelineJobRecord | None = None


@dataclass
class ModelVersionComparisonResult:
    model_name: str
    candidate_version: str
    baseline_version: str
    candidate_metrics: dict[str, float]
    baseline_metrics: dict[str, float]
    metric_deltas: dict[str, float]
    degraded_metrics: dict[str, float]
    improved_metrics: dict[str, float]
    missing_metrics: list[str]


@dataclass
class AutoRetrainDecisionResult:
    triggered: bool
    reason: str
    comparison: ModelVersionComparisonResult | None = None
    degradation_exceeds: dict[str, float] = field(default_factory=dict)
    job: PipelineJobRecord | None = None


def _as_float_metric(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _split_metric_names(value: str | None) -> set[str]:
    if value is None:
        return set()
    return {item.strip() for item in value.split(",") if item.strip()}


class LocalJobStore:
    def __init__(self, jobs_root: str | Path = ".xrommtools/jobs"):
        self.jobs_root = Path(jobs_root)
        self.jobs_root.mkdir(parents=True, exist_ok=True)

    def _job_path(self, job_id: str) -> Path:
        return self.jobs_root / f"{job_id}.json"

    def save(self, record: PipelineJobRecord) -> Path:
        record.last_updated_at_utc = utc_now_iso()
        path = self._job_path(record.job_id)
        path.write_text(json.dumps(asdict(record), indent=2))
        return path

    def load(self, job_id: str) -> PipelineJobRecord:
        path = self._job_path(job_id)
        payload = json.loads(path.read_text())
        return PipelineJobRecord(**payload)

    def list_jobs(self, limit: int | None = None) -> list[PipelineJobRecord]:
        files = sorted(
            self.jobs_root.glob("*.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if limit is not None:
            files = files[: max(int(limit), 0)]
        out: list[PipelineJobRecord] = []
        for path in files:
            payload = json.loads(path.read_text())
            out.append(PipelineJobRecord(**payload))
        return out


class OperationExecutor(Protocol):
    def supported_operations(self) -> list[str]: ...

    def execute(
        self,
        operation: str,
        parameters: dict[str, Any],
        *,
        metadata_out: str | None = None,
    ) -> Any: ...


class LocalApiExecutor:
    def __init__(self):
        self._dispatch: dict[str, Callable[..., Any]] = {
            "xma_to_dlc": run_xma_to_dlc,
            "dlc_to_xma": run_dlc_to_xma,
            "analyze_xromm_videos": run_analyze_xromm_videos,
            "add_frames": run_add_frames,
            "optimize_dlc_predictions": run_optimize_dlc_predictions,
            "triangulate_dlc_predictions": run_triangulate_dlc_predictions,
        }

    def supported_operations(self) -> list[str]:
        return sorted(self._dispatch.keys())

    def execute(
        self,
        operation: str,
        parameters: dict[str, Any],
        *,
        metadata_out: str | None = None,
    ) -> Any:
        if operation not in self._dispatch:
            supported = ", ".join(self.supported_operations())
            raise ValueError(f"Unsupported operation {operation!r}. Supported: {supported}")
        runner = self._dispatch[operation]
        kwargs = dict(parameters)
        if metadata_out is not None and "metadata_out" not in kwargs:
            kwargs["metadata_out"] = metadata_out
        return runner(**kwargs)


class SchedulerAdapter(Protocol):
    def name(self) -> str: ...

    def enqueue(self, record: PipelineJobRecord, store: LocalJobStore) -> PipelineJobRecord: ...

    def drain(
        self,
        orchestrator: "LocalPipelineOrchestrator",
        *,
        max_jobs: int | None = None,
    ) -> list[PipelineJobRecord]: ...

    def cancel(self, record: PipelineJobRecord, store: LocalJobStore) -> PipelineJobRecord: ...


class LocalQueueSchedulerAdapter:
    def name(self) -> str:
        return "local"

    def enqueue(self, record: PipelineJobRecord, store: LocalJobStore) -> PipelineJobRecord:
        if record.status not in {"pending", "retrying"}:
            record.status = "pending"
        store.save(record)
        return record

    def drain(
        self,
        orchestrator: "LocalPipelineOrchestrator",
        *,
        max_jobs: int | None = None,
    ) -> list[PipelineJobRecord]:
        queue = [
            r
            for r in orchestrator.list_jobs(limit=None)
            if r.scheduler == self.name()
            and r.status in {"pending", "retrying"}
            and not r.cancel_requested
        ]
        queue.sort(key=lambda r: r.submitted_at_utc)
        if max_jobs is not None:
            queue = queue[: max(int(max_jobs), 0)]
        out: list[PipelineJobRecord] = []
        for record in queue:
            out.append(orchestrator.execute_job(record.job_id))
        return out

    def cancel(self, record: PipelineJobRecord, store: LocalJobStore) -> PipelineJobRecord:
        record.cancel_requested = True
        if record.status in {"pending", "retrying"}:
            record.status = "canceled"
            record.finished_at_utc = utc_now_iso()
        store.save(record)
        return record


class HpcStubSchedulerAdapter:
    def name(self) -> str:
        return "hpc-stub"

    def enqueue(self, record: PipelineJobRecord, store: LocalJobStore) -> PipelineJobRecord:
        record.status = "queued_remote"
        record.scheduler_job_id = f"hpcstub-{uuid4().hex[:12]}"
        store.save(record)
        return record

    def drain(
        self,
        orchestrator: "LocalPipelineOrchestrator",
        *,
        max_jobs: int | None = None,
    ) -> list[PipelineJobRecord]:
        return []

    def cancel(self, record: PipelineJobRecord, store: LocalJobStore) -> PipelineJobRecord:
        record.cancel_requested = True
        if record.status in {"queued_remote", "pending", "retrying"}:
            record.status = "canceled"
            record.finished_at_utc = utc_now_iso()
        store.save(record)
        return record


class LocalPipelineOrchestrator:
    def __init__(
        self,
        store: LocalJobStore | None = None,
        executor: OperationExecutor | None = None,
        scheduler_adapters: dict[str, SchedulerAdapter] | None = None,
        model_registry: ModelRegistry | None = None,
    ):
        self.store = store or LocalJobStore()
        self.executor = executor or LocalApiExecutor()
        self.model_registry = model_registry or ModelRegistry()
        self.scheduler_adapters = scheduler_adapters or {
            "local": LocalQueueSchedulerAdapter(),
            "hpc-stub": HpcStubSchedulerAdapter(),
        }

    def supported_operations(self) -> list[str]:
        return self.executor.supported_operations()

    def supported_schedulers(self) -> list[str]:
        return sorted(self.scheduler_adapters.keys())

    def register_model_version(
        self,
        model_name: str,
        artifact_path: str,
        *,
        version: str | None = None,
        metrics: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ModelVersionRecord:
        return self.model_registry.register_version(
            model_name,
            artifact_path,
            version=version,
            metrics=metrics,
            metadata=metadata,
        )

    def list_model_versions(self, model_name: str | None = None) -> list[ModelVersionRecord]:
        return self.model_registry.list_versions(model_name=model_name)

    def promote_model_version(
        self,
        model_name: str,
        version: str,
        *,
        stage: str = "production",
        notes: str | None = None,
    ):
        return self.model_registry.promote_version(
            model_name,
            version,
            stage=stage,
            notes=notes,
        )

    def get_active_model_version(
        self,
        model_name: str,
        *,
        stage: str = "production",
    ) -> ModelVersionRecord | None:
        return self.model_registry.get_active_version(model_name, stage=stage)

    def compare_model_versions(
        self,
        model_name: str,
        *,
        candidate_version: str,
        baseline_version: str | None = None,
        baseline_stage: str = "production",
        higher_is_better_metrics: list[str] | None = None,
    ) -> ModelVersionComparisonResult:
        candidate = self.model_registry.get_version(model_name, candidate_version)
        if candidate is None:
            raise ValueError(
                f"Unknown candidate version {candidate_version!r} for model {model_name!r}"
            )
        if baseline_version is not None:
            baseline = self.model_registry.get_version(model_name, baseline_version)
            if baseline is None:
                raise ValueError(
                    f"Unknown baseline version {baseline_version!r} for model {model_name!r}"
                )
        else:
            baseline = self.model_registry.get_active_version(model_name, stage=baseline_stage)
            if baseline is None:
                raise ValueError(
                    f"No active baseline found for model {model_name!r} at stage {baseline_stage!r}"
                )

        higher_is_better = set(higher_is_better_metrics or [])
        all_metric_names = sorted(set(candidate.metrics.keys()) | set(baseline.metrics.keys()))
        candidate_metrics: dict[str, float] = {}
        baseline_metrics: dict[str, float] = {}
        metric_deltas: dict[str, float] = {}
        degraded_metrics: dict[str, float] = {}
        improved_metrics: dict[str, float] = {}
        missing_metrics: list[str] = []

        for metric_name in all_metric_names:
            candidate_metric = _as_float_metric(candidate.metrics.get(metric_name))
            baseline_metric = _as_float_metric(baseline.metrics.get(metric_name))
            if candidate_metric is None or baseline_metric is None:
                missing_metrics.append(metric_name)
                continue
            candidate_metrics[metric_name] = candidate_metric
            baseline_metrics[metric_name] = baseline_metric
            delta = candidate_metric - baseline_metric
            metric_deltas[metric_name] = delta
            if metric_name in higher_is_better:
                if delta < 0:
                    degraded_metrics[metric_name] = -delta
                elif delta > 0:
                    improved_metrics[metric_name] = delta
            else:
                if delta > 0:
                    degraded_metrics[metric_name] = delta
                elif delta < 0:
                    improved_metrics[metric_name] = -delta

        return ModelVersionComparisonResult(
            model_name=model_name,
            candidate_version=candidate.version,
            baseline_version=baseline.version,
            candidate_metrics=candidate_metrics,
            baseline_metrics=baseline_metrics,
            metric_deltas=metric_deltas,
            degraded_metrics=degraded_metrics,
            improved_metrics=improved_metrics,
            missing_metrics=missing_metrics,
        )

    def submit(
        self,
        operation: str,
        parameters: dict[str, Any],
        *,
        metadata_out: str | None = None,
        max_retries: int = 0,
        run_immediately: bool = False,
        scheduler: str = "local",
    ) -> PipelineJobRecord:
        if operation not in self.supported_operations():
            supported = ", ".join(self.supported_operations())
            raise ValueError(f"Unsupported operation {operation!r}. Supported: {supported}")
        if scheduler not in self.scheduler_adapters:
            supported = ", ".join(self.supported_schedulers())
            raise ValueError(f"Unsupported scheduler {scheduler!r}. Supported: {supported}")

        record = PipelineJobRecord(
            job_id=str(uuid4()),
            operation=operation,
            status="pending",
            submitted_at_utc=utc_now_iso(),
            started_at_utc=None,
            finished_at_utc=None,
            parameters=parameters,
            metadata_out=metadata_out,
            operation_run=None,
            error=None,
            attempt_count=0,
            max_retries=max(int(max_retries), 0),
            cancel_requested=False,
            last_updated_at_utc=utc_now_iso(),
            scheduler=scheduler,
            scheduler_job_id=None,
        )
        self.store.save(record)
        adapter = self.scheduler_adapters[scheduler]
        record = adapter.enqueue(record, self.store)
        if run_immediately and scheduler == "local":
            return self.execute_job(record.job_id)
        return record

    def trigger_retraining(
        self,
        *,
        operation: str,
        parameters: dict[str, Any],
        corrected_frames_count: int,
        corrected_frames_threshold: int = 25,
        force: bool = False,
        metadata_out: str | None = None,
        max_retries: int = 0,
        run_immediately: bool = False,
        scheduler: str = "local",
    ) -> RetrainTriggerResult:
        if force:
            reason = "Forced retraining trigger requested."
        elif corrected_frames_count >= corrected_frames_threshold:
            reason = (
                "Corrected-frame threshold met "
                f"({corrected_frames_count} >= {corrected_frames_threshold})."
            )
        else:
            reason = (
                "Retraining trigger not met "
                f"({corrected_frames_count} < {corrected_frames_threshold})."
            )
            return RetrainTriggerResult(triggered=False, reason=reason, job=None)

        job = self.submit(
            operation,
            parameters,
            metadata_out=metadata_out,
            max_retries=max_retries,
            run_immediately=run_immediately,
            scheduler=scheduler,
        )
        return RetrainTriggerResult(triggered=True, reason=reason, job=job)

    def auto_trigger_retraining(
        self,
        *,
        model_name: str,
        candidate_version: str,
        operation: str,
        parameters: dict[str, Any],
        corrected_frames_count: int,
        corrected_frames_threshold: int = 25,
        degradation_thresholds: dict[str, float] | None = None,
        higher_is_better_metrics: list[str] | None = None,
        baseline_version: str | None = None,
        baseline_stage: str = "production",
        force: bool = False,
        metadata_out: str | None = None,
        max_retries: int = 0,
        run_immediately: bool = False,
        scheduler: str = "local",
    ) -> AutoRetrainDecisionResult:
        thresholds = {
            key: float(value)
            for key, value in (degradation_thresholds or {}).items()
            if float(value) >= 0
        }
        comparison = self.compare_model_versions(
            model_name,
            candidate_version=candidate_version,
            baseline_version=baseline_version,
            baseline_stage=baseline_stage,
            higher_is_better_metrics=higher_is_better_metrics,
        )

        degradation_exceeds: dict[str, float] = {}
        for metric_name, amount in comparison.degraded_metrics.items():
            threshold = thresholds.get(metric_name)
            if threshold is not None and amount >= threshold:
                degradation_exceeds[metric_name] = amount

        reasons: list[str] = []
        if force:
            reasons.append("Forced retraining trigger requested.")
        if corrected_frames_count >= corrected_frames_threshold:
            reasons.append(
                "Corrected-frame threshold met "
                f"({corrected_frames_count} >= {corrected_frames_threshold})."
            )
        if degradation_exceeds:
            reasons.append(
                "Model degradation exceeded configured thresholds for: "
                + ", ".join(
                    f"{name}={value:.6f}" for name, value in sorted(degradation_exceeds.items())
                )
            )
        if not reasons:
            reason = (
                "Auto retraining trigger not met "
                f"({corrected_frames_count} < {corrected_frames_threshold} and no metric degradation threshold exceeded)."
            )
            return AutoRetrainDecisionResult(
                triggered=False,
                reason=reason,
                comparison=comparison,
                degradation_exceeds=degradation_exceeds,
                job=None,
            )

        job = self.submit(
            operation,
            parameters,
            metadata_out=metadata_out,
            max_retries=max_retries,
            run_immediately=run_immediately,
            scheduler=scheduler,
        )
        return AutoRetrainDecisionResult(
            triggered=True,
            reason=" ".join(reasons),
            comparison=comparison,
            degradation_exceeds=degradation_exceeds,
            job=job,
        )

    def submit_and_run(
        self,
        operation: str,
        parameters: dict[str, Any],
        *,
        metadata_out: str | None = None,
        max_retries: int = 0,
        scheduler: str = "local",
    ) -> PipelineJobRecord:
        return self.submit(
            operation,
            parameters,
            metadata_out=metadata_out,
            max_retries=max_retries,
            run_immediately=True,
            scheduler=scheduler,
        )

    def get_job(self, job_id: str) -> PipelineJobRecord:
        return self.store.load(job_id)

    def list_jobs(self, limit: int | None = None) -> list[PipelineJobRecord]:
        return self.store.list_jobs(limit=limit)

    def run_queue(
        self,
        *,
        scheduler: str = "local",
        max_jobs: int | None = None,
    ) -> list[PipelineJobRecord]:
        if scheduler not in self.scheduler_adapters:
            supported = ", ".join(self.supported_schedulers())
            raise ValueError(f"Unsupported scheduler {scheduler!r}. Supported: {supported}")
        adapter = self.scheduler_adapters[scheduler]
        return adapter.drain(self, max_jobs=max_jobs)

    def request_cancel(self, job_id: str) -> PipelineJobRecord:
        record = self.store.load(job_id)
        adapter = self.scheduler_adapters.get(record.scheduler)
        if adapter is None:
            record.cancel_requested = True
            if record.status in {"pending", "retrying"}:
                record.status = "canceled"
                record.finished_at_utc = utc_now_iso()
            self.store.save(record)
            return record
        return adapter.cancel(record, self.store)

    def retry_job(self, job_id: str, *, extra_retries: int = 1) -> PipelineJobRecord:
        record = self.store.load(job_id)
        if record.status not in {"failed", "canceled", "queued_remote"}:
            raise ValueError(f"Job {job_id} is not retryable from state {record.status!r}")

        consumed_retries = max(record.attempt_count - 1, 0)
        record.max_retries = max(record.max_retries, consumed_retries) + max(
            int(extra_retries), 1
        )
        record.cancel_requested = False
        record.status = "retrying"
        record.finished_at_utc = None
        record.error = None
        record.operation_run = None
        self.store.save(record)
        if record.scheduler == "local":
            return self.execute_job(job_id)
        adapter = self.scheduler_adapters.get(record.scheduler)
        if adapter is None:
            return record
        return adapter.enqueue(record, self.store)

    def execute_job(self, job_id: str) -> PipelineJobRecord:
        record = self.store.load(job_id)
        if record.status in {"succeeded", "canceled"}:
            return record
        if record.scheduler != "local":
            raise ValueError(
                f"Job {job_id} uses scheduler {record.scheduler!r}; "
                "run via run-queue for that scheduler."
            )
        if record.cancel_requested:
            record.status = "canceled"
            record.finished_at_utc = utc_now_iso()
            self.store.save(record)
            return record
        if record.operation not in self.supported_operations():
            record.status = "failed"
            record.error = (
                f"Unsupported operation {record.operation!r}. "
                f"Supported: {', '.join(self.supported_operations())}"
            )
            record.finished_at_utc = utc_now_iso()
            self.store.save(record)
            return record

        while True:
            if record.cancel_requested:
                record.status = "canceled"
                record.finished_at_utc = utc_now_iso()
                self.store.save(record)
                return record

            record.status = "running"
            if record.started_at_utc is None:
                record.started_at_utc = utc_now_iso()
            record.attempt_count += 1
            self.store.save(record)

            try:
                op_run = self.executor.execute(
                    record.operation,
                    record.parameters,
                    metadata_out=record.metadata_out,
                )
                record.status = "succeeded"
                record.operation_run = asdict(op_run)
                record.error = None
                record.finished_at_utc = utc_now_iso()
                self.store.save(record)
                return record
            except Exception as exc:
                record.error = str(exc)
                record.operation_run = None
                if record.cancel_requested:
                    record.status = "canceled"
                    record.finished_at_utc = utc_now_iso()
                    self.store.save(record)
                    return record
                if record.attempt_count <= record.max_retries:
                    record.status = "retrying"
                    self.store.save(record)
                    continue
                record.status = "failed"
                record.finished_at_utc = utc_now_iso()
                self.store.save(record)
                return record
