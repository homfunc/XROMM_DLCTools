from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from .models import OperationRunMetadata, OperationSpec


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def create_run_metadata(
    operation: OperationSpec, declared_outputs: dict
) -> OperationRunMetadata:
    return OperationRunMetadata(
        run_id=str(uuid4()),
        operation=operation.name,
        status="running",
        started_at_utc=utc_now_iso(),
        finished_at_utc=None,
        duration_seconds=None,
        parameters=operation.parameters,
        declared_outputs=declared_outputs,
        error=None,
    )


def finalize_run_metadata(
    record: OperationRunMetadata,
    *,
    status: str,
    error: str | None = None,
) -> OperationRunMetadata:
    finished = utc_now_iso()
    started = datetime.fromisoformat(record.started_at_utc)
    finished_dt = datetime.fromisoformat(finished)
    record.status = status
    record.finished_at_utc = finished
    record.duration_seconds = round((finished_dt - started).total_seconds(), 3)
    record.error = error
    return record


def write_run_metadata(path: str | Path, record: OperationRunMetadata) -> Path:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(asdict(record), indent=2))
    return out_path
