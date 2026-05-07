from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .metadata import utc_now_iso


@dataclass
class ModelVersionRecord:
    model_name: str
    version: str
    artifact_path: str
    created_at_utc: str
    metrics: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelPromotionRecord:
    model_name: str
    version: str
    stage: str
    promoted_at_utc: str
    notes: str | None = None


class ModelRegistry:
    def __init__(self, registry_path: str | Path = ".xrommtools/model_registry.json"):
        self.registry_path = Path(registry_path)
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)

    def _default_payload(self) -> dict[str, Any]:
        return {"models": {}, "promotions": []}

    def _load(self) -> dict[str, Any]:
        if not self.registry_path.exists():
            return self._default_payload()
        payload = json.loads(self.registry_path.read_text())
        if "models" not in payload:
            payload["models"] = {}
        if "promotions" not in payload:
            payload["promotions"] = []
        return payload

    def _save(self, payload: dict[str, Any]) -> None:
        self.registry_path.write_text(json.dumps(payload, indent=2))

    def register_version(
        self,
        model_name: str,
        artifact_path: str,
        *,
        version: str | None = None,
        metrics: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ModelVersionRecord:
        payload = self._load()
        models = payload["models"]
        model_entry = models.setdefault(model_name, {"versions": [], "stage_aliases": {}})
        versions = model_entry.setdefault("versions", [])
        if version is None:
            version = f"v{len(versions) + 1}"

        for existing in versions:
            if existing.get("version") == version:
                raise ValueError(f"Model {model_name!r} already has version {version!r}")

        record = ModelVersionRecord(
            model_name=model_name,
            version=version,
            artifact_path=artifact_path,
            created_at_utc=utc_now_iso(),
            metrics=metrics or {},
            metadata=metadata or {},
        )
        versions.append(asdict(record))
        self._save(payload)
        return record

    def list_versions(self, model_name: str | None = None) -> list[ModelVersionRecord]:
        payload = self._load()
        out: list[ModelVersionRecord] = []
        models = payload["models"]
        if model_name is not None:
            model_entry = models.get(model_name, {})
            for item in model_entry.get("versions", []):
                out.append(ModelVersionRecord(**item))
            return out

        for name, entry in models.items():
            for item in entry.get("versions", []):
                if "model_name" not in item:
                    item = dict(item)
                    item["model_name"] = name
                out.append(ModelVersionRecord(**item))
        out.sort(key=lambda r: (r.model_name, r.created_at_utc))
        return out

    def get_version(self, model_name: str, version: str) -> ModelVersionRecord | None:
        payload = self._load()
        models = payload["models"]
        model_entry = models.get(model_name)
        if model_entry is None:
            return None
        for item in model_entry.get("versions", []):
            if item.get("version") == version:
                return ModelVersionRecord(**item)
        return None

    def promote_version(
        self,
        model_name: str,
        version: str,
        *,
        stage: str = "production",
        notes: str | None = None,
    ) -> ModelPromotionRecord:
        payload = self._load()
        models = payload["models"]
        model_entry = models.get(model_name)
        if model_entry is None:
            raise ValueError(f"Unknown model {model_name!r}")
        versions = model_entry.get("versions", [])
        if not any(v.get("version") == version for v in versions):
            raise ValueError(f"Unknown version {version!r} for model {model_name!r}")

        model_entry.setdefault("stage_aliases", {})[stage] = version
        promotion = ModelPromotionRecord(
            model_name=model_name,
            version=version,
            stage=stage,
            promoted_at_utc=utc_now_iso(),
            notes=notes,
        )
        payload.setdefault("promotions", []).append(asdict(promotion))
        self._save(payload)
        return promotion

    def get_active_version(
        self,
        model_name: str,
        *,
        stage: str = "production",
    ) -> ModelVersionRecord | None:
        payload = self._load()
        models = payload["models"]
        model_entry = models.get(model_name)
        if model_entry is None:
            return None
        active_version = model_entry.get("stage_aliases", {}).get(stage)
        if active_version is None:
            return None
        for item in model_entry.get("versions", []):
            if item.get("version") == active_version:
                return ModelVersionRecord(**item)
        return None
