from __future__ import annotations

import base64
import json
from dataclasses import asdict
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, cast
from urllib.parse import parse_qs, urlparse
from uuid import uuid4

from .registry import ModelRegistry
from .service import LocalJobStore, LocalPipelineOrchestrator

_OPERATION_SPECS: dict[str, dict[str, Any]] = {
    "xma_to_dlc": {
        "label": "1) Upload / ingest",
        "description": "Convert XMALab trial data into DLC labeled-data format.",
        "parameters_template": {
            "path_config_file": "path/to/config.yaml",
            "data_path": "path/to/trials_root",
            "dataset_name": "dataset_name",
            "scorer": "scorer_name",
            "nframes": 25,
            "nnetworks": 1,
            "path_config_file_cam2": [],
        },
    },
    "analyze_xromm_videos": {
        "label": "2) Predict",
        "description": "Run DLC prediction across trial videos and convert output to XMALab layout.",
        "parameters_template": {
            "path_config_file": "path/to/config.yaml",
            "path_data_to_analyze": "path/to/trials_root",
            "iteration": 1,
            "nnetworks": 1,
            "path_config_file_cam2": [],
        },
    },
    "optimize_dlc_predictions": {
        "label": "3a) Review queue (temporal)",
        "description": "Generate optimization report with ranked review frames.",
        "parameters_template": {
            "cam1data": "path/to/cam1_predictions.csv",
            "cam2data": "path/to/cam2_predictions.csv",
            "trialname": "trial_001",
            "savepath": "path/to/output_dir",
            "transition_weight": 0.02,
            "process_noise": 0.01,
            "top_k_review_frames": 25,
        },
    },
    "triangulate_dlc_predictions": {
        "label": "3b) Review queue (3D)",
        "description": "Generate triangulation report with reprojection/rigid residual review frames.",
        "parameters_template": {
            "cam1data": "path/to/cam1_predictions.csv",
            "cam2data": "path/to/cam2_predictions.csv",
            "trialname": "trial_001",
            "savepath": "path/to/output_dir",
            "cam1_projection": "path/to/cam1_projection.json",
            "cam2_projection": "path/to/cam2_projection.json",
            "min_confidence": 0.1,
            "top_k_review_frames": 25,
            "rigid_iterations": 0,
            "rigid_step_size": 0.05,
            "reprojection_anchor_weight": 0.4,
            "constraints_json": None,
        },
    },
    "add_frames": {
        "label": "4) Retrain augmentation",
        "description": "Append corrected frames into existing DLC labeled-data artifacts.",
        "parameters_template": {
            "path_config_file": "path/to/config.yaml",
            "data_path": "path/to/trials_root",
            "iteration": 1,
            "frames": "path/to/frames_to_add.csv",
            "nnetworks": 1,
            "path_config_file_cam2": "enterpathofcam2config",
        },
    },
    "dlc_to_xma": {
        "label": "5) Export",
        "description": "Convert camera prediction files into XMALab Predicted2DPoints output.",
        "parameters_template": {
            "cam1data": "path/to/cam1_predictions.csv",
            "cam2data": "path/to/cam2_predictions.csv",
            "trialname": "trial_001",
            "savepath": "path/to/output_dir",
        },
    },
}

_WORKFLOW_BLUEPRINT = {
    "name": "phase3_single_interface",
    "label": "Phase 3 guided single-run flow",
    "steps": [
        {"operation": "xma_to_dlc", "title": "Upload/Ingest"},
        {"operation": "analyze_xromm_videos", "title": "Predict"},
        {"operation": "optimize_dlc_predictions", "title": "Review queue"},
        {"operation": "add_frames", "title": "Retrain augmentation"},
        {"operation": "dlc_to_xma", "title": "Export"},
    ],
}

_UI_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>XROMM Tools Workflow UI</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <style>
    body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 1rem auto; max-width: 1180px; padding: 0 1rem; }
    h1, h2 { margin-bottom: 0.35rem; }
    .muted { color: #666; margin-top: 0; }
    .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(330px, 1fr)); gap: 1rem; }
    .panel { border: 1px solid #ddd; border-radius: 8px; padding: 0.75rem; }
    label { display: block; margin-top: 0.5rem; font-weight: 600; }
    input, select, textarea, button { width: 100%; box-sizing: border-box; margin-top: 0.2rem; }
    textarea { min-height: 110px; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }
    .row { display: grid; grid-template-columns: 1fr 1fr; gap: 0.5rem; }
    .actions { display: flex; gap: 0.5rem; margin-top: 0.7rem; }
    .actions button { width: auto; padding: 0.35rem 0.7rem; }
    pre { background: #f6f8fa; border-radius: 6px; padding: 0.65rem; overflow: auto; max-height: 360px; }
    .hint { font-size: 0.9rem; color: #444; margin-top: 0.4rem; }
    .pill { display: inline-block; padding: 0.12rem 0.5rem; border-radius: 999px; background: #eef3ff; margin-right: 0.35rem; }
  </style>
</head>
<body>
  <h1>XROMM Tools Workflow UI</h1>
  <p class="muted">Slice 9 guided flow: upload/ingest → predict → review → retrain → export with summary artifact.</p>
  <div class="grid">
    <section class="panel">
      <h2>Service</h2>
      <div id="serviceStatus">Checking...</div>
      <div class="actions">
        <button id="refreshCapabilities">Refresh capabilities</button>
      </div>
      <pre id="capabilitiesOut"></pre>
    </section>
    <section class="panel">
      <h2>Upload artifacts</h2>
      <label for="uploadFile">Choose file</label>
      <input id="uploadFile" type="file">
      <label for="uploadSubdir">Destination subdirectory</label>
      <input id="uploadSubdir" value="phase3_uploads">
      <div class="actions">
        <button id="uploadFileBtn">Upload</button>
      </div>
      <pre id="uploadOut"></pre>
    </section>
    <section class="panel">
      <h2>Guided workflow wizard</h2>
      <div id="wizardStatus"></div>
      <div class="actions">
        <button id="wizardPrevBtn">Previous</button>
        <button id="wizardNextBtn">Next</button>
      </div>
      <label for="wizardOperation">Current operation</label>
      <input id="wizardOperation" readonly>
      <div class="hint" id="wizardDescription"></div>
      <label for="wizardParams">Current step parameters (JSON object)</label>
      <textarea id="wizardParams">{}</textarea>
      <div class="row">
        <div>
          <label for="wizardScheduler">Scheduler</label>
          <input id="wizardScheduler" value="local">
        </div>
        <div>
          <label for="wizardMaxRetries">Max retries</label>
          <input id="wizardMaxRetries" type="number" min="0" value="0">
        </div>
      </div>
      <label><input id="wizardRunImmediately" type="checkbox" checked> Run immediately</label>
      <div class="actions">
        <button id="wizardRunBtn">Run current step</button>
      </div>
      <pre id="wizardOut"></pre>
    </section>
    <section class="panel">
      <h2>Review queue → retrain handoff</h2>
      <label for="reviewReportPath">Review report path</label>
      <input id="reviewReportPath" placeholder="path/to/*-OptimizationReport.json or *-TriangulationReport.json">
      <div class="actions">
        <button id="loadReviewBtn">Load review report</button>
      </div>
      <label for="reviewTrialName">Trial name</label>
      <input id="reviewTrialName" placeholder="trial_001">
      <label for="reviewSelectedFrames">Selected frames (comma-separated)</label>
      <input id="reviewSelectedFrames" placeholder="12,31,52">
      <div class="row">
        <div>
          <label for="reviewConfigPath">DLC config path</label>
          <input id="reviewConfigPath" placeholder="path/to/config.yaml">
        </div>
        <div>
          <label for="reviewDataPath">Trial root path</label>
          <input id="reviewDataPath" placeholder="path/to/trials_root">
        </div>
      </div>
      <div class="row">
        <div>
          <label for="reviewIteration">Iteration</label>
          <input id="reviewIteration" type="number" min="1" value="1">
        </div>
        <div>
          <label for="reviewScheduler">Scheduler</label>
          <input id="reviewScheduler" value="local">
        </div>
      </div>
      <label><input id="reviewRunImmediately" type="checkbox" checked> Run retrain augmentation immediately</label>
      <div class="actions">
        <button id="reviewToRetrainBtn">Submit retrain augmentation</button>
      </div>
      <pre id="reviewOut"></pre>
    </section>
    <section class="panel">
      <h2>One-click export completion</h2>
      <label for="exportParams">Export parameters (JSON object for dlc_to_xma)</label>
      <textarea id="exportParams">{
  "cam1data": "path/to/cam1_predictions.csv",
  "cam2data": "path/to/cam2_predictions.csv",
  "trialname": "trial_001",
  "savepath": "path/to/output_dir"
}</textarea>
      <div class="row">
        <div>
          <label for="exportScheduler">Scheduler</label>
          <input id="exportScheduler" value="local">
        </div>
        <div>
          <label for="exportMaxRetries">Max retries</label>
          <input id="exportMaxRetries" type="number" min="0" value="0">
        </div>
      </div>
      <label><input id="exportRunImmediately" type="checkbox" checked> Run export immediately</label>
      <div class="actions">
        <button id="completeExportBtn">Complete export + write summary</button>
      </div>
      <pre id="exportOut"></pre>
    </section>
    <section class="panel">
      <h2>Jobs</h2>
      <div class="row">
        <div>
          <label for="jobsLimit">List limit</label>
          <input id="jobsLimit" type="number" min="1" value="20">
        </div>
        <div>
          <label for="queueScheduler">Run queue scheduler</label>
          <input id="queueScheduler" value="local">
        </div>
      </div>
      <div class="actions">
        <button id="refreshJobsBtn">Refresh jobs</button>
        <button id="runQueueBtn">Run queue</button>
      </div>
      <label for="jobLookupId">Lookup job by ID</label>
      <input id="jobLookupId" placeholder="job-id">
      <div class="actions">
        <button id="lookupJobBtn">Lookup</button>
      </div>
      <pre id="jobsOut"></pre>
    </section>
  </div>
  <script src="/ui/app.js"></script>
</body>
</html>
"""

_UI_APP_JS = """const setText = (id, text) => {
  const el = document.getElementById(id);
  if (el) el.textContent = text;
};

const pretty = (value) => JSON.stringify(value, null, 2);
const splitNames = (value) => value.split(",").map((item) => item.trim()).filter((item) => item);

const readJsonTextarea = (id) => {
  const raw = document.getElementById(id).value.trim();
  if (!raw) return {};
  const parsed = JSON.parse(raw);
  if (typeof parsed !== "object" || Array.isArray(parsed) || parsed === null) {
    throw new Error(id + " must contain a JSON object");
  }
  return parsed;
};

const api = async (path, method = "GET", body = null) => {
  const options = { method, headers: {} };
  if (body !== null) {
    options.headers["Content-Type"] = "application/json";
    options.body = JSON.stringify(body);
  }
  const response = await fetch(path, options);
  const text = await response.text();
  let payload = null;
  if (text) {
    try { payload = JSON.parse(text); } catch (err) { payload = text; }
  }
  if (!response.ok) {
    if (payload && typeof payload === "object" && payload.error) throw new Error(payload.error);
    throw new Error("HTTP " + response.status);
  }
  return payload;
};

let operationSpecs = {};
let wizardSteps = [];
let wizardIndex = 0;
const wizardRunHistory = [];

const renderWizard = () => {
  if (wizardSteps.length === 0) return;
  const step = wizardSteps[wizardIndex];
  document.getElementById("wizardOperation").value = step.operation;
  const spec = operationSpecs[step.operation] || {};
  setText("wizardDescription", (step.title || step.operation) + ": " + (spec.description || ""));
  document.getElementById("wizardParams").value = pretty(spec.parameters_template || {});
  const labels = wizardSteps.map((item, idx) => {
    const marker = idx === wizardIndex ? "▶" : (idx < wizardIndex ? "✓" : "•");
    return marker + " " + (item.title || item.operation);
  });
  setText("wizardStatus", labels.join("   "));
};

const loadCapabilities = async () => {
  const health = await api("/health");
  const capabilities = await api("/capabilities");
  const specs = await api("/operation-specs");
  const blueprint = await api("/workflow-blueprints");
  operationSpecs = specs || {};
  wizardSteps = (blueprint && blueprint.steps) || [];
  wizardIndex = 0;
  setText("serviceStatus", "Health: " + health.status);
  setText("capabilitiesOut", pretty(capabilities));
  renderWizard();
};

const loadJobs = async () => {
  const limit = Number(document.getElementById("jobsLimit").value || "20");
  const jobs = await api("/jobs?limit=" + encodeURIComponent(limit));
  setText("jobsOut", pretty(jobs));
};

document.getElementById("refreshCapabilities").addEventListener("click", async () => {
  try { await loadCapabilities(); } catch (err) { setText("capabilitiesOut", String(err)); }
});

document.getElementById("wizardPrevBtn").addEventListener("click", () => {
  if (wizardIndex > 0) wizardIndex -= 1;
  renderWizard();
});

document.getElementById("wizardNextBtn").addEventListener("click", () => {
  if (wizardIndex < wizardSteps.length - 1) wizardIndex += 1;
  renderWizard();
});

document.getElementById("wizardRunBtn").addEventListener("click", async () => {
  try {
    const operation = document.getElementById("wizardOperation").value.trim();
    const payload = {
      operation,
      parameters: readJsonTextarea("wizardParams"),
      scheduler: document.getElementById("wizardScheduler").value || "local",
      max_retries: Number(document.getElementById("wizardMaxRetries").value || "0"),
      run_immediately: document.getElementById("wizardRunImmediately").checked
    };
    const result = await api("/run-operation", "POST", payload);
    wizardRunHistory.push({ operation, job_id: result.job_id, status: result.status });
    setText("wizardOut", pretty({ result, history: wizardRunHistory }));
    await loadJobs();
  } catch (err) {
    setText("wizardOut", String(err));
  }
});

document.getElementById("uploadFileBtn").addEventListener("click", async () => {
  try {
    const fileInput = document.getElementById("uploadFile");
    const file = fileInput.files && fileInput.files[0];
    if (!file) throw new Error("choose a file first");
    const dataUrl = await new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => resolve(reader.result);
      reader.onerror = () => reject(new Error("failed to read file"));
      reader.readAsDataURL(file);
    });
    const base64Content = String(dataUrl).split(",", 2)[1] || "";
    const result = await api("/upload-file", "POST", {
      filename: file.name,
      content_base64: base64Content,
      subdir: document.getElementById("uploadSubdir").value || "phase3_uploads"
    });
    setText("uploadOut", pretty(result));
  } catch (err) {
    setText("uploadOut", String(err));
  }
});

document.getElementById("loadReviewBtn").addEventListener("click", async () => {
  try {
    const reportPath = document.getElementById("reviewReportPath").value.trim();
    if (!reportPath) throw new Error("report path is required");
    const result = await api("/review-report?path=" + encodeURIComponent(reportPath));
    const frames = (result.review_frame_numbers || []).slice(0, 50);
    document.getElementById("reviewSelectedFrames").value = frames.join(",");
    setText("reviewOut", pretty(result));
  } catch (err) {
    setText("reviewOut", String(err));
  }
});

document.getElementById("reviewToRetrainBtn").addEventListener("click", async () => {
  try {
    const frames = splitNames(document.getElementById("reviewSelectedFrames").value).map((item) => Number(item));
    const payload = {
      trial_name: document.getElementById("reviewTrialName").value.trim(),
      selected_frames: frames,
      path_config_file: document.getElementById("reviewConfigPath").value.trim(),
      data_path: document.getElementById("reviewDataPath").value.trim(),
      iteration: Number(document.getElementById("reviewIteration").value || "1"),
      scheduler: document.getElementById("reviewScheduler").value || "local",
      run_immediately: document.getElementById("reviewRunImmediately").checked
    };
    const result = await api("/review-to-retrain", "POST", payload);
    setText("reviewOut", pretty(result));
    await loadJobs();
  } catch (err) {
    setText("reviewOut", String(err));
  }
});

document.getElementById("completeExportBtn").addEventListener("click", async () => {
  try {
    const payload = {
      parameters: readJsonTextarea("exportParams"),
      scheduler: document.getElementById("exportScheduler").value || "local",
      max_retries: Number(document.getElementById("exportMaxRetries").value || "0"),
      run_immediately: document.getElementById("exportRunImmediately").checked,
      workflow_context: {
        wizard_history: wizardRunHistory
      }
    };
    const result = await api("/one-click-export", "POST", payload);
    setText("exportOut", pretty(result));
    await loadJobs();
  } catch (err) {
    setText("exportOut", String(err));
  }
});

document.getElementById("refreshJobsBtn").addEventListener("click", async () => {
  try { await loadJobs(); } catch (err) { setText("jobsOut", String(err)); }
});

document.getElementById("runQueueBtn").addEventListener("click", async () => {
  try {
    const payload = {
      scheduler: document.getElementById("queueScheduler").value || "local",
      max_jobs: Number(document.getElementById("jobsLimit").value || "20")
    };
    const result = await api("/run-queue", "POST", payload);
    setText("jobsOut", pretty(result));
  } catch (err) {
    setText("jobsOut", String(err));
  }
});

document.getElementById("lookupJobBtn").addEventListener("click", async () => {
  try {
    const jobId = document.getElementById("jobLookupId").value.trim();
    if (!jobId) throw new Error("job id is required");
    const result = await api("/jobs/" + encodeURIComponent(jobId));
    setText("jobsOut", pretty(result));
  } catch (err) {
    setText("jobsOut", String(err));
  }
});

(async () => {
  try {
    await loadCapabilities();
    await loadJobs();
  } catch (err) {
    setText("serviceStatus", "Error: " + String(err));
  }
})();
"""


def _collect_review_candidates(payload: Any, *, max_items: int = 50) -> list[Any]:
    candidates: list[Any] = []

    def _walk(value: Any) -> None:
        if len(candidates) >= max_items:
            return
        if isinstance(value, dict):
            for key, nested in value.items():
                if "review" in str(key).lower() and isinstance(nested, list):
                    for item in nested:
                        candidates.append(item)
                        if len(candidates) >= max_items:
                            return
                _walk(nested)
        elif isinstance(value, list):
            for item in value:
                _walk(item)
                if len(candidates) >= max_items:
                    return

    _walk(payload)
    return candidates[:max_items]


def _extract_review_frames(candidates: list[Any], *, max_items: int = 50) -> list[int]:
    frames: list[int] = []
    for item in candidates:
        if len(frames) >= max_items:
            break
        if isinstance(item, dict):
            for key in ("frame", "frame_index", "frame_idx"):
                value = item.get(key)
                if isinstance(value, (int, float)):
                    frame = int(value)
                    if frame not in frames:
                        frames.append(frame)
                    break
    return frames


class OrchestratorHttpServer(ThreadingHTTPServer):
    def __init__(
        self,
        server_address: tuple[str, int],
        request_handler_class: type[BaseHTTPRequestHandler],
        orchestrator: LocalPipelineOrchestrator,
    ):
        self.orchestrator = orchestrator
        super().__init__(server_address, request_handler_class)


class _JsonApiHandler(BaseHTTPRequestHandler):
    @property
    def orchestrator(self) -> LocalPipelineOrchestrator:
        return cast(OrchestratorHttpServer, self.server).orchestrator

    def do_GET(self) -> None:  # noqa: N802
        try:
            parsed = urlparse(self.path)
            path = parsed.path.rstrip("/") or "/"
            query = parse_qs(parsed.query)
            if path in {"/", "/ui"}:
                self._write_text(200, _UI_HTML, content_type="text/html; charset=utf-8")
                return
            if path == "/ui/app.js":
                self._write_text(
                    200,
                    _UI_APP_JS,
                    content_type="text/javascript; charset=utf-8",
                )
                return
            if path == "/health":
                self._write_json(200, {"status": "ok"})
                return
            if path == "/capabilities":
                self._write_json(
                    200,
                    {
                        "operations": self.orchestrator.supported_operations(),
                        "schedulers": self.orchestrator.supported_schedulers(),
                    },
                )
                return
            if path == "/operation-specs":
                specs = {
                    name: spec
                    for name, spec in _OPERATION_SPECS.items()
                    if name in self.orchestrator.supported_operations()
                }
                self._write_json(200, specs)
                return
            if path == "/workflow-blueprints":
                self._write_json(200, _WORKFLOW_BLUEPRINT)
                return
            if path == "/review-report":
                report_path = self._query_required(query, "path")
                payload = json.loads(Path(report_path).read_text())
                review_candidates = _collect_review_candidates(payload)
                self._write_json(
                    200,
                    {
                        "path": report_path,
                        "review_candidates": review_candidates,
                        "review_frame_numbers": _extract_review_frames(review_candidates),
                        "report": payload,
                    },
                )
                return
            if path == "/model-versions":
                model_name = self._query_first(query, "model_name")
                records = self.orchestrator.list_model_versions(model_name=model_name)
                self._write_json(200, [asdict(record) for record in records])
                return
            if path == "/active-model-version":
                model_name = self._query_required(query, "model_name")
                stage = self._query_first(query, "stage") or "production"
                active = self.orchestrator.get_active_model_version(
                    model_name,
                    stage=stage,
                )
                if active is None:
                    self._write_json(200, None)
                else:
                    self._write_json(200, asdict(active))
                return
            if path == "/jobs":
                limit_raw = self._query_first(query, "limit")
                limit = 20 if limit_raw is None else int(limit_raw)
                records = self.orchestrator.list_jobs(limit=limit)
                self._write_json(200, [asdict(record) for record in records])
                return
            if path.startswith("/jobs/"):
                job_id = path.split("/", maxsplit=2)[-1]
                if not job_id:
                    raise ValueError("job id is required")
                record = self.orchestrator.get_job(job_id)
                self._write_json(200, asdict(record))
                return
            self._write_json(404, {"error": f"unknown path: {path}"})
        except FileNotFoundError:
            self._write_json(404, {"error": "file or job not found"})
        except ValueError as exc:
            self._write_json(400, {"error": str(exc)})
        except Exception as exc:  # pragma: no cover - defensive handler
            self._write_json(500, {"error": str(exc)})

    def do_POST(self) -> None:  # noqa: N802
        try:
            parsed = urlparse(self.path)
            path = parsed.path.rstrip("/") or "/"
            body = self._read_json_dict()
            if path == "/upload-file":
                filename = Path(str(body["filename"])).name
                if not filename:
                    raise ValueError("filename is required")
                content_base64 = str(body.get("content_base64", "")).strip()
                if not content_base64:
                    raise ValueError("content_base64 is required")
                file_bytes = base64.b64decode(content_base64, validate=True)
                subdir = self._optional_string(body.get("subdir")) or "phase3_uploads"
                safe_subdir = Path(subdir.replace("..", "").lstrip("/"))
                upload_dir = self._workspace_root() / "uploads" / safe_subdir
                upload_dir.mkdir(parents=True, exist_ok=True)
                output_path = upload_dir / f"{uuid4().hex[:10]}_{filename}"
                output_path.write_bytes(file_bytes)
                self._write_json(
                    200,
                    {
                        "saved_path": str(output_path),
                        "size_bytes": len(file_bytes),
                        "filename": filename,
                    },
                )
                return
            if path in {"/submit-job", "/run-operation"}:
                parameters = body.get("parameters")
                if not isinstance(parameters, dict):
                    raise ValueError("parameters must be a JSON object")
                record = self.orchestrator.submit(
                    str(body["operation"]),
                    parameters,
                    metadata_out=self._optional_string(body.get("metadata_out")),
                    max_retries=int(body.get("max_retries", 0)),
                    run_immediately=bool(body.get("run_immediately", path == "/run-operation")),
                    scheduler=str(body.get("scheduler", "local")),
                )
                self._write_json(200, asdict(record))
                return
            if path == "/review-to-retrain":
                trial_name = self._required_string(body, "trial_name")
                path_config_file = self._required_string(body, "path_config_file")
                data_path = self._required_string(body, "data_path")
                iteration = int(body.get("iteration", 1))
                frames_input = body.get("selected_frames")
                selected_frames = self._parse_selected_frames(frames_input)
                if not selected_frames:
                    raise ValueError("selected_frames must include at least one frame")
                workflow_runs_dir = self._workspace_root() / "workflow_runs"
                workflow_runs_dir.mkdir(parents=True, exist_ok=True)
                frames_csv_path = (
                    workflow_runs_dir / f"review_frames_{uuid4().hex[:10]}.csv"
                )
                frames_csv_path.write_text(
                    ",".join([trial_name, *[str(frame) for frame in selected_frames]])
                    + "\n"
                )
                parameters = {
                    "path_config_file": path_config_file,
                    "data_path": data_path,
                    "iteration": iteration,
                    "frames": str(frames_csv_path),
                    "nnetworks": int(body.get("nnetworks", 1)),
                    "path_config_file_cam2": body.get(
                        "path_config_file_cam2",
                        "enterpathofcam2config",
                    ),
                }
                record = self.orchestrator.submit(
                    "add_frames",
                    parameters,
                    max_retries=int(body.get("max_retries", 0)),
                    run_immediately=bool(body.get("run_immediately", True)),
                    scheduler=str(body.get("scheduler", "local")),
                )
                self._write_json(
                    200,
                    {
                        "frames_csv_path": str(frames_csv_path),
                        "selected_frames": selected_frames,
                        "job": asdict(record),
                    },
                )
                return
            if path == "/one-click-export":
                parameters = body.get("parameters")
                if not isinstance(parameters, dict):
                    raise ValueError("parameters must be a JSON object")
                record = self.orchestrator.submit(
                    "dlc_to_xma",
                    parameters,
                    metadata_out=self._optional_string(body.get("metadata_out")),
                    max_retries=int(body.get("max_retries", 0)),
                    run_immediately=bool(body.get("run_immediately", True)),
                    scheduler=str(body.get("scheduler", "local")),
                )
                summary_dir = self._workspace_root() / "workflow_runs"
                summary_dir.mkdir(parents=True, exist_ok=True)
                summary_path = summary_dir / f"export_summary_{record.job_id}.json"
                summary_payload = {
                    "created_at_utc": datetime.now(timezone.utc).isoformat(),
                    "workflow_stage": "export_completed",
                    "export_job": asdict(record),
                    "workflow_context": body.get("workflow_context", {}),
                }
                summary_path.write_text(json.dumps(summary_payload, indent=2))
                self._write_json(
                    200,
                    {
                        "summary_path": str(summary_path),
                        "summary": summary_payload,
                    },
                )
                return
            if path == "/run-job":
                record = self.orchestrator.execute_job(str(body["job_id"]))
                self._write_json(200, asdict(record))
                return
            if path == "/retry-job":
                record = self.orchestrator.retry_job(
                    str(body["job_id"]),
                    extra_retries=int(body.get("extra_retries", 1)),
                )
                self._write_json(200, asdict(record))
                return
            if path == "/cancel-job":
                record = self.orchestrator.request_cancel(str(body["job_id"]))
                self._write_json(200, asdict(record))
                return
            if path == "/run-queue":
                scheduler = str(body.get("scheduler", "local"))
                max_jobs_raw = body.get("max_jobs")
                max_jobs = None if max_jobs_raw is None else int(max_jobs_raw)
                records = self.orchestrator.run_queue(
                    scheduler=scheduler,
                    max_jobs=max_jobs,
                )
                self._write_json(200, [asdict(record) for record in records])
                return
            if path == "/compare-model-versions":
                higher_is_better = self._metric_name_list(
                    body.get("higher_is_better_metrics")
                )
                result = self.orchestrator.compare_model_versions(
                    str(body["model_name"]),
                    candidate_version=str(body["candidate_version"]),
                    baseline_version=self._optional_string(body.get("baseline_version")),
                    baseline_stage=str(body.get("baseline_stage", "production")),
                    higher_is_better_metrics=higher_is_better,
                )
                self._write_json(200, asdict(result))
                return
            if path == "/trigger-retrain":
                parameters = body.get("parameters")
                if not isinstance(parameters, dict):
                    raise ValueError("parameters must be a JSON object")
                result = self.orchestrator.trigger_retraining(
                    operation=str(body["operation"]),
                    parameters=parameters,
                    corrected_frames_count=int(body["corrected_frames_count"]),
                    corrected_frames_threshold=int(
                        body.get("corrected_frames_threshold", 25)
                    ),
                    force=bool(body.get("force", False)),
                    metadata_out=self._optional_string(body.get("metadata_out")),
                    max_retries=int(body.get("max_retries", 0)),
                    run_immediately=bool(body.get("run_immediately", False)),
                    scheduler=str(body.get("scheduler", "local")),
                )
                self._write_json(200, asdict(result))
                return
            if path == "/auto-trigger-retrain":
                parameters = body.get("parameters")
                if not isinstance(parameters, dict):
                    raise ValueError("parameters must be a JSON object")
                degradation_thresholds = body.get("degradation_thresholds", {})
                if not isinstance(degradation_thresholds, dict):
                    raise ValueError("degradation_thresholds must be a JSON object")
                higher_is_better = self._metric_name_list(
                    body.get("higher_is_better_metrics")
                )
                result = self.orchestrator.auto_trigger_retraining(
                    model_name=str(body["model_name"]),
                    candidate_version=str(body["candidate_version"]),
                    operation=str(body["operation"]),
                    parameters=parameters,
                    corrected_frames_count=int(body["corrected_frames_count"]),
                    corrected_frames_threshold=int(
                        body.get("corrected_frames_threshold", 25)
                    ),
                    degradation_thresholds={
                        str(name): float(value)
                        for name, value in degradation_thresholds.items()
                    },
                    higher_is_better_metrics=higher_is_better,
                    baseline_version=self._optional_string(body.get("baseline_version")),
                    baseline_stage=str(body.get("baseline_stage", "production")),
                    force=bool(body.get("force", False)),
                    metadata_out=self._optional_string(body.get("metadata_out")),
                    max_retries=int(body.get("max_retries", 0)),
                    run_immediately=bool(body.get("run_immediately", False)),
                    scheduler=str(body.get("scheduler", "local")),
                )
                self._write_json(200, asdict(result))
                return
            if path == "/register-model-version":
                metrics = body.get("metrics", {})
                metadata = body.get("metadata", {})
                if not isinstance(metrics, dict):
                    raise ValueError("metrics must be a JSON object")
                if not isinstance(metadata, dict):
                    raise ValueError("metadata must be a JSON object")
                record = self.orchestrator.register_model_version(
                    str(body["model_name"]),
                    str(body["artifact_path"]),
                    version=self._optional_string(body.get("version")),
                    metrics=metrics,
                    metadata=metadata,
                )
                self._write_json(200, asdict(record))
                return
            if path == "/promote-model-version":
                promotion = self.orchestrator.promote_model_version(
                    str(body["model_name"]),
                    str(body["version"]),
                    stage=str(body.get("stage", "production")),
                    notes=self._optional_string(body.get("notes")),
                )
                self._write_json(200, asdict(promotion))
                return
            self._write_json(404, {"error": f"unknown path: {path}"})
        except KeyError as exc:
            self._write_json(400, {"error": f"missing required field: {exc.args[0]}"})
        except ValueError as exc:
            self._write_json(400, {"error": str(exc)})
        except Exception as exc:  # pragma: no cover - defensive handler
            self._write_json(500, {"error": str(exc)})

    def _workspace_root(self) -> Path:
        root = self.orchestrator.store.jobs_root.parent
        root.mkdir(parents=True, exist_ok=True)
        return root

    def _required_string(self, body: dict[str, Any], key: str) -> str:
        value = self._optional_string(body.get(key))
        if value is None:
            raise ValueError(f"{key} is required")
        return value

    def _parse_selected_frames(self, value: Any) -> list[int]:
        if value is None:
            return []
        if isinstance(value, list):
            out: list[int] = []
            for item in value:
                if isinstance(item, (int, float, str)):
                    out.append(int(item))
            return sorted(set(out))
        if isinstance(value, str):
            out: list[int] = []
            for item in value.split(","):
                text = item.strip()
                if text:
                    out.append(int(text))
            return sorted(set(out))
        return []

    def _read_json_dict(self) -> dict[str, Any]:
        content_length = int(self.headers.get("Content-Length", "0"))
        if content_length <= 0:
            return {}
        raw = self.rfile.read(content_length)
        payload = json.loads(raw.decode("utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("request body must be a JSON object")
        return payload

    def _query_first(self, query: dict[str, list[str]], key: str) -> str | None:
        values = query.get(key)
        if not values:
            return None
        value = str(values[0]).strip()
        if not value:
            return None
        return value

    def _query_required(self, query: dict[str, list[str]], key: str) -> str:
        value = self._query_first(query, key)
        if value is None:
            raise ValueError(f"query parameter '{key}' is required")
        return value

    def _optional_string(self, value: Any) -> str | None:
        if value is None:
            return None
        value_str = str(value).strip()
        if not value_str:
            return None
        return value_str

    def _metric_name_list(self, value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            return [item.strip() for item in value.split(",") if item.strip()]
        if isinstance(value, list):
            out: list[str] = []
            for item in value:
                text = str(item).strip()
                if text:
                    out.append(text)
            return out
        raise ValueError("higher_is_better_metrics must be a list or comma string")

    def _write_json(self, status_code: int, payload: Any) -> None:
        encoded = json.dumps(payload, indent=2).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def _write_text(self, status_code: int, payload: str, *, content_type: str) -> None:
        encoded = payload.encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)


def create_api_server(
    *,
    host: str = "127.0.0.1",
    port: int = 8765,
    jobs_root: str = ".xrommtools/jobs",
    registry_path: str = ".xrommtools/model_registry.json",
) -> OrchestratorHttpServer:
    orchestrator = LocalPipelineOrchestrator(
        store=LocalJobStore(jobs_root),
        model_registry=ModelRegistry(registry_path),
    )
    return OrchestratorHttpServer((host, int(port)), _JsonApiHandler, orchestrator)


def serve_api(
    *,
    host: str = "127.0.0.1",
    port: int = 8765,
    jobs_root: str = ".xrommtools/jobs",
    registry_path: str = ".xrommtools/model_registry.json",
) -> None:
    server = create_api_server(
        host=host,
        port=port,
        jobs_root=jobs_root,
        registry_path=registry_path,
    )
    with server:
        server.serve_forever()
