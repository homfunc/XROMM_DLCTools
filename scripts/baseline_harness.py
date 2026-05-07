#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import importlib
import json
import random
import shutil
import subprocess
import sys
import threading
import time
import traceback
import types
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable
from urllib.parse import quote
from urllib.request import Request, urlopen


class ScenarioSkipped(Exception):
    pass


@dataclass
class HarnessContext:
    repo_root: Path
    output_root: Path
    deeplabcut_repo: Path
    xmalab_repo: Path


def _load_xrommtools(repo_root: Path):
    predict_module_name = "deeplabcut.pose_estimation_tensorflow.predict_videos"
    has_predict_module = predict_module_name in sys.modules
    if not has_predict_module:
        try:
            importlib.import_module(predict_module_name)
            has_predict_module = True
        except Exception:
            has_predict_module = False

    if not has_predict_module:
        deeplabcut_mod = types.ModuleType("deeplabcut")
        pose_mod = types.ModuleType("deeplabcut.pose_estimation_tensorflow")
        predict_mod = types.ModuleType(predict_module_name)

        def analyze_videos(*_args, **_kwargs):
            raise RuntimeError(
                "DeepLabCut not installed; analyze_videos is unavailable in stub mode."
            )

        predict_mod.analyze_videos = analyze_videos
        deeplabcut_mod.pose_estimation_tensorflow = pose_mod
        pose_mod.predict_videos = predict_mod
        if "deeplabcut" not in sys.modules:
            sys.modules["deeplabcut"] = deeplabcut_mod
        else:
            sys.modules["deeplabcut"].pose_estimation_tensorflow = pose_mod
        sys.modules["deeplabcut.pose_estimation_tensorflow"] = pose_mod
        sys.modules[predict_module_name] = predict_mod

    module_path = repo_root / "functions" / "xrommtools.py"
    spec = importlib.util.spec_from_file_location("xrommtools_runtime", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module spec for {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run_cmd(cmd: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        text=True,
        capture_output=True,
    )


def scenario_xmalab_csv_contract(ctx: HarnessContext) -> dict[str, Any]:
    trial_cpp = ctx.xmalab_repo / "src" / "core" / "Trial.cpp"
    if not trial_cpp.exists():
        raise ScenarioSkipped(f"XMALab source not found at {trial_cpp}")

    src = trial_cpp.read_text()
    required_snippets = [
        "int Trial::load2dPoints(QString input",
        '_cam" << j + 1 << "_X"',
        'outfile << "NaN"',
    ]
    missing = [snippet for snippet in required_snippets if snippet not in src]
    if missing:
        raise AssertionError(f"XMALab CSV contract snippets missing: {missing}")

    out_dir = ctx.output_root / "xmalab_csv_contract"
    out_dir.mkdir(parents=True, exist_ok=True)
    sample_csv = out_dir / "sample_xmalab_2dpoints.csv"

    header = [
        "marker001_cam1_X",
        "marker001_cam1_Y",
        "marker001_cam2_X",
        "marker001_cam2_Y",
        "marker002_cam1_X",
        "marker002_cam1_Y",
        "marker002_cam2_X",
        "marker002_cam2_Y",
    ]
    rows = [
        ["10.0", "20.0", "11.0", "21.0", "30.0", "40.0", "31.0", "41.0"],
        ["12.0", "22.0", "13.0", "23.0", "NaN", "NaN", "33.0", "43.0"],
        ["14.0", "24.0", "15.0", "25.0", "34.0", "44.0", "35.0", "45.0"],
    ]
    sample_csv.write_text(
        ",".join(header) + "\n" + "\n".join(",".join(row) for row in rows) + "\n"
    )

    extracted_names = [header[i][:-7] for i in range(0, len(header), 4)]
    if extracted_names != ["marker001", "marker002"]:
        raise AssertionError(
            f"Point-name extraction mismatch from XMALab-style header: {extracted_names}"
        )

    return {
        "trial_cpp": str(trial_cpp),
        "sample_csv": str(sample_csv),
        "markers_detected": extracted_names,
        "rows": len(rows),
    }


def scenario_xrommtools_dlc_to_xma_contract(ctx: HarnessContext) -> dict[str, Any]:
    import numpy as np
    import pandas as pd

    xrommtools = _load_xrommtools(ctx.repo_root)
    out_dir = ctx.output_root / "xrommtools_dlc_to_xma_contract"
    out_dir.mkdir(parents=True, exist_ok=True)

    scorer = "baseline_scorer"
    bodyparts = ["marker001", "marker002"]
    cols = pd.MultiIndex.from_product(
        [[scorer], bodyparts, ["x", "y", "likelihood"]],
        names=["scorer", "bodyparts", "coords"],
    )
    cam1 = pd.DataFrame(
        [
            [10.0, 20.0, 0.99, 30.0, 40.0, 0.98],
            [11.0, 21.0, 0.97, np.nan, np.nan, 0.10],
            [12.0, 22.0, 0.96, 32.0, 42.0, 0.95],
        ],
        columns=cols,
    )
    cam2 = pd.DataFrame(
        [
            [13.0, 23.0, 0.88, 33.0, 43.0, 0.87],
            [14.0, 24.0, 0.86, 34.0, 44.0, 0.85],
            [15.0, 25.0, 0.84, 35.0, 45.0, 0.83],
        ],
        columns=cols,
    )

    trial = "baseline_trial"
    xrommtools.dlc_to_xma(cam1, cam2, trial, str(out_dir))
    csv_path = out_dir / f"{trial}-Predicted2DPoints.csv"
    h5_path = out_dir / f"{trial}-Predicted2DPoints.h5"
    if not csv_path.exists() or not h5_path.exists():
        raise AssertionError("Expected dlc_to_xma output files were not created")

    out_df = pd.read_csv(csv_path)
    expected_cols = [
        "marker001_cam1_X",
        "marker001_cam1_Y",
        "marker001_cam2_X",
        "marker001_cam2_Y",
        "marker002_cam1_X",
        "marker002_cam1_Y",
        "marker002_cam2_X",
        "marker002_cam2_Y",
    ]
    if list(out_df.columns) != expected_cols:
        raise AssertionError(
            f"Unexpected output columns from dlc_to_xma: {list(out_df.columns)}"
        )
    if out_df.shape != (3, 8):
        raise AssertionError(f"Unexpected output shape from dlc_to_xma: {out_df.shape}")

    return {
        "csv_path": str(csv_path),
        "h5_path": str(h5_path),
        "rows": int(out_df.shape[0]),
        "columns": expected_cols,
    }


def scenario_xrommtools_xma_to_dlc_fixture(ctx: HarnessContext) -> dict[str, Any]:
    import pandas as pd

    xrommtools = _load_xrommtools(ctx.repo_root)
    trainingdata_path = ctx.repo_root / "templates" / "trainingdata"
    if not trainingdata_path.exists():
        raise AssertionError(f"Missing fixture data at {trainingdata_path}")

    out_dir = ctx.output_root / "xrommtools_xma_to_dlc_fixture"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    project_root = out_dir / "dlc_project"
    project_root.mkdir(parents=True, exist_ok=True)
    config_path = project_root / "config.yaml"
    config_path.write_text("# baseline harness config placeholder\n")

    random.seed(0)
    xrommtools.xma_to_dlc(
        str(config_path),
        str(trainingdata_path),
        "baseline_dataset",
        "baseline",
        4,
    )

    dataset_dir = project_root / "labeled-data" / "baseline_dataset"
    csv_path = dataset_dir / "CollectedData_baseline.csv"
    h5_path = dataset_dir / "CollectedData_baseline.h5"
    if not csv_path.exists() or not h5_path.exists():
        raise AssertionError("Expected xma_to_dlc output files were not created")

    df = pd.read_csv(csv_path, header=[0, 1, 2], index_col=0)
    png_count = len(list(dataset_dir.glob("*.png")))
    if png_count != 8:
        raise AssertionError(f"Expected 8 extracted images, found {png_count}")
    if df.shape[0] != 8:
        raise AssertionError(f"Expected 8 label rows, found {df.shape[0]}")
    if df.shape[1] % 2 != 0:
        raise AssertionError(f"Expected even coordinate columns, found {df.shape[1]}")

    return {
        "dataset_dir": str(dataset_dir),
        "images_extracted": png_count,
        "label_rows": int(df.shape[0]),
        "label_columns": int(df.shape[1]),
    }


def scenario_deeplabcut_repo_smoke(ctx: HarnessContext) -> dict[str, Any]:
    if not ctx.deeplabcut_repo.exists():
        raise ScenarioSkipped(f"DeepLabCut repository not found at {ctx.deeplabcut_repo}")

    required = [
        ctx.deeplabcut_repo / "examples" / "utils.py",
        ctx.deeplabcut_repo / "examples" / "testscript_pytorch_single_animal.py",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise AssertionError(f"DeepLabCut smoke prerequisites missing: {missing}")

    import_check = _run_cmd(
        [
            sys.executable,
            "-c",
            (
                "import sys; "
                f"sys.path.insert(0, {str(ctx.deeplabcut_repo)!r}); "
                "import deeplabcut; "
                "print(deeplabcut.__file__)"
            ),
        ]
    )
    if import_check.returncode != 0:
        raise ScenarioSkipped(
            "DeepLabCut import failed in this env; run with DeepLabCut deps installed."
        )

    smoke_script = (
        "from pathlib import Path\n"
        "import sys\n"
        f"repo = Path({str(ctx.deeplabcut_repo)!r})\n"
        "sys.path.insert(0, str(repo))\n"
        "sys.path.insert(0, str(repo / 'examples'))\n"
        "from utils import SyntheticProjectParameters, cleanup, create_fake_project\n"
        "project = repo / 'tmp_baseline_harness_project'\n"
        "if project.exists():\n"
        "    cleanup(project)\n"
        "create_fake_project(\n"
        "    path=project,\n"
        "    params=SyntheticProjectParameters(\n"
        "        multianimal=False,\n"
        "        num_bodyparts=3,\n"
        "        num_frames=8,\n"
        "        num_individuals=1,\n"
        "        num_unique=0,\n"
        "        frame_shape=(64, 64),\n"
        "    ),\n"
        ")\n"
        "cleanup(project)\n"
        "print('deeplabcut-smoke-ok')\n"
    )
    smoke = _run_cmd([sys.executable, "-c", smoke_script])
    if smoke.returncode != 0:
        raise AssertionError(
            "DeepLabCut synthetic project smoke failed:\n"
            f"{smoke.stdout}\n{smoke.stderr}"
        )

    return {
        "deeplabcut_import": import_check.stdout.strip(),
        "smoke_output": smoke.stdout.strip(),
    }


def _request_json(
    base_url: str,
    method: str,
    path: str,
    payload: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
):
    data = None
    request_headers = dict(headers or {})
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        request_headers["Content-Type"] = "application/json"
    request = Request(base_url + path, data=data, method=method, headers=request_headers)
    with urlopen(request, timeout=20) as response:
        return response.status, json.loads(response.read().decode("utf-8"))

def scenario_phase3_local_workflow_e2e(ctx: HarnessContext) -> dict[str, Any]:
    import base64

    import pandas as pd

    deeplabcut_details = scenario_deeplabcut_repo_smoke(ctx)
    xmalab_details = scenario_xmalab_csv_contract(ctx)

    if str(ctx.repo_root) not in sys.path:
        sys.path.insert(0, str(ctx.repo_root))

    from xrommtools.server import create_api_server

    out_dir = ctx.output_root / "phase3_local_workflow_e2e"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    jobs_root = out_dir / ".xrommtools" / "jobs"
    registry_path = out_dir / ".xrommtools" / "model_registry.json"
    project_root = out_dir / "dlc_project"
    project_root.mkdir(parents=True, exist_ok=True)
    config_path = project_root / "config.yaml"
    config_path.write_text("# e2e baseline harness config placeholder\n")

    server = create_api_server(
        host="127.0.0.1",
        port=0,
        jobs_root=str(jobs_root),
        registry_path=str(registry_path),
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{server.server_address[1]}"

    try:
        upload_status, upload_result = _request_json(
            base_url,
            "POST",
            "/upload-file",
            {
                "filename": "phase3-note.txt",
                "content_base64": base64.b64encode(
                    b"phase3 local workflow integration\n"
                ).decode("ascii"),
                "subdir": "phase3_e2e",
            },
        )
        if upload_status != 200:
            raise AssertionError(f"upload-file did not return 200: {upload_status}")

        _, blueprint = _request_json(base_url, "GET", "/workflow-blueprints")
        if len(blueprint.get("steps", [])) < 5:
            raise AssertionError(f"Unexpected workflow blueprint: {blueprint}")

        trainingdata_path = ctx.repo_root / "templates" / "trainingdata"
        ingest_status, ingest = _request_json(
            base_url,
            "POST",
            "/run-operation",
            {
                "operation": "xma_to_dlc",
                "parameters": {
                    "path_config_file": str(config_path),
                    "data_path": str(trainingdata_path),
                    "dataset_name": "e2e_dataset",
                    "scorer": "e2e",
                    "nframes": 4,
                    "nnetworks": 1,
                    "path_config_file_cam2": [],
                },
                "scheduler": "local",
                "run_immediately": True,
            },
        )
        if ingest_status != 200 or ingest.get("status") != "succeeded":
            raise AssertionError(f"Ingest step failed: {ingest}")

        operation_run = ingest.get("operation_run") or {}
        declared_outputs = operation_run.get("declared_outputs") or {}
        labeled_csv = Path(declared_outputs["csv"])
        if not labeled_csv.exists():
            raise AssertionError(f"Expected labeled CSV missing: {labeled_csv}")
        labels = pd.read_csv(labeled_csv, header=[0, 1, 2], index_col=0)
        bodyparts = [
            item
            for item in labels.columns.get_level_values(1).unique().tolist()
            if item != "bodyparts"
        ]
        if not bodyparts:
            raise AssertionError("No bodyparts detected from xma_to_dlc output")

        scorer = "e2e_pred"
        cols = pd.MultiIndex.from_product(
            [[scorer], bodyparts, ["x", "y", "likelihood"]],
            names=["scorer", "bodyparts", "coords"],
        )
        rows_cam1: list[list[float]] = []
        rows_cam2: list[list[float]] = []
        for frame_idx in range(3):
            row_cam1: list[float] = []
            row_cam2: list[float] = []
            for part_idx, _ in enumerate(bodyparts):
                base = 10.0 + part_idx * 7.0 + frame_idx
                row_cam1.extend([base, base + 0.5, 0.99])
                row_cam2.extend([base + 1.0, base + 1.5, 0.98])
            rows_cam1.append(row_cam1)
            rows_cam2.append(row_cam2)
        cam1_df = pd.DataFrame(rows_cam1, columns=cols)
        cam2_df = pd.DataFrame(rows_cam2, columns=cols)

        export_inputs_dir = out_dir / "export_inputs"
        export_inputs_dir.mkdir(parents=True, exist_ok=True)
        cam1_h5 = export_inputs_dir / "cam1_predictions.h5"
        cam2_h5 = export_inputs_dir / "cam2_predictions.h5"
        cam1_df.to_hdf(cam1_h5, key="df_with_missing", mode="w")
        cam2_df.to_hdf(cam2_h5, key="df_with_missing", mode="w")

        review_report_path = out_dir / "synthetic_review_report.json"
        review_report_path.write_text(
            json.dumps(
                {
                    "optimization": {
                        "top_review_frames": [
                            {"frame": 7, "score": 0.91},
                            {"frame": 11, "score": 0.88},
                        ]
                    }
                },
                indent=2,
            )
        )
        _, review_report = _request_json(
            base_url,
            "GET",
            "/review-report?path=" + quote(str(review_report_path), safe=""),
        )
        review_frames = review_report.get("review_frame_numbers") or []
        if review_frames[:2] != [7, 11]:
            raise AssertionError(f"Unexpected review frames extracted: {review_report}")

        _, retrain = _request_json(
            base_url,
            "POST",
            "/review-to-retrain",
            {
                "trial_name": "trial_001",
                "selected_frames": review_frames,
                "path_config_file": str(config_path),
                "data_path": str(trainingdata_path),
                "iteration": 1,
                "scheduler": "hpc-stub",
                "run_immediately": False,
            },
        )
        retrain_job = retrain.get("job") or {}
        if retrain_job.get("status") != "queued_remote":
            raise AssertionError(f"Unexpected retrain handoff result: {retrain}")
        frames_csv_path = Path(retrain["frames_csv_path"])
        if not frames_csv_path.exists():
            raise AssertionError(f"Expected review frames CSV missing: {frames_csv_path}")

        export_dir = out_dir / "export_outputs"
        export_dir.mkdir(parents=True, exist_ok=True)
        _, export = _request_json(
            base_url,
            "POST",
            "/one-click-export",
            {
                "parameters": {
                    "cam1data": str(cam1_h5),
                    "cam2data": str(cam2_h5),
                    "trialname": "e2e_trial",
                    "savepath": str(export_dir),
                },
                "scheduler": "local",
                "run_immediately": True,
                "workflow_context": {
                    "ingest_job_id": ingest["job_id"],
                    "retrain_job_id": retrain_job.get("job_id"),
                },
            },
        )
        summary_path = Path(export["summary_path"])
        if not summary_path.exists():
            raise AssertionError(f"Export summary artifact missing: {summary_path}")
        export_job = export["summary"]["export_job"]
        if export_job.get("status") != "succeeded":
            raise AssertionError(f"Export step failed: {export_job}")
        export_run = export_job.get("operation_run") or {}
        export_outputs = export_run.get("declared_outputs") or {}
        exported_csv = Path(export_outputs["csv"])
        if not exported_csv.exists():
            raise AssertionError(f"Expected exported CSV missing: {exported_csv}")
        exported_df = pd.read_csv(exported_csv)

        expected_prefix_cols = [
            f"{bodyparts[0]}_cam1_X",
            f"{bodyparts[0]}_cam1_Y",
            f"{bodyparts[0]}_cam2_X",
            f"{bodyparts[0]}_cam2_Y",
        ]
        if list(exported_df.columns[:4]) != expected_prefix_cols:
            raise AssertionError(
                f"Unexpected exported CSV columns: {list(exported_df.columns[:8])}"
            )

        _, jobs = _request_json(base_url, "GET", "/jobs?limit=20")
        if len(jobs) < 3:
            raise AssertionError(f"Expected at least 3 workflow jobs, found: {jobs}")

        return {
            "deeplabcut_smoke": deeplabcut_details,
            "xmalab_contract": {
                "trial_cpp": xmalab_details["trial_cpp"],
                "markers_detected": xmalab_details["markers_detected"],
            },
            "upload_saved_path": upload_result["saved_path"],
            "workflow_blueprint_steps": [
                step["title"] for step in blueprint.get("steps", [])
            ],
            "ingest_job_id": ingest["job_id"],
            "labeled_csv": str(labeled_csv),
            "bodyparts_detected": bodyparts,
            "review_report_path": str(review_report_path),
            "review_frames": review_frames,
            "retrain_frames_csv": str(frames_csv_path),
            "retrain_job_id": retrain_job.get("job_id"),
            "export_summary_path": str(summary_path),
            "exported_csv": str(exported_csv),
            "exported_rows": int(exported_df.shape[0]),
            "exported_columns_sample": list(exported_df.columns[:8]),
            "jobs_observed": len(jobs),
        }
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


SCENARIOS: dict[str, Callable[[HarnessContext], dict[str, Any]]] = {
    "xmalab_csv_contract": scenario_xmalab_csv_contract,
    "xrommtools_dlc_to_xma_contract": scenario_xrommtools_dlc_to_xma_contract,
    "xrommtools_xma_to_dlc_fixture": scenario_xrommtools_xma_to_dlc_fixture,
    "deeplabcut_repo_smoke": scenario_deeplabcut_repo_smoke,
    "phase3_local_workflow_e2e": scenario_phase3_local_workflow_e2e,
}

SCENARIO_GROUPS: dict[str, list[str]] = {
    "ci": [
        "xrommtools_dlc_to_xma_contract",
        "xrommtools_xma_to_dlc_fixture",
    ],
    "core": [
        "xmalab_csv_contract",
        "xrommtools_dlc_to_xma_contract",
        "xrommtools_xma_to_dlc_fixture",
    ],
    "all": [
        "xmalab_csv_contract",
        "xrommtools_dlc_to_xma_contract",
        "xrommtools_xma_to_dlc_fixture",
        "deeplabcut_repo_smoke",
        "phase3_local_workflow_e2e",
    ],
}


def _run_one(name: str, fn: Callable[[HarnessContext], dict[str, Any]], ctx: HarnessContext):
    start = time.time()
    try:
        details = fn(ctx)
        status = "passed"
        error = None
    except ScenarioSkipped as exc:
        details = {}
        status = "skipped"
        error = str(exc)
    except Exception as exc:
        details = {}
        status = "failed"
        error = f"{exc}\n{traceback.format_exc()}"
    duration = round(time.time() - start, 3)
    print(f"[{status.upper():7}] {name} ({duration:.3f}s)")
    if error and status != "failed":
        print(f"           {error}")
    return {
        "name": name,
        "status": status,
        "duration_seconds": duration,
        "details": details,
        "error": error,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run baseline regression scenarios for XROMM_DLCTools + adjacent repos."
    )
    parser.add_argument(
        "--scenario",
        default="core",
        choices=["ci", "core", "all", *SCENARIOS.keys()],
        help="Scenario group (core/all) or a single scenario.",
    )
    parser.add_argument(
        "--output-dir",
        default="baseline_artifacts/latest",
        help="Directory where baseline artifacts and summary JSON are written.",
    )
    parser.add_argument(
        "--deeplabcut-repo",
        default="../DeepLabCut",
        help="Path to local DeepLabCut repository.",
    )
    parser.add_argument(
        "--xmalab-repo",
        default="../xmalab",
        help="Path to local xmalab repository.",
    )
    parser.add_argument(
        "--fail-on-skip",
        action="store_true",
        help="Return non-zero if any selected scenario is skipped.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    output_root = (repo_root / args.output_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    ctx = HarnessContext(
        repo_root=repo_root,
        output_root=output_root,
        deeplabcut_repo=(repo_root / args.deeplabcut_repo).resolve(),
        xmalab_repo=(repo_root / args.xmalab_repo).resolve(),
    )

    selected = SCENARIO_GROUPS.get(args.scenario, [args.scenario])
    run_results = [_run_one(name, SCENARIOS[name], ctx) for name in selected]

    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "repo_root": str(repo_root),
        "output_root": str(output_root),
        "requested_scenario": args.scenario,
        "selected_scenarios": selected,
        "results": run_results,
        "counts": {
            "passed": sum(1 for r in run_results if r["status"] == "passed"),
            "failed": sum(1 for r in run_results if r["status"] == "failed"),
            "skipped": sum(1 for r in run_results if r["status"] == "skipped"),
        },
    }

    summary_path = output_root / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"Summary written to: {summary_path}")
    if summary["counts"]["failed"] > 0:
        return 1
    if args.fail_on_skip and summary["counts"]["skipped"] > 0:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
