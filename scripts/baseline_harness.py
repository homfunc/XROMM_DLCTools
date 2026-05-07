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
import time
import traceback
import types
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable
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

SCENARIOS: dict[str, Callable[[HarnessContext], dict[str, Any]]] = {
    "xmalab_csv_contract": scenario_xmalab_csv_contract,
    "xrommtools_dlc_to_xma_contract": scenario_xrommtools_dlc_to_xma_contract,
    "xrommtools_xma_to_dlc_fixture": scenario_xrommtools_xma_to_dlc_fixture,
    "deeplabcut_repo_smoke": scenario_deeplabcut_repo_smoke,
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
