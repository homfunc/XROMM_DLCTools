#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _result_map(summary: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {r["name"]: r for r in summary["results"]}


def _build_manifest(summary: dict[str, Any], version: str) -> dict[str, Any]:
    result_by_name = _result_map(summary)
    selected = summary["selected_scenarios"]
    statuses = {name: result_by_name[name]["status"] for name in selected}

    checksums: dict[str, str] = {}
    invariants: dict[str, Any] = {}

    if "xmalab_csv_contract" in result_by_name and statuses.get("xmalab_csv_contract") == "passed":
        sample_csv = Path(result_by_name["xmalab_csv_contract"]["details"]["sample_csv"])
        checksums["xmalab_csv_contract.sample_csv.sha256"] = _sha256(sample_csv)
        invariants["xmalab_csv_contract.markers_detected"] = result_by_name[
            "xmalab_csv_contract"
        ]["details"]["markers_detected"]
        invariants["xmalab_csv_contract.rows"] = result_by_name["xmalab_csv_contract"]["details"][
            "rows"
        ]

    if (
        "xrommtools_dlc_to_xma_contract" in result_by_name
        and statuses.get("xrommtools_dlc_to_xma_contract") == "passed"
    ):
        csv_path = Path(result_by_name["xrommtools_dlc_to_xma_contract"]["details"]["csv_path"])
        checksums["xrommtools_dlc_to_xma_contract.csv.sha256"] = _sha256(csv_path)
        invariants["xrommtools_dlc_to_xma_contract.rows"] = result_by_name[
            "xrommtools_dlc_to_xma_contract"
        ]["details"]["rows"]
        invariants["xrommtools_dlc_to_xma_contract.columns"] = result_by_name[
            "xrommtools_dlc_to_xma_contract"
        ]["details"]["columns"]

    if (
        "xrommtools_xma_to_dlc_fixture" in result_by_name
        and statuses.get("xrommtools_xma_to_dlc_fixture") == "passed"
    ):
        dataset_dir = Path(result_by_name["xrommtools_xma_to_dlc_fixture"]["details"]["dataset_dir"])
        csv_path = dataset_dir / "CollectedData_baseline.csv"
        checksums["xrommtools_xma_to_dlc_fixture.collected_data_csv.sha256"] = _sha256(csv_path)
        invariants["xrommtools_xma_to_dlc_fixture.images_extracted"] = result_by_name[
            "xrommtools_xma_to_dlc_fixture"
        ]["details"]["images_extracted"]
        invariants["xrommtools_xma_to_dlc_fixture.label_rows"] = result_by_name[
            "xrommtools_xma_to_dlc_fixture"
        ]["details"]["label_rows"]
        invariants["xrommtools_xma_to_dlc_fixture.label_columns"] = result_by_name[
            "xrommtools_xma_to_dlc_fixture"
        ]["details"]["label_columns"]

    if "deeplabcut_repo_smoke" in result_by_name and statuses.get("deeplabcut_repo_smoke") == "passed":
        invariants["deeplabcut_repo_smoke.smoke_output"] = result_by_name["deeplabcut_repo_smoke"][
            "details"
        ]["smoke_output"]

    return {
        "manifest_version": version,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "requested_scenario": summary["requested_scenario"],
        "selected_scenarios": selected,
        "statuses": statuses,
        "checksums": checksums,
        "invariants": invariants,
    }


def _verify(summary: dict[str, Any], manifest: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    result_by_name = _result_map(summary)

    if summary["selected_scenarios"] != manifest["selected_scenarios"]:
        errors.append(
            "Selected scenario mismatch: "
            f"{summary['selected_scenarios']} != {manifest['selected_scenarios']}"
        )

    for name, expected_status in manifest["statuses"].items():
        actual = result_by_name.get(name, {}).get("status")
        if actual != expected_status:
            errors.append(f"Status mismatch for {name}: expected {expected_status}, got {actual}")

    rebuilt = _build_manifest(summary, manifest["manifest_version"])
    for key, expected in manifest["checksums"].items():
        actual = rebuilt["checksums"].get(key)
        if actual != expected:
            errors.append(f"Checksum mismatch for {key}: expected {expected}, got {actual}")

    for key, expected in manifest["invariants"].items():
        actual = rebuilt["invariants"].get(key)
        if actual != expected:
            errors.append(f"Invariant mismatch for {key}: expected {expected}, got {actual}")

    return errors


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Freeze/verify golden baseline manifests.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    freeze = subparsers.add_parser("freeze", help="Create a golden manifest from a summary.")
    freeze.add_argument("--summary", required=True, help="Path to baseline summary JSON.")
    freeze.add_argument("--out", required=True, help="Path to write manifest JSON.")
    freeze.add_argument("--version", default="v1", help="Manifest version label.")

    verify = subparsers.add_parser("verify", help="Verify a summary against a manifest.")
    verify.add_argument("--summary", required=True, help="Path to baseline summary JSON.")
    verify.add_argument("--manifest", required=True, help="Path to golden manifest JSON.")

    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.command == "freeze":
        summary_path = Path(args.summary).resolve()
        out_path = Path(args.out).resolve()
        summary = _read_json(summary_path)
        manifest = _build_manifest(summary, args.version)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(manifest, indent=2))
        print(f"Wrote manifest: {out_path}")
        return 0

    summary = _read_json(Path(args.summary).resolve())
    manifest = _read_json(Path(args.manifest).resolve())
    errors = _verify(summary, manifest)
    if errors:
        print("Baseline manifest verification failed:")
        for err in errors:
            print(f"- {err}")
        return 1
    print("Baseline manifest verification passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
