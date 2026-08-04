from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Any

import pytest


runner = importlib.import_module(
    "Standalone_Scripts.ACR.run_bca20_followup_pipeline"
)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_full_runner_wires_all_stages_and_writes_root_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project_root = tmp_path / "ACR project"
    project_root.mkdir()
    (project_root / "project.json").write_text("{}", encoding="utf-8")
    roi_config = tmp_path / "rois.json"
    roi_config.write_text("{}", encoding="utf-8")
    output = tmp_path / "output"
    calls: dict[str, dict[str, Any]] = {}

    def fake_aggregate(**kwargs: Any) -> dict[str, Any]:
        calls["aggregation"] = kwargs
        stage = kwargs["output_dir"]
        stage.mkdir(parents=True)
        (stage / "configured_roi_bca20_long.csv").write_text(
            "subject,group,condition,roi,raw,mean_norm,rms_norm\n",
            encoding="utf-8",
        )
        manifest = {
            "aggregation_counts": {
                "participants": 34,
                "group_participant_counts": {
                    "anxious": 18,
                    "non_anxious": 16,
                },
            },
            "included_conditions": ["Neutral Sad"],
            "exclusions": {"effective_full_participants": ["P20"]},
            "harmonic_definition": {"included_orders": [1, 2, 3]},
        }
        _write_json(stage / "aggregation_manifest.json", manifest)
        return manifest

    def fake_pi(**kwargs: Any) -> dict[str, Any]:
        calls["pi"] = kwargs
        stage = kwargs["output_dir"]
        manifest = {
            "analysis_version": "test_pi",
            "analysis_success": True,
            "input": {
                "participant_counts": {
                    "anxious": 18,
                    "non_anxious": 16,
                }
            },
            "outputs": {"condition_specific_lmm_tests.csv": {}},
            "required_model_status": {
                "analysis_success": True,
                "required_models": 1,
                "failed_models": 0,
                "nonconverged_models": 0,
                "families": {},
            },
        }
        _write_json(stage / "analysis_manifest.json", manifest)
        return manifest

    def fake_sad(**kwargs: Any) -> dict[str, Any]:
        calls["sad"] = kwargs
        stage = kwargs["output_dir"]
        manifest = {
            "target_condition": kwargs["target_condition"],
            "other_conditions": ["Neutral Angry"],
            "shared_other_conditions": ["Neutral Angry"],
            "group_participant_counts": {
                "anxious": 18,
                "non_anxious": 16,
            },
        }
        _write_json(stage / "analysis_manifest.json", manifest)
        return manifest

    monkeypatch.setattr(runner, "aggregate_bca20_followup", fake_aggregate)
    monkeypatch.setattr(runner, "analyze_bca20_pi_followup", fake_pi)
    monkeypatch.setattr(runner, "analyze_sad_uniqueness", fake_sad)

    manifest = runner.run_bca20_followup_pipeline(
        project_root,
        output,
        roi_config_path=roi_config,
        excluded_subjects=("P99",),
        influence_subjects=("P27", "P33"),
    )

    assert manifest["pipeline_version"] == runner.PIPELINE_VERSION
    assert calls["aggregation"]["excluded_subjects"] == ("P99",)
    assert calls["pi"]["configured_roi_path"] == (
        output / "01_bca20_aggregation" / "configured_roi_bca20_long.csv"
    )
    assert calls["pi"]["excluded_subjects"] == ()
    assert calls["sad"]["influence_subjects"] == ("P27", "P33")
    assert set(manifest["stage_manifests"]) == {
        "aggregation",
        "pi_followup",
        "sad_uniqueness",
    }
    assert all(
        len(receipt["sha256"]) == 64
        for receipt in manifest["stage_manifests"].values()
    )
    saved = json.loads(
        (output / "pipeline_manifest.json").read_text(encoding="utf-8")
    )
    assert saved["configuration"]["influence_subjects"] == ["P27", "P33"]
    assert "analysis_contract" in saved["code"]
    assert "scipy" in saved["software_versions"]


def test_runner_refuses_a_nonempty_output_without_explicit_override(
    tmp_path: Path,
) -> None:
    project_root = tmp_path / "project"
    project_root.mkdir()
    (project_root / "project.json").write_text("{}", encoding="utf-8")
    roi_config = tmp_path / "rois.json"
    roi_config.write_text("{}", encoding="utf-8")
    output = tmp_path / "output"
    output.mkdir()
    (output / "unrelated.txt").write_text("keep", encoding="utf-8")

    with pytest.raises(FileExistsError, match="not empty"):
        runner.run_bca20_followup_pipeline(
            project_root,
            output,
            roi_config_path=roi_config,
        )

    assert (output / "unrelated.txt").read_text(encoding="utf-8") == "keep"
