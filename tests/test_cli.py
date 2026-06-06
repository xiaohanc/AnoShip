import json

import pytest
from anoship.app.cli import build_parser, main


def test_list_command(capsys):
    rc = main(["list"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "detectors:" in out
    assert "diffusion" in out
    assert "scenarios:" in out


def test_run_healthy(capsys):
    rc = main(["run", "--scenario", "healthy", "--detector", "diffusion", "--quiet"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "COMPLETED" in out


def test_run_regression_rollback(capsys):
    rc = main(
        [
            "run",
            "--scenario",
            "regression",
            "--detector",
            "diffusion",
            "--risk-tier",
            "high_impact",
            "--quiet",
        ]
    )
    out = capsys.readouterr().out
    assert rc == 0
    assert "ROLLED BACK" in out


def test_run_writes_json(tmp_path, capsys):
    out_path = tmp_path / "events.json"
    main(
        [
            "run",
            "--scenario",
            "regression",
            "--detector",
            "diffusion",
            "--quiet",
            "--json",
            str(out_path),
        ]
    )
    assert out_path.exists()
    records = json.loads(out_path.read_text())
    kinds = {r["kind"] for r in records}
    assert "deployment_started" in kinds
    assert "gate_evaluated" in kinds


def test_config_run(tmp_path, capsys):
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(
        "detector: diffusion\nrollout: canary\npolicy: risk_aware\n"
        "scenario: healthy\nrisk_tier: standard\n"
    )
    rc = main(["run", "--config", str(cfg), "--quiet"])
    assert rc == 0
    assert "COMPLETED" in capsys.readouterr().out
