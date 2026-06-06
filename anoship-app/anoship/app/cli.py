"""Command-line interface for anoship.

Examples
--------
    anoship list
    anoship run --scenario regression --detector diffusion --risk-tier high_impact
    anoship run --config configs/high_impact.yaml --json run.json
"""

from __future__ import annotations

import argparse
import sys
from typing import List, Optional

from anoship.app import __version__
from anoship.app.builder import build_pipeline
from anoship.core.config import ComponentSpec, DeploymentConfig
from anoship.core.registry import (
    DETECTORS,
    EXPORTERS,
    POLICIES,
    ROLLOUTS,
    SCENARIOS,
    THRESHOLDS,
)
from anoship.observability.exporters import ConsoleExporter, JSONExporter
from anoship.reporting.report import render_report, render_summary
from anoship.scenarios import build_scenario


def _cmd_list(args: argparse.Namespace) -> int:
    import anoship.app  # noqa: F401  (ensure registration)

    sections = {
        "detectors": DETECTORS,
        "rollouts": ROLLOUTS,
        "policies": POLICIES,
        "thresholds": THRESHOLDS,
        "exporters": EXPORTERS,
        "scenarios": SCENARIOS,
    }
    for title, reg in sections.items():
        print(f"{title}:")
        for name in reg.names():
            print(f"  - {name}")
    return 0


def _config_from_args(args: argparse.Namespace) -> DeploymentConfig:
    data = {
        "detector": args.detector,
        "rollout": args.rollout,
        "policy": args.policy,
        "risk_tier": args.risk_tier,
        "seed": args.seed,
    }
    if args.threshold:
        data["threshold"] = args.threshold
    if args.scenario:
        data["scenario"] = args.scenario
    return DeploymentConfig.from_dict(data)


def _cmd_run(args: argparse.Namespace) -> int:
    import anoship.app  # noqa: F401  (ensure registration)

    if args.config:
        config = DeploymentConfig.from_yaml(args.config)
    else:
        config = _config_from_args(args)

    scenario_spec = config.scenario or ComponentSpec(name=args.scenario or "regression")
    scenario = build_scenario(
        scenario_spec.name, seed=config.seed, **scenario_spec.params
    )

    exporters: List = []
    if not args.quiet:
        exporters.append(ConsoleExporter())
    json_exporter: Optional[JSONExporter] = None
    if args.json:
        json_exporter = JSONExporter(path=args.json)
        exporters.append(json_exporter)

    pipeline = build_pipeline(config, exporters=exporters)
    pipeline.fit(scenario.baseline)
    report = pipeline.run(scenario.source)

    print()
    if args.full:
        print(render_report(report))
    else:
        print(render_summary(report))
    if json_exporter is not None:
        report_json = json_exporter  # already flushed by bus
        print(f"\nevents written to {args.json}")
    return 0 if (report.success or report.rolled_back) else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="anoship",
        description="Anomaly-detection-driven deployment safety framework.",
    )
    parser.add_argument("--version", action="version", version=f"anoship {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p_list = sub.add_parser("list", help="list available pluggable components")
    p_list.set_defaults(func=_cmd_list)

    p_run = sub.add_parser("run", help="run a deployment-safety simulation")
    p_run.add_argument("--config", help="path to a YAML deployment config")
    p_run.add_argument("--scenario", default="regression", help="scenario name")
    p_run.add_argument("--detector", default="diffusion", help="detector name")
    p_run.add_argument("--rollout", default="canary", help="rollout strategy")
    p_run.add_argument("--policy", default="risk_aware", help="gate policy")
    p_run.add_argument("--threshold", default=None, help="threshold strategy")
    p_run.add_argument("--risk-tier", default="standard", dest="risk_tier")
    p_run.add_argument("--seed", type=int, default=0)
    p_run.add_argument("--json", default=None, help="write event log to this path")
    p_run.add_argument("--full", action="store_true", help="print full report")
    p_run.add_argument("--quiet", action="store_true", help="suppress live timeline")
    p_run.set_defaults(func=_cmd_run)
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
