"""End-to-end demonstration of the anoship deployment-safety framework.

Run with:  python examples/end_to_end.py

It shows two things:

1. A full, narrated deployment run (progressive rollout -> health gate ->
   automated rollback -> root cause) for a subtle regression.
2. A comparison matrix of every detector x every scenario, illustrating why the
   *choice* of anomaly-detection method matters -- the core thesis of the
   pluggable framework.
"""

from __future__ import annotations

import anoship.app as ans
from anoship.observability.exporters import ConsoleExporter
from anoship.reporting.report import render_summary


def narrated_regression() -> None:
    print("=" * 70)
    print("1) Narrated deployment: subtle regression, high-impact risk tier")
    print("=" * 70)
    scn = ans.build_scenario("regression")
    pipeline = ans.DeploymentPipeline(
        detector=ans.DETECTORS.create("diffusion"),
        rollout=ans.ROLLOUTS.create("canary"),
        policy=ans.POLICIES.create("risk_aware"),
        exporters=[ConsoleExporter()],
        risk_tier="high_impact",
    ).fit(scn.baseline)
    report = pipeline.run(scn.source)
    print()
    print(render_summary(report))


def comparison_matrix() -> None:
    print()
    print("=" * 70)
    print("2) Detector x scenario outcome matrix (canary, standard tier)")
    print("=" * 70)
    detectors = ["ewma", "habituation", "causal", "diffusion", "spatiotemporal"]
    scenarios = ["healthy", "regression", "drift", "spike", "noisy"]

    header = f"{'detector':16s}" + "".join(f"{s:>12s}" for s in scenarios)
    print(header)
    print("-" * len(header))
    for det_name in detectors:
        row = f"{det_name:16s}"
        for scn_name in scenarios:
            scn = ans.build_scenario(scn_name)
            pipeline = ans.DeploymentPipeline(
                detector=ans.DETECTORS.create(det_name),
                rollout=ans.ROLLOUTS.create("canary"),
                policy=ans.POLICIES.create("risk_aware"),
                risk_tier="standard",
            ).fit(scn.baseline)
            report = pipeline.run(scn.source)
            if report.success:
                outcome = "complete"
            elif report.rolled_back:
                outcome = "rollback"
            else:
                outcome = "held"
            row += f"{outcome:>12s}"
        print(row)
    print()
    print("Ideal: 'complete' for healthy/noisy, 'rollback' for the faults.")
    print("Note how the diffusion detector uniquely stays calm on pure noise.")


if __name__ == "__main__":
    narrated_regression()
    comparison_matrix()
