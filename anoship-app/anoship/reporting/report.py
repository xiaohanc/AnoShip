"""Render a RunReport into human-readable summaries and timelines."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..pipeline.pipeline import RunReport

__all__ = ["render_summary", "render_timeline", "render_report"]


def _outcome(report: "RunReport") -> str:
    if report.success:
        return "COMPLETED (candidate fully rolled out and marked safe)"
    if report.rolled_back:
        return "ROLLED BACK (automated mitigation)"
    if report.held:
        return "HELD (promotion paused pending investigation)"
    return report.summary.final_state.upper()


def render_summary(report: "RunReport") -> str:
    s = report.summary
    lines = [
        f"Deployment {s.deployment_id}: {_outcome(report)}",
        f"  stages promoted : {s.stages_promoted}/{s.stages_total}",
        f"  final state     : {s.final_state}",
    ]
    if s.detection_delay is not None:
        lines.append(f"  detected at     : stage index {int(s.detection_delay)}")
    if s.root_cause:
        lines.append(f"  root cause      : {s.root_cause}")
    return "\n".join(lines)


def render_timeline(report: "RunReport") -> str:
    lines = ["Timeline:"]
    for ev in report.events:
        stage = f"[{ev.stage}] " if ev.stage else ""
        lines.append(f"  - {ev.kind.value:22s} {stage}{ev.message}")
    return "\n".join(lines)


def render_report(report: "RunReport") -> str:
    """Full markdown-ish report: summary + decisions + timeline."""
    parts = ["# anoship deployment report", "", render_summary(report), ""]
    parts.append("## Gate decisions")
    for d in report.decisions:
        ar = d.anomaly_result
        rate = f"{ar.anomaly_rate:.0%}" if ar else "n/a"
        parts.append(
            f"- {d.stage.name}: **{d.action.value}** ({d.reason}); "
            f"anomaly_rate={rate}"
        )
    parts.append("")
    parts.append("## " + render_timeline(report).replace("\n", "\n"))
    return "\n".join(parts)
