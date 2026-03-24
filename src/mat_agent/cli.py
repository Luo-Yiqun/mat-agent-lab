from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict

from .models import TaskRoute, UserRequest
from .orchestrator import MaterialsAgentApp


def safe_print(text: str) -> None:
    try:
        print(text)
    except UnicodeEncodeError:
        encoding = sys.stdout.encoding or "utf-8"
        sanitized = text.encode(encoding, errors="replace").decode(encoding, errors="replace")
        print(sanitized)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mat-agent",
        description="Simplified materials agent application built around the proposal flowchart.",
    )
    parser.add_argument("--task", required=True, help="Question or computational task to execute.")
    parser.add_argument("--material-id", help="CSD/MP identifier.")
    parser.add_argument("--material-name", help="Human-readable material name.")
    parser.add_argument("--paper", action="append", default=[], help="Path to a local paper file. Repeatable.")
    parser.add_argument("--structure", action="append", default=[], help="Path to a local structure file. Repeatable.")
    parser.add_argument("--software", help="Preferred simulation software backend.")
    parser.add_argument("--accuracy", help="Accuracy target or method hint.")
    parser.add_argument("--budget", help="Budget or runtime constraint.")
    parser.add_argument("--route", choices=[route.value for route in TaskRoute], help="Force a route.")
    parser.add_argument("--approve-execution", action="store_true", help="Reserved for future real execution backends; not needed for current dry-run file generation.")
    parser.add_argument("--state-root", default="run-artifacts", help="Directory for run state and deliverables.")
    parser.add_argument("--config", default="config.json", help="Path to the AI gateway config JSON file.")
    parser.add_argument("--no-ai", action="store_true", help="Disable AI agents and use deterministic fallbacks only.")
    parser.add_argument("--json", action="store_true", help="Print the full deliverable as JSON.")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    request = UserRequest(
        task=args.task,
        material_id=args.material_id,
        material_name=args.material_name,
        paper_paths=args.paper,
        structure_paths=args.structure,
        constraints={
            key: value
            for key, value in {
                "software": args.software,
                "accuracy": args.accuracy,
                "budget": args.budget,
            }.items()
            if value
        },
        route_hint=TaskRoute(args.route) if args.route else None,
    )

    app = MaterialsAgentApp(
        state_root=args.state_root,
        enable_ai=not args.no_ai,
        config_path=args.config,
    )
    deliverable = app.run(request, approve_execution=args.approve_execution)

    if args.json:
        safe_print(json.dumps(asdict(deliverable), indent=2))
        return

    safe_print(f"run_id: {deliverable.run_id}")
    safe_print(f"route: {deliverable.route.value}")
    safe_print(f"confidence: {deliverable.confidence}")
    safe_print(f"state: {deliverable.state_path}")
    safe_print(f"deliverable: {deliverable.deliverable_path}")
    if deliverable.summary:
        safe_print("summary:")
        safe_print(deliverable.summary)
    generated_files = deliverable.structured_output.get("generated_files", {})
    if generated_files:
        safe_print("generated_files:")
        for name, path in generated_files.items():
            safe_print(f"- {name}: {path}")
    if deliverable.warnings:
        safe_print("warnings:")
        for warning in deliverable.warnings:
            safe_print(f"- {warning}")
    if deliverable.agent_usage:
        safe_print("agent_usage:")
        for usage in deliverable.agent_usage:
            tools = ", ".join(usage.get("tools", []))
            safe_print(f"- {usage['role']}: {usage['model']} ({usage['status']})")
            if tools:
                safe_print(f"  tools: {tools}")


if __name__ == "__main__":
    main()

