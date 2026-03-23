from __future__ import annotations

import argparse
import json
from dataclasses import asdict

from .models import TaskRoute, UserRequest
from .orchestrator import MaterialsAgentApp


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
    parser.add_argument("--approve-execution", action="store_true", help="Pass the human approval gate for dry-run execution.")
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
        print(json.dumps(asdict(deliverable), indent=2))
        return

    print(f"run_id: {deliverable.run_id}")
    print(f"route: {deliverable.route.value}")
    print(f"confidence: {deliverable.confidence}")
    print(f"state: {deliverable.state_path}")
    print(f"deliverable: {deliverable.deliverable_path}")
    if deliverable.warnings:
        print("warnings:")
        for warning in deliverable.warnings:
            print(f"- {warning}")
    if deliverable.agent_usage:
        print("agent_usage:")
        for usage in deliverable.agent_usage:
            print(f"- {usage['role']}: {usage['model']} ({usage['status']})")


if __name__ == "__main__":
    main()

