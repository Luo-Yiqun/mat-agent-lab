from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any
from uuid import uuid4

from .models import FinalDeliverable, RunState, UserRequest


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def to_jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return {key: to_jsonable(item) for key, item in asdict(value).items()}
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [to_jsonable(item) for item in value]
    return value


class RunStateStore:
    def __init__(self, root: str | Path = "run-artifacts") -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def start(self, request: UserRequest) -> RunState:
        run_id = uuid4().hex[:12]
        run_dir = self.root / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        state = RunState(
            run_id=run_id,
            request=request,
            status="running",
            paths={
                "run_dir": str(run_dir.resolve()),
                "state": str((run_dir / "state.json").resolve()),
                "deliverable": str((run_dir / "deliverable.json").resolve()),
                "deliverable_text": str((run_dir / "deliverable.txt").resolve()),
            },
        )
        self.record_event(state, "start", "ok", {"created_at": utc_now()})
        return state

    def record_event(
        self,
        state: RunState,
        stage: str,
        status: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        state.events.append(
            {
                "timestamp": utc_now(),
                "stage": stage,
                "status": status,
                "details": details or {},
            }
        )
        self.write_state(state)

    def add_warning(self, state: RunState, warning: str) -> None:
        state.warnings.append(warning)
        self.write_state(state)

    def attach_artifact(self, state: RunState, name: str, value: Any) -> None:
        state.artifacts[name] = value
        self.write_state(state)

    def attach_output(self, state: RunState, name: str, value: Any) -> None:
        state.outputs[name] = value
        self.write_state(state)

    def write_state(self, state: RunState) -> Path:
        path = Path(state.paths["state"])
        path.write_text(json.dumps(to_jsonable(state), indent=2), encoding="utf-8")
        return path

    def write_deliverable(self, state: RunState, payload: Any) -> Path:
        path = Path(state.paths["deliverable"])
        path.write_text(json.dumps(to_jsonable(payload), indent=2), encoding="utf-8")
        return path

    def write_deliverable_text(self, state: RunState, payload: FinalDeliverable) -> Path:
        path = Path(state.paths["deliverable_text"])
        path.write_text(self.render_deliverable_text(payload), encoding="utf-8")
        return path

    def render_deliverable_text(self, payload: FinalDeliverable) -> str:
        lines = [
            f"Run ID: {payload.run_id}",
            f"Route: {payload.route.value}",
            f"Confidence: {payload.confidence}",
            "",
            "Summary:",
            payload.summary or "(no summary)",
        ]

        generated_files = payload.structured_output.get("generated_files", {})
        if isinstance(generated_files, dict) and generated_files:
            lines.extend(["", "Generated Files:"])
            for name, location in generated_files.items():
                lines.append(f"- {name}: {location}")

        if payload.citations:
            lines.extend(["", "Citations:"])
            for citation in payload.citations[:8]:
                title = citation.get("title", "unknown source")
                source_type = citation.get("source_type", "source")
                locator = citation.get("location") or citation.get("source_id") or citation.get("excerpt")
                if locator:
                    lines.append(f"- {title} [{source_type}]: {locator}")
                else:
                    lines.append(f"- {title} [{source_type}]")

        if payload.warnings:
            lines.extend(["", "Warnings:"])
            for warning in payload.warnings:
                lines.append(f"- {warning}")

        if payload.agent_usage:
            lines.extend(["", "Agent Usage:"])
            for usage in payload.agent_usage:
                lines.append(f"- {usage['role']}: {usage['model']} ({usage['status']})")

        return "\n".join(lines).strip() + "\n"

