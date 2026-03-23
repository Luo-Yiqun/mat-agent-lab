from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any
from uuid import uuid4

from .models import RunState, UserRequest


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

