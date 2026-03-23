from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class TaskRoute(str, Enum):
    QA = "qa"
    SIMULATION = "simulation"


class FailureAction(str, Enum):
    RETRY = "retry"
    DEGRADE = "degrade"
    ESCALATE = "escalate"


@dataclass
class UserRequest:
    task: str
    material_id: str | None = None
    material_name: str | None = None
    paper_paths: list[str] = field(default_factory=list)
    structure_paths: list[str] = field(default_factory=list)
    constraints: dict[str, Any] = field(default_factory=dict)
    route_hint: TaskRoute | None = None


@dataclass
class EvidenceRecord:
    title: str
    content: str = ""
    source_type: str = "unknown"
    provenance: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievalResult:
    records: list[EvidenceRecord] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    artifacts: dict[str, Any] = field(default_factory=dict)
    coverage: dict[str, bool] = field(default_factory=dict)


@dataclass
class GateResult:
    passed: bool
    confidence: float
    issues: list[str] = field(default_factory=list)


@dataclass
class QAResult:
    answer: str
    citations: list[dict[str, Any]] = field(default_factory=list)
    structured_output: dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0


@dataclass
class SimulationPlan:
    summary: str
    software: str
    parameters: dict[str, Any] = field(default_factory=dict)
    requires_human_approval: bool = True
    artifacts: dict[str, Any] = field(default_factory=dict)


@dataclass
class ExecutionResult:
    status: str
    summary: str
    logs: list[str] = field(default_factory=list)
    artifacts: dict[str, Any] = field(default_factory=dict)


@dataclass
class AnalysisResult:
    summary: str
    structured_output: dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0


@dataclass
class TriageDecision:
    action: FailureAction
    message: str
    retry_stage: str | None = None


@dataclass
class FinalDeliverable:
    run_id: str
    route: TaskRoute
    summary: str
    structured_output: dict[str, Any]
    citations: list[dict[str, Any]]
    confidence: float
    agent_usage: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    artifacts: dict[str, Any] = field(default_factory=dict)
    state_path: str | None = None
    deliverable_path: str | None = None


@dataclass
class RunState:
    run_id: str
    request: UserRequest
    status: str
    route: TaskRoute | None = None
    warnings: list[str] = field(default_factory=list)
    events: list[dict[str, Any]] = field(default_factory=list)
    artifacts: dict[str, Any] = field(default_factory=dict)
    outputs: dict[str, Any] = field(default_factory=dict)
    paths: dict[str, str] = field(default_factory=dict)

