from __future__ import annotations

from ..ai import AgentManager
from ..models import AnalysisResult, ExecutionResult, RetrievalResult, SimulationPlan, UserRequest


SIMULATION_SOFTWARE_HINTS = {
    "vasp": "VASP",
    "quantum espresso": "Quantum ESPRESSO",
    "espresso": "Quantum ESPRESSO",
    "berkeleygw": "BerkeleyGW",
}


class SimulationPipeline:
    def __init__(self, agent_manager: AgentManager | None = None) -> None:
        self.agent_manager = agent_manager

    def prepare(self, request: UserRequest, retrieval: RetrievalResult) -> SimulationPlan:
        if self.agent_manager is not None:
            ai_result = self.agent_manager.plan_simulation(
                request.task,
                {
                    "material_id": request.material_id,
                    "material_name": request.material_name,
                    "constraints": request.constraints,
                    "evidence": [
                        {
                            "title": record.title,
                            "source_type": record.source_type,
                            "provenance": record.provenance,
                            "snippet": " ".join(record.content.split())[:1200],
                        }
                        for record in retrieval.records[:5]
                    ],
                },
            )
            if ai_result:
                parameters = ai_result.get("parameters", {})
                if not isinstance(parameters, dict):
                    parameters = {}
                parameters.setdefault("task", request.task)
                parameters.setdefault("material_id", request.material_id)
                parameters.setdefault("material_name", request.material_name)
                parameters.setdefault("constraints", request.constraints)
                return SimulationPlan(
                    summary=str(ai_result.get("summary", "")).strip(),
                    software=str(ai_result.get("software", self._choose_software(request))).strip(),
                    parameters=parameters,
                    requires_human_approval=bool(ai_result.get("requires_human_approval", True)),
                    artifacts={
                        "assumptions": ai_result.get("assumptions", []),
                        "agent_used": True,
                    },
                )

        software = self._choose_software(request)
        summary = (
            f"Prepared a dry-run simulation plan for task '{request.task}' "
            f"using {software}."
        )
        parameters = {
            "task": request.task,
            "material_id": request.material_id,
            "material_name": request.material_name,
            "paper_count": len([record for record in retrieval.records if record.source_type == "paper"]),
            "structure_count": len([record for record in retrieval.records if record.source_type == "structure"]),
            "constraints": request.constraints,
        }
        return SimulationPlan(
            summary=summary,
            software=software,
            parameters=parameters,
            requires_human_approval=True,
            artifacts={
                "agent_used": False,
                "prep_notes": [
                    "Execution backend is intentionally dry-run until a real scheduler wrapper is configured.",
                    "Human approval is required before any expensive or external job submission.",
                ]
            },
        )

    def execute(self, plan: SimulationPlan, approved: bool) -> ExecutionResult:
        if not approved:
            return ExecutionResult(
                status="needs-approval",
                summary="Simulation plan created but not executed because human approval was not granted.",
                logs=["Execution halted at approval gate."],
                artifacts={"approved": False},
            )

        return ExecutionResult(
            status="dry-run",
            summary="Simulation execution backend is not configured; returning a dry-run artifact bundle.",
            logs=["Dry-run mode: no external software was invoked."],
            artifacts={"approved": True, "software": plan.software},
        )

    def analyze(self, plan: SimulationPlan, execution: ExecutionResult) -> AnalysisResult:
        summary = (
            f"{execution.summary} Prepared parameters are ready for a future execution wrapper."
        )
        confidence = 0.45 if execution.status == "dry-run" else 0.25
        if execution.status == "needs-approval":
            confidence = 0.3
        return AnalysisResult(
            summary=summary,
            structured_output={
                "software": plan.software,
                "execution_status": execution.status,
                "parameters": plan.parameters,
                "artifacts": execution.artifacts,
            },
            confidence=confidence,
        )

    def _choose_software(self, request: UserRequest) -> str:
        software_hint = request.constraints.get("software", "").lower()
        task_text = request.task.lower()
        for keyword, name in SIMULATION_SOFTWARE_HINTS.items():
            if keyword in software_hint or keyword in task_text:
                return name
        return "Unspecified simulation backend"

