from __future__ import annotations

from ..models import AnalysisResult, ExecutionResult, RetrievalResult, SimulationPlan, UserRequest


SIMULATION_SOFTWARE_HINTS = {
    "vasp": "VASP",
    "quantum espresso": "Quantum ESPRESSO",
    "espresso": "Quantum ESPRESSO",
    "berkeleygw": "BerkeleyGW",
}


class SimulationPipeline:
    def prepare(self, request: UserRequest, retrieval: RetrievalResult) -> SimulationPlan:
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

