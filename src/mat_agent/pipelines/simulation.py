from __future__ import annotations

from ..ai import AgentManager
from ..models import AnalysisResult, ExecutionResult, RetrievalResult, SimulationPlan, UserRequest


SIMULATION_SOFTWARE_HINTS = {
    "fhi-aims": "FHI-aims",
    "fhi aims": "FHI-aims",
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
        artifacts = {
            "agent_used": False,
            "prep_notes": [
                "Execution backend is intentionally dry-run until a real scheduler wrapper is configured.",
                "Human approval is required before any expensive or external job submission.",
            ]
        }
        generated_files = self._generate_input_files(request, software)
        if generated_files:
            artifacts["generated_files"] = generated_files
        return SimulationPlan(
            summary=summary,
            software=software,
            parameters=parameters,
            requires_human_approval=True,
            artifacts=artifacts,
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

    def _generate_input_files(self, request: UserRequest, software: str) -> dict[str, str]:
        task_text = request.task.lower()
        if not any(keyword in task_text for keyword in ("single-point", "single point", "spe", "energy")):
            return {}

        label = request.material_id or request.material_name or "unknown_material"
        structure_hint = request.structure_paths[0] if request.structure_paths else "<provide structure file>"

        if software == "Quantum ESPRESSO":
            qe_input = (
                "&control\n"
                "  calculation = 'scf'\n"
                "  prefix = 'mat_agent'\n"
                "  pseudo_dir = './pseudo'\n"
                "  outdir = './tmp'\n"
                "/\n"
                "&system\n"
                "  ibrav = 0,\n"
                "  nat = 0,\n"
                "  ntyp = 0,\n"
                "  ecutwfc = 60.0,\n"
                "/\n"
                "&electrons\n"
                "  conv_thr = 1.0d-8,\n"
                "/\n"
                "ATOMIC_SPECIES\n"
                "! Fill in species and pseudopotentials based on the structure source.\n"
                "CELL_PARAMETERS angstrom\n"
                "! Fill in lattice vectors using the structure file.\n"
                "ATOMIC_POSITIONS angstrom\n"
                f"! Source structure: {structure_hint}\n"
                "K_POINTS automatic\n"
                "4 4 4 0 0 0\n"
            )
            qe_run = (
                "#!/usr/bin/env bash\n"
                "set -euo pipefail\n\n"
                "PW_BIN=${PW_BIN:-pw.x}\n"
                "${PW_BIN} -in qe_scf.in > qe.out\n"
            )
            return {
                "qe_scf.in": qe_input,
                "run_qe.sh": qe_run,
            }

        if software != "FHI-aims":
            return {}

        geometry = (
            f"# geometry.in placeholder for {label}\n"
            "# Replace this file with the actual structure once retrieval/export is wired.\n"
            "# Example atom block:\n"
            "# atom 0.0 0.0 0.0 C\n"
        )
        control = (
            "# control.in for an FHI-aims single-point energy run\n"
            "xc pbe\n"
            "spin none\n"
            "relativistic atomic_zora scalar\n"
            "sc_accuracy_etot 1e-6\n"
            "sc_accuracy_eev 1e-3\n"
            "sc_accuracy_rho 1e-5\n"
            "output band 0 0 0 0 0 0 1\n"
        )
        run_script = (
            "#!/usr/bin/env bash\n"
            "set -euo pipefail\n\n"
            "AIMS_BIN=${AIMS_BIN:-aims.x}\n"
            "${AIMS_BIN} > aims.out\n"
        )
        return {
            "geometry.in": geometry,
            "control.in": control,
            "run_fhi_aims.sh": run_script,
        }

