from __future__ import annotations

from dataclasses import asdict

from .ai import AgentManager
from .models import FinalDeliverable, TaskRoute, UserRequest
from .pipelines.qa import QAPipeline
from .pipelines.retrieval import RetrievalPipeline
from .pipelines.simulation import SimulationPipeline
from .router import route_request
from .state import RunStateStore
from .triage import FailureTriage
from .validators import EvidenceValidator, FinalConfidenceCalibrator, InputGate, QAQualityGate, ResultGate


class MaterialsAgentApp:
    def __init__(
        self,
        state_root: str = "run-artifacts",
        enable_ai: bool = True,
        config_path: str = "config.json",
    ) -> None:
        self.state_store = RunStateStore(state_root)
        self.agent_manager = AgentManager(enable_ai=enable_ai, config_path=config_path)
        self.retrieval_pipeline = RetrievalPipeline()
        self.qa_pipeline = QAPipeline(agent_manager=self.agent_manager)
        self.simulation_pipeline = SimulationPipeline(agent_manager=self.agent_manager)

    def run(self, request: UserRequest, approve_execution: bool = False) -> FinalDeliverable:
        state = self.state_store.start(request)
        route = route_request(request)
        state.route = route
        self.state_store.record_event(state, "route", "ok", {"route": route.value})

        retrieval = self.retrieval_pipeline.run(request)
        self.state_store.attach_artifact(state, "retrieval", asdict(retrieval))
        for warning in retrieval.warnings:
            self.state_store.add_warning(state, warning)
        evidence_gate = EvidenceValidator.validate(request, retrieval)
        self.state_store.record_event(
            state,
            "evidence-validator",
            "ok" if evidence_gate.passed else "failed",
            {"confidence": evidence_gate.confidence, "issues": evidence_gate.issues},
        )

        warnings = list(retrieval.warnings)
        if not evidence_gate.passed:
            triage = FailureTriage.decide("retrieval", evidence_gate.issues)
            warnings.extend(evidence_gate.issues)
            warnings.append(triage.message)
            return self._finalize(
                state=state,
                route=route,
                summary="Retrieval validation failed before downstream processing.",
                structured_output={
                    "triage_action": triage.action.value,
                    "retry_stage": triage.retry_stage,
                    "issues": evidence_gate.issues,
                },
                citations=[],
                confidence=evidence_gate.confidence,
                warnings=warnings,
            )

        if route == TaskRoute.QA:
            qa_result = self.qa_pipeline.run(request, retrieval)
            qa_gate = QAQualityGate.validate(qa_result, retrieval)
            self.state_store.attach_output(state, "qa", asdict(qa_result))
            self.state_store.record_event(
                state,
                "qa-quality-gate",
                "ok" if qa_gate.passed else "failed",
                {"confidence": qa_gate.confidence, "issues": qa_gate.issues},
            )
            warnings.extend(qa_gate.issues)
            return self._finalize(
                state=state,
                route=route,
                summary=qa_result.answer,
                structured_output=qa_result.structured_output,
                citations=qa_result.citations,
                confidence=FinalConfidenceCalibrator.calibrate(
                    evidence_gate.confidence, qa_gate.confidence, qa_result.confidence
                ),
                warnings=warnings,
            )

        simulation_plan = self.simulation_pipeline.prepare(request, retrieval)
        plan_gate = InputGate.validate(simulation_plan)
        self.state_store.attach_output(state, "simulation_plan", asdict(simulation_plan))
        self.state_store.record_event(
            state,
            "input-gate",
            "ok" if plan_gate.passed else "failed",
            {"confidence": plan_gate.confidence, "issues": plan_gate.issues},
        )

        if not plan_gate.passed:
            triage = FailureTriage.decide("simulation-prep", plan_gate.issues)
            warnings.extend(plan_gate.issues)
            warnings.append(triage.message)
            return self._finalize(
                state=state,
                route=route,
                summary="Simulation planning failed validation.",
                structured_output={
                    "triage_action": triage.action.value,
                    "retry_stage": triage.retry_stage,
                    "issues": plan_gate.issues,
                },
                citations=[],
                confidence=FinalConfidenceCalibrator.calibrate(evidence_gate.confidence, plan_gate.confidence),
                warnings=warnings,
            )

        execution = self.simulation_pipeline.execute(simulation_plan, approve_execution)
        analysis = self.simulation_pipeline.analyze(simulation_plan, execution)
        result_gate = ResultGate.validate(execution, analysis)
        self.state_store.attach_output(
            state,
            "simulation",
            {
                "plan": asdict(simulation_plan),
                "execution": asdict(execution),
                "analysis": asdict(analysis),
            },
        )
        self.state_store.record_event(
            state,
            "result-gate",
            "ok" if result_gate.passed else "failed",
            {"confidence": result_gate.confidence, "issues": result_gate.issues},
        )

        warnings.extend(plan_gate.issues)
        warnings.extend(result_gate.issues)
        return self._finalize(
            state=state,
            route=route,
            summary=analysis.summary,
            structured_output=analysis.structured_output,
            citations=[],
            confidence=FinalConfidenceCalibrator.calibrate(
                evidence_gate.confidence, plan_gate.confidence, result_gate.confidence, analysis.confidence
            ),
            warnings=warnings,
        )

    def _finalize(
        self,
        state,
        route: TaskRoute,
        summary: str,
        structured_output: dict,
        citations: list[dict],
        confidence: float,
        warnings: list[str],
    ) -> FinalDeliverable:
        state.status = "completed"
        deliverable = FinalDeliverable(
            run_id=state.run_id,
            route=route,
            summary=summary,
            structured_output=structured_output,
            citations=citations,
            confidence=confidence,
            agent_usage=self.agent_manager.get_usage(),
            warnings=warnings,
            artifacts=state.artifacts,
            state_path=state.paths["state"],
        )
        deliverable.deliverable_path = state.paths["deliverable"]
        self.state_store.attach_artifact(state, "agent_usage", self.agent_manager.get_usage())
        deliverable_path = self.state_store.write_deliverable(state, deliverable)
        deliverable.deliverable_path = str(deliverable_path.resolve())
        self.state_store.write_deliverable(state, deliverable)
        self.state_store.record_event(state, "finalize", "ok", {"deliverable": deliverable.deliverable_path})
        self.state_store.write_state(state)
        return deliverable

