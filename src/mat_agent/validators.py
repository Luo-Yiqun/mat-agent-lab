from __future__ import annotations

from .models import AnalysisResult, ExecutionResult, GateResult, QAResult, RetrievalResult, SimulationPlan, UserRequest


class EvidenceValidator:
    @staticmethod
    def validate(request: UserRequest, result: RetrievalResult) -> GateResult:
        issues: list[str] = []

        if not result.records:
            issues.append("No evidence records were produced.")

        records_with_provenance = 0
        records_with_content = 0
        for record in result.records:
            if record.provenance.get("location") or record.provenance.get("source_id"):
                records_with_provenance += 1
            else:
                issues.append(f"Missing provenance for evidence record '{record.title}'.")

            if record.title.strip() == "":
                issues.append("Evidence record missing title.")
            if record.content.strip():
                records_with_content += 1

        coverage_score = 0
        possible_coverage = 0

        if request.task:
            possible_coverage += 1
            if result.records:
                coverage_score += 1
        if request.material_id or request.material_name:
            possible_coverage += 1
            if any(
                (request.material_id and request.material_id in (record.content + str(record.metadata)))
                or (request.material_name and request.material_name.lower() in (record.content + str(record.metadata)).lower())
                for record in result.records
            ):
                coverage_score += 1
        if request.paper_paths:
            possible_coverage += 1
            if sum(record.source_type == "paper" for record in result.records) >= len(request.paper_paths):
                coverage_score += 1

        provenance_score = 0 if not result.records else records_with_provenance / len(result.records)
        content_score = 0 if not result.records else records_with_content / len(result.records)
        coverage_ratio = 1.0 if possible_coverage == 0 else coverage_score / possible_coverage

        confidence = max(0.0, min(1.0, 0.2 + 0.4 * provenance_score + 0.2 * content_score + 0.2 * coverage_ratio))
        passed = not issues or (bool(result.records) and provenance_score >= 0.5)
        return GateResult(passed=passed, confidence=confidence, issues=issues)


class QAQualityGate:
    @staticmethod
    def validate(result: QAResult, retrieval: RetrievalResult) -> GateResult:
        issues: list[str] = []
        if not result.answer.strip():
            issues.append("QA pipeline produced an empty answer.")
        if retrieval.records and not result.citations:
            issues.append("QA output has no citations despite available evidence.")
        if not 0.0 <= result.confidence <= 1.0:
            issues.append("QA confidence is outside the valid range.")
        confidence = result.confidence if 0.0 <= result.confidence <= 1.0 else 0.0
        return GateResult(passed=not issues, confidence=confidence, issues=issues)


class InputGate:
    @staticmethod
    def validate(plan: SimulationPlan) -> GateResult:
        issues: list[str] = []
        if not plan.summary.strip():
            issues.append("Simulation plan is missing a summary.")
        if not plan.software.strip():
            issues.append("Simulation plan is missing a software target.")
        if "task" not in plan.parameters:
            issues.append("Simulation plan is missing the original task.")
        confidence = 0.0 if issues else 0.75
        return GateResult(passed=not issues, confidence=confidence, issues=issues)


class ResultGate:
    @staticmethod
    def validate(execution: ExecutionResult, analysis: AnalysisResult) -> GateResult:
        issues: list[str] = []
        if execution.status not in {"dry-run", "succeeded", "needs-approval"}:
            issues.append(f"Unexpected execution status '{execution.status}'.")
        if not analysis.summary.strip():
            issues.append("Analysis summary is empty.")
        confidence = max(0.0, min(1.0, analysis.confidence))
        return GateResult(passed=not issues, confidence=confidence, issues=issues)


class FinalConfidenceCalibrator:
    @staticmethod
    def calibrate(*scores: float) -> float:
        valid_scores = [score for score in scores if 0.0 <= score <= 1.0]
        if not valid_scores:
            return 0.0
        return round(sum(valid_scores) / len(valid_scores), 3)

