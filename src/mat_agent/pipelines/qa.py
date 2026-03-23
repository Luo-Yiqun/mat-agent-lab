from __future__ import annotations

from ..ai import AgentManager
from ..models import QAResult, RetrievalResult, UserRequest


class QAPipeline:
    def __init__(self, agent_manager: AgentManager | None = None) -> None:
        self.agent_manager = agent_manager

    def run(self, request: UserRequest, retrieval: RetrievalResult) -> QAResult:
        citations = []
        evidence_summaries = []
        evidence_payload = []
        for record in retrieval.records[:5]:
            citations.append(
                {
                    "title": record.title,
                    "source_type": record.source_type,
                    "location": record.provenance.get("location"),
                    "source_id": record.provenance.get("source_id"),
                }
            )
            snippet = " ".join(record.content.split())[:220]
            if snippet:
                evidence_summaries.append(f"{record.title}: {snippet}")
            evidence_payload.append(
                {
                    "title": record.title,
                    "source_type": record.source_type,
                    "provenance": record.provenance,
                    "snippet": " ".join(record.content.split())[:1200],
                }
            )

        ai_result = None
        if self.agent_manager is not None:
            ai_result = self.agent_manager.answer_question(request.task, evidence_payload)
        if ai_result and str(ai_result.get("answer", "")).strip():
            return QAResult(
                answer=str(ai_result.get("answer", "")).strip(),
                citations=ai_result.get("citations", citations) if isinstance(ai_result.get("citations", citations), list) else citations,
                structured_output={
                    "task": request.task,
                    "evidence_count": len(retrieval.records),
                    "warnings": retrieval.warnings,
                    "legacy_handoff_ready": "legacy_literature_review" in retrieval.artifacts,
                    "gaps": ai_result.get("gaps", []) if isinstance(ai_result.get("gaps", []), list) else [],
                    "agent_used": True,
                },
                confidence=float(ai_result.get("confidence", 0.0) or 0.0),
            )

        answer_lines = [f"Task: {request.task}"]
        if evidence_summaries:
            answer_lines.append("Evidence summary:")
            answer_lines.extend(evidence_summaries[:3])
        if "legacy_literature_review" in retrieval.artifacts:
            answer_lines.append(
                "Legacy LiteratureReview handoff is prepared for material-based cited-paper retrieval."
            )

        confidence = min(0.9, 0.35 + 0.12 * len(citations))
        structured_output = {
            "task": request.task,
            "evidence_count": len(retrieval.records),
            "warnings": retrieval.warnings,
            "legacy_handoff_ready": "legacy_literature_review" in retrieval.artifacts,
            "agent_used": False,
        }

        return QAResult(
            answer="\n".join(answer_lines),
            citations=citations,
            structured_output=structured_output,
            confidence=round(confidence, 3),
        )

