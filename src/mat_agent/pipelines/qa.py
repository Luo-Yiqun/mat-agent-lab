from __future__ import annotations

from ..models import QAResult, RetrievalResult, UserRequest


class QAPipeline:
    def run(self, request: UserRequest, retrieval: RetrievalResult) -> QAResult:
        citations = []
        evidence_summaries = []
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
        }

        return QAResult(
            answer="\n".join(answer_lines),
            citations=citations,
            structured_output=structured_output,
            confidence=round(confidence, 3),
        )

