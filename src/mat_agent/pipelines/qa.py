from __future__ import annotations

import re

from ..ai import AgentManager
from ..models import QAResult, RetrievalResult, UserRequest


FIGURE_PATTERN = re.compile(r"\b(?:Fig(?:ure)?|Table|Tab|Scheme)\.?\s*S?\d+[A-Za-z]?\b", re.IGNORECASE)
EV_PATTERN = re.compile(r"\b\d+(?:\.\d+)?\s*eV\b", re.IGNORECASE)
TOKEN_PATTERN = re.compile(r"[A-Za-z][A-Za-z0-9\-]+")
SENTENCE_SPLIT_PATTERN = re.compile(r"(?<=[.!?])\s+|\n+")
STOPWORDS = {
    "a",
    "an",
    "and",
    "answer",
    "are",
    "band",
    "by",
    "for",
    "from",
    "gap",
    "in",
    "is",
    "of",
    "paper",
    "papers",
    "question",
    "reported",
    "reported?",
    "the",
    "these",
    "this",
    "what",
    "where",
    "which",
    "with",
}


class QAPipeline:
    def __init__(self, agent_manager: AgentManager | None = None) -> None:
        self.agent_manager = agent_manager

    def run(self, request: UserRequest, retrieval: RetrievalResult) -> QAResult:
        citations = []
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
            supporting_references = ai_result.get("supporting_references", [])
            if not isinstance(supporting_references, list):
                supporting_references = []
            direct_answer = str(ai_result.get("direct_answer", "")).strip() or str(ai_result.get("answer", "")).strip()
            return QAResult(
                answer=self._compose_answer_text(direct_answer, supporting_references),
                citations=self._normalize_citations(ai_result.get("citations"), fallback=citations),
                structured_output={
                    "task": request.task,
                    "evidence_count": len(retrieval.records),
                    "warnings": retrieval.warnings,
                    "legacy_handoff_ready": "legacy_literature_review" in retrieval.artifacts,
                    "gaps": ai_result.get("gaps", []) if isinstance(ai_result.get("gaps", []), list) else [],
                    "agent_used": True,
                    "direct_answer": direct_answer,
                    "supporting_references": supporting_references,
                },
                confidence=self._normalize_confidence(ai_result.get("confidence", 0.0)),
            )

        direct_answer, supporting_references = self._answer_from_evidence(request, retrieval)
        confidence = min(0.92, 0.45 + 0.08 * len(supporting_references) + 0.04 * len(citations))
        structured_output = {
            "task": request.task,
            "evidence_count": len(retrieval.records),
            "warnings": retrieval.warnings,
            "legacy_handoff_ready": "legacy_literature_review" in retrieval.artifacts,
            "agent_used": False,
            "direct_answer": direct_answer,
            "supporting_references": supporting_references,
        }

        return QAResult(
            answer=self._compose_answer_text(direct_answer, supporting_references),
            citations=self._merge_citations_with_references(citations, supporting_references),
            structured_output=structured_output,
            confidence=round(confidence, 3),
        )

    def _normalize_citations(self, value: object, fallback: list[dict]) -> list[dict]:
        if not isinstance(value, list):
            return fallback
        result: list[dict] = []
        for item in value:
            if isinstance(item, dict):
                result.append(item)
            elif isinstance(item, str) and item.strip():
                result.append({"title": item.strip(), "source_type": "ai-citation"})
        return result if result else fallback

    def _normalize_confidence(self, value: object) -> float:
        if isinstance(value, bool):
            return 1.0 if value else 0.0
        if isinstance(value, (int, float)):
            return max(0.0, min(1.0, float(value)))
        if isinstance(value, str):
            text = value.strip().lower()
            qualitative = {
                "very low": 0.15,
                "low": 0.3,
                "medium": 0.55,
                "moderate": 0.55,
                "medium-high": 0.7,
                "high": 0.8,
                "very high": 0.92,
            }
            if text in qualitative:
                return qualitative[text]
            try:
                return max(0.0, min(1.0, float(text)))
            except ValueError:
                return 0.0
        return 0.0

    def _answer_from_evidence(self, request: UserRequest, retrieval: RetrievalResult) -> tuple[str, list[dict[str, str]]]:
        legacy_status = self._legacy_status(retrieval)
        if legacy_status in {"blocked-ccdc", "blocked-webdriver", "error"} and self._is_citation_task(request):
            return (
                "The LiteratureReview cited-paper backend was invoked but failed before retrieval completed, "
                "so no cited-paper list could be produced from the current environment.",
                [],
            )
        references = self._collect_supporting_references(request, retrieval)
        direct_answer = self._derive_direct_answer(request, references)

        if not direct_answer:
            if references:
                direct_answer = "The papers contain relevant evidence, but no single explicit answer could be extracted confidently."
            else:
                direct_answer = "No explicit answer was found in the provided evidence."

        if "legacy_literature_review" in retrieval.artifacts and not references:
            direct_answer += " Legacy LiteratureReview handoff is prepared for material-based cited-paper retrieval."

        return direct_answer, references

    def _legacy_status(self, retrieval: RetrievalResult) -> str:
        legacy = retrieval.artifacts.get("legacy_literature_review", {})
        if isinstance(legacy, dict):
            return str(legacy.get("status", ""))
        return ""

    def _is_citation_task(self, request: UserRequest) -> bool:
        text = request.task.lower()
        return any(keyword in text for keyword in ("paper", "papers", "citation", "citations", "cited", "reference"))

    def _derive_direct_answer(self, request: UserRequest, references: list[dict[str, str]]) -> str:
        task = request.task.lower()
        for reference in references:
            excerpt = reference.get("excerpt", "")
            value_match = EV_PATTERN.search(excerpt)
            if value_match and any(keyword in task for keyword in ("band gap", "optical gap", "gap")):
                return f"The reported gap is {value_match.group(0)}."

        for reference in references:
            excerpt = reference.get("excerpt", "")
            if excerpt:
                return excerpt
        return ""

    def _collect_supporting_references(
        self,
        request: UserRequest,
        retrieval: RetrievalResult,
    ) -> list[dict[str, str]]:
        keywords = self._task_keywords(request.task)
        references: list[dict[str, str]] = []
        seen: set[tuple[str, str, str]] = set()

        for record in retrieval.records:
            if not record.content.strip():
                continue

            lines = [line.strip() for line in record.content.splitlines() if line.strip()]
            sentences = [
                sentence.strip()
                for sentence in SENTENCE_SPLIT_PATTERN.split(record.content)
                if sentence.strip()
            ]

            for line in lines:
                figure_match = FIGURE_PATTERN.search(line)
                if figure_match:
                    entry = {
                        "source": record.title,
                        "reference_type": "figure/table",
                        "locator": figure_match.group(0),
                        "excerpt": self._truncate(line),
                    }
                    key = (entry["source"], entry["reference_type"], entry["locator"])
                    if key not in seen:
                        seen.add(key)
                        references.append(entry)

            for sentence in sentences:
                if self._matches_keywords(sentence, keywords) or EV_PATTERN.search(sentence):
                    locator = self._best_locator(sentence)
                    entry = {
                        "source": record.title,
                        "reference_type": "sentence",
                        "locator": locator,
                        "excerpt": self._truncate(sentence),
                    }
                    key = (entry["source"], entry["reference_type"], entry["excerpt"])
                    if key not in seen:
                        seen.add(key)
                        references.append(entry)

            if len(references) >= 6:
                break

        return references[:6]

    def _task_keywords(self, task: str) -> set[str]:
        tokens = {
            token.lower()
            for token in TOKEN_PATTERN.findall(task)
            if token.lower() not in STOPWORDS and len(token) > 2
        }
        if "band gap" in task.lower() or "optical gap" in task.lower():
            tokens.update({"gap", "band", "optical", "ev"})
        return tokens

    def _matches_keywords(self, text: str, keywords: set[str]) -> bool:
        lowered = text.lower()
        if not keywords:
            return True
        return any(keyword in lowered for keyword in keywords)

    def _best_locator(self, text: str) -> str:
        figure_match = FIGURE_PATTERN.search(text)
        if figure_match:
            return figure_match.group(0)
        return "sentence"

    def _truncate(self, text: str, limit: int = 220) -> str:
        compact = " ".join(text.split())
        if len(compact) <= limit:
            return compact
        return compact[: limit - 3] + "..."

    def _compose_answer_text(self, direct_answer: str, supporting_references: list[dict[str, str]]) -> str:
        lines = [f"Direct answer: {direct_answer}"]
        if supporting_references:
            lines.append("Supporting references:")
            for reference in supporting_references[:4]:
                locator = reference.get("locator", "sentence")
                lines.append(
                    f"- {reference.get('source', 'unknown source')} [{reference.get('reference_type', 'evidence')}; {locator}]: "
                    f"{reference.get('excerpt', '')}"
                )
        return "\n".join(lines)

    def _merge_citations_with_references(
        self,
        citations: list[dict],
        supporting_references: list[dict[str, str]],
    ) -> list[dict]:
        merged = list(citations)
        for reference in supporting_references:
            merged.append(
                {
                    "title": reference.get("source"),
                    "source_type": reference.get("reference_type"),
                    "location": reference.get("locator"),
                    "excerpt": reference.get("excerpt"),
                }
            )
        return merged
