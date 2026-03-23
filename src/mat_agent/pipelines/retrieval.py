from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from ..ai import AgentManager
from ..legacy.literature_review import LegacyLiteratureReviewAdapter
from ..models import EvidenceRecord, RetrievalResult, UserRequest

try:
    from pypdf import PdfReader
except ImportError:  # pragma: no cover
    PdfReader = None


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class RetrievalPipeline:
    def __init__(
        self,
        legacy_adapter: LegacyLiteratureReviewAdapter | None = None,
        agent_manager: AgentManager | None = None,
    ) -> None:
        self.legacy_adapter = legacy_adapter or LegacyLiteratureReviewAdapter()
        self.agent_manager = agent_manager

    def run(self, request: UserRequest) -> RetrievalResult:
        result = RetrievalResult()

        if request.material_id:
            result.records.append(
                EvidenceRecord(
                    title=f"Material identifier {request.material_id}",
                    content=f"User provided material identifier: {request.material_id}",
                    source_type="material-id",
                    provenance={
                        "source_id": request.material_id,
                        "retrieved_at": utc_now(),
                    },
                    metadata={"material_id": request.material_id},
                )
            )
            result.coverage["material_id"] = True
            result.artifacts["legacy_literature_review"] = self.legacy_adapter.build_retrieval_handoff(request.material_id)

        if request.material_name:
            result.records.append(
                EvidenceRecord(
                    title=f"Material name {request.material_name}",
                    content=f"User provided material name: {request.material_name}",
                    source_type="material-name",
                    provenance={
                        "source_id": request.material_name,
                        "retrieved_at": utc_now(),
                    },
                    metadata={"material_name": request.material_name},
                )
            )
            result.coverage["material_name"] = True

        for paper_path in request.paper_paths:
            self._load_local_file(Path(paper_path), "paper", result)

        for structure_path in request.structure_paths:
            self._load_local_file(Path(structure_path), "structure", result)

        result.artifacts["legacy_assets"] = self.legacy_adapter.describe_assets()
        result.coverage.setdefault("task", bool(request.task))
        self._plan_and_critique(request, result)
        return result

    def _load_local_file(self, path: Path, source_type: str, result: RetrievalResult) -> None:
        if not path.exists():
            result.warnings.append(f"Local {source_type} file not found: {path}")
            return

        content = self._read_content(path)
        result.records.append(
            EvidenceRecord(
                title=path.name,
                content=content,
                source_type=source_type,
                provenance={
                    "location": str(path.resolve()),
                    "retrieved_at": utc_now(),
                },
                metadata={
                    "suffix": path.suffix.lower(),
                    "size_bytes": path.stat().st_size,
                },
            )
        )
        result.coverage[f"{source_type}:{path.name}"] = True

    def _read_content(self, path: Path) -> str:
        if path.suffix.lower() == ".pdf":
            return self._read_pdf(path)
        return path.read_text(encoding="utf-8", errors="replace")

    def _read_pdf(self, path: Path) -> str:
        if PdfReader is None:
            return f"PDF parsing unavailable for {path.name}; install pypdf to extract text."
        reader = PdfReader(str(path))
        parts: list[str] = []
        for page in reader.pages:
            parts.append(page.extract_text() or "")
        return "\n".join(parts)

    def _plan_and_critique(self, request: UserRequest, result: RetrievalResult) -> None:
        if self.agent_manager is None:
            return

        evidence = [
            {
                "title": record.title,
                "source_type": record.source_type,
                "provenance": record.provenance,
                "metadata": record.metadata,
                "snippet": " ".join(record.content.split())[:1200],
            }
            for record in result.records[:6]
        ]
        request_context = {
            "material_id": request.material_id,
            "material_name": request.material_name,
            "paper_paths": request.paper_paths,
            "structure_paths": request.structure_paths,
            "constraints": request.constraints,
        }
        ai_result = self.agent_manager.plan_retrieval(request.task, request_context, evidence)
        if not ai_result:
            result.artifacts["retrieval_agent"] = {"agent_used": False}
            return

        missing_information = ai_result.get("missing_information", [])
        if isinstance(missing_information, list):
            for item in missing_information:
                if isinstance(item, str) and item.strip():
                    result.warnings.append(f"retrieval gap: {item.strip()}")

        result.artifacts["retrieval_agent"] = {
            "agent_used": True,
            "retrieval_focus": ai_result.get("retrieval_focus", ""),
            "priority_sources": ai_result.get("priority_sources", []),
            "coverage_assessment": ai_result.get("coverage_assessment", ""),
            "missing_information": missing_information if isinstance(missing_information, list) else [],
            "critique": ai_result.get("critique", ""),
        }

