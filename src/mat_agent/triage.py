from __future__ import annotations

from .models import FailureAction, TriageDecision


class FailureTriage:
    @staticmethod
    def decide(stage: str, issues: list[str]) -> TriageDecision:
        if not issues:
            return TriageDecision(FailureAction.DEGRADE, f"No issues recorded for stage '{stage}'.")

        joined = " ".join(issues).lower()
        if "no evidence records" in joined:
            return TriageDecision(
                action=FailureAction.ESCALATE,
                message="Retrieval did not produce evidence. Provide papers, a structure file, or enable a live retrieval backend.",
                retry_stage="retrieval",
            )
        if "missing provenance" in joined:
            return TriageDecision(
                action=FailureAction.RETRY,
                message="Evidence exists but provenance is incomplete. Retry retrieval with stricter source tracking.",
                retry_stage="retrieval",
            )
        return TriageDecision(
            action=FailureAction.DEGRADE,
            message=f"Stage '{stage}' produced validation issues. Returning a partial deliverable instead of failing hard.",
            retry_stage=None,
        )

