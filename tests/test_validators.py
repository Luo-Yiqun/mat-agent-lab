from mat_agent.models import EvidenceRecord, RetrievalResult, UserRequest
from mat_agent.validators import EvidenceValidator


def test_evidence_validator_accepts_provenanced_record():
    request = UserRequest(task="Extract cited papers", material_id="BENZEN")
    result = RetrievalResult(
        records=[
            EvidenceRecord(
                title="BENZEN",
                content="User provided material identifier: BENZEN",
                source_type="material-id",
                provenance={"source_id": "BENZEN"},
            )
        ]
    )

    gate = EvidenceValidator.validate(request, result)

    assert gate.passed is True
    assert 0.0 <= gate.confidence <= 1.0


