from pathlib import Path

from mat_agent.models import TaskRoute, UserRequest
from mat_agent.orchestrator import MaterialsAgentApp


def test_qa_run_writes_deliverable(tmp_path: Path):
    paper = tmp_path / "paper.txt"
    paper.write_text("BENZEN optical gap is discussed in the cited paper.", encoding="utf-8")

    app = MaterialsAgentApp(state_root=tmp_path / "state", enable_ai=False)
    deliverable = app.run(
        UserRequest(
            task="Extract cited papers for BENZEN",
            material_id="BENZEN",
            paper_paths=[str(paper)],
        )
    )

    assert deliverable.route == TaskRoute.QA
    assert Path(deliverable.deliverable_path).exists()
    assert deliverable.structured_output["evidence_count"] >= 2
    assert deliverable.structured_output["agent_used"] is False
    assert any(item["role"] == "retrieval_planner" for item in deliverable.agent_usage)
    assert any(item["role"] == "qa_agent" for item in deliverable.agent_usage)


def test_simulation_run_stops_at_approval_gate(tmp_path: Path):
    structure = tmp_path / "test.cif"
    structure.write_text("data_test\n_cell_length_a 1.0\n", encoding="utf-8")

    app = MaterialsAgentApp(state_root=tmp_path / "state", enable_ai=False)
    deliverable = app.run(
        UserRequest(
            task="Prepare a VASP single-point calculation",
            material_id="BENZEN",
            structure_paths=[str(structure)],
            constraints={"software": "vasp"},
        )
    )

    assert deliverable.route == TaskRoute.SIMULATION
    assert deliverable.structured_output["execution_status"] == "needs-approval"
    assert any(item["role"] == "retrieval_planner" for item in deliverable.agent_usage)
    assert any(item["role"] == "simulation_planner" for item in deliverable.agent_usage)


def test_fhi_aims_single_point_templates_are_generated(tmp_path: Path):
    app = MaterialsAgentApp(state_root=tmp_path / "state", enable_ai=False)
    deliverable = app.run(
        UserRequest(
            task="Generate FHI-aims single-point energy calculation scripts",
            material_id="BENZEN",
            constraints={"software": "fhi-aims"},
        )
    )

    generated_dir = Path(deliverable.deliverable_path).parent / "generated"

    assert deliverable.route == TaskRoute.SIMULATION
    assert (generated_dir / "geometry.in").exists()
    assert (generated_dir / "control.in").exists()
    assert (generated_dir / "run_fhi_aims.sh").exists()


