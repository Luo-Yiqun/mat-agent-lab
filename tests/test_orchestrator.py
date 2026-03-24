from pathlib import Path

from mat_agent.ai import AgentManager
from mat_agent.legacy.literature_review import LegacyLiteratureReviewAdapter
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
    assert Path(deliverable.deliverable_text_path).exists()
    assert deliverable.structured_output["evidence_count"] >= 2
    assert deliverable.structured_output["agent_used"] is False
    assert any(item["role"] == "retrieval_planner" for item in deliverable.agent_usage)
    assert any(item["role"] == "qa_agent" for item in deliverable.agent_usage)
    assert "Summary:" in Path(deliverable.deliverable_text_path).read_text(encoding="utf-8")


def test_pdf_question_returns_direct_answer_with_supporting_references(tmp_path: Path):
    paper = tmp_path / "bandgap.txt"
    paper.write_text(
        (
            "Figure 2 shows the absorption onset for the crystal.\n"
            "The reported band gap is 2.30 eV for the polymorph studied here.\n"
            "Table 1 summarizes the optical measurements.\n"
        ),
        encoding="utf-8",
    )

    app = MaterialsAgentApp(state_root=tmp_path / "state", enable_ai=False)
    deliverable = app.run(
        UserRequest(
            task="What is the reported band gap in this paper?",
            paper_paths=[str(paper)],
            route_hint=TaskRoute.QA,
        )
    )

    assert deliverable.route == TaskRoute.QA
    assert "Direct answer: The reported gap is 2.30 eV." in deliverable.summary
    assert deliverable.structured_output["direct_answer"] == "The reported gap is 2.30 eV."
    assert any(
        reference["locator"].lower().startswith("figure 2") or reference["locator"].lower().startswith("fig")
        for reference in deliverable.structured_output["supporting_references"]
    )
    assert any(
        "2.30 eV" in reference["excerpt"]
        for reference in deliverable.structured_output["supporting_references"]
    )


def test_csd_to_papers_uses_legacy_literature_review_backend(tmp_path: Path, monkeypatch):
    def fake_resolve(self, material_id: str, allow_live: bool = False):
        assert material_id == "BENZEN"
        assert allow_live is True
        return {
            "material_id": material_id,
            "legacy_root": str((tmp_path / "LiteratureReview").resolve()),
            "backend": "LiteratureReview.CCDCCitingPaper",
            "status": "cached",
            "source_json": str((tmp_path / "citing.json").resolve()),
            "original_papers": [
                {
                    "authors": "A. Author",
                    "journal": "J. Chem.",
                    "year": "2024",
                    "doi": "10.1000/example",
                }
            ],
            "citing_papers": [
                {
                    "google_scholar_title": "A cited paper for BENZEN",
                    "google_scholar_snippet": "This paper cites the BENZEN polymorph.",
                    "publication_link": "https://example.com/paper",
                }
            ],
            "environment": {},
        }

    monkeypatch.setattr(LegacyLiteratureReviewAdapter, "resolve_citing_papers", fake_resolve)

    app = MaterialsAgentApp(state_root=tmp_path / "state", enable_ai=False)
    deliverable = app.run(
        UserRequest(
            task="Find cited papers for CSD reference code BENZEN",
            material_id="BENZEN",
            route_hint=TaskRoute.QA,
        )
    )

    assert deliverable.route == TaskRoute.QA
    assert any(citation["title"] == "A cited paper for BENZEN" for citation in deliverable.citations)
    assert deliverable.artifacts["retrieval"]["artifacts"]["legacy_literature_review"]["backend"] == "LiteratureReview.CCDCCitingPaper"
    assert deliverable.artifacts["retrieval"]["coverage"]["legacy_cited_papers"] is True


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


def test_qe_single_point_templates_are_generated_from_structure(tmp_path: Path):
    structure = tmp_path / "benzene.cif"
    structure.write_text("data_benzene\n_cell_length_a 1.0\n", encoding="utf-8")

    app = MaterialsAgentApp(state_root=tmp_path / "state", enable_ai=False)
    deliverable = app.run(
        UserRequest(
            task="Generate Quantum ESPRESSO single-point energy calculation scripts",
            structure_paths=[str(structure)],
            constraints={"software": "quantum espresso"},
        )
    )

    generated_dir = Path(deliverable.deliverable_path).parent / "generated"

    assert deliverable.route == TaskRoute.SIMULATION
    assert (generated_dir / "qe_scf.in").exists()
    assert (generated_dir / "run_qe.sh").exists()
    assert deliverable.structured_output["generated_files"]["qe_scf.in"].endswith("qe_scf.in")
    assert "Generated Files:" in Path(deliverable.deliverable_text_path).read_text(encoding="utf-8")


def test_qe_single_point_templates_are_generated_with_ai_planner(tmp_path: Path, monkeypatch):
    structure = tmp_path / "benzene.cif"
    structure.write_text("data_benzene\n_cell_length_a 1.0\n", encoding="utf-8")

    def fake_plan(self, task: str, context: dict):
        return {
            "summary": "AI planned a QE single-point calculation.",
            "software": "Quantum ESPRESSO",
            "requires_human_approval": True,
            "parameters": {"ecutwfc": 80},
            "assumptions": ["Using a placeholder SCF setup."],
        }

    monkeypatch.setattr(AgentManager, "plan_simulation", fake_plan)

    app = MaterialsAgentApp(state_root=tmp_path / "state", enable_ai=False)
    deliverable = app.run(
        UserRequest(
            task="Generate Quantum ESPRESSO single-point energy calculation scripts",
            structure_paths=[str(structure)],
            constraints={"software": "quantum espresso"},
        )
    )

    generated_dir = Path(deliverable.deliverable_path).parent / "generated"

    assert deliverable.route == TaskRoute.SIMULATION
    assert (generated_dir / "qe_scf.in").exists()
    assert (generated_dir / "run_qe.sh").exists()
    assert deliverable.structured_output["generated_files"]["qe_scf.in"].endswith("qe_scf.in")


