from mat_agent.models import TaskRoute, UserRequest
from mat_agent.router import route_request


def test_routes_qa_task():
    request = UserRequest(task="Extract cited papers for BENZEN", material_id="BENZEN")
    assert route_request(request) == TaskRoute.QA


def test_routes_simulation_task():
    request = UserRequest(
        task="Prepare a VASP single-point calculation",
        material_id="BENZEN",
        constraints={"software": "vasp"},
    )
    assert route_request(request) == TaskRoute.SIMULATION

