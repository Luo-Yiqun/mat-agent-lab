from __future__ import annotations

from .models import TaskRoute, UserRequest


SIMULATION_KEYWORDS = {
    "calculate",
    "calculation",
    "compute",
    "simulation",
    "simulate",
    "single-point",
    "single point",
    "relax",
    "optimization",
    "optimize",
    "vasp",
    "quantum espresso",
    "espresso",
    "berkeleygw",
    "dft",
    "gw",
    "bse",
    "band structure",
    "density of states",
    "dos",
    "submit job",
}

QA_KEYWORDS = {
    "extract",
    "summarize",
    "summary",
    "review",
    "literature",
    "paper",
    "papers",
    "citation",
    "cited",
    "find papers",
    "what is",
    "question",
    "answer",
}


def route_request(request: UserRequest) -> TaskRoute:
    if request.route_hint:
        return request.route_hint

    text = " ".join(
        [
            request.task or "",
            request.constraints.get("software", ""),
            request.material_id or "",
            request.material_name or "",
        ]
    ).lower()

    sim_score = sum(keyword in text for keyword in SIMULATION_KEYWORDS)
    qa_score = sum(keyword in text for keyword in QA_KEYWORDS)

    if request.structure_paths:
        sim_score += 1
    if request.paper_paths:
        qa_score += 1
    if request.constraints.get("software"):
        sim_score += 2

    return TaskRoute.SIMULATION if sim_score > qa_score else TaskRoute.QA

