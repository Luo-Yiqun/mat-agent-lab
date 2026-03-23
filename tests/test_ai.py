from mat_agent.ai import AgentManager


def test_agent_manager_disabled_skips_calls():
    manager = AgentManager(enable_ai=False)

    result = manager.answer_question("What is the optical gap?", [])

    assert result is None
    usage = manager.get_usage()
    assert usage
    assert usage[0]["status"] == "skipped"


def test_retrieval_planner_disabled_skips_calls():
    manager = AgentManager(enable_ai=False)

    result = manager.plan_retrieval("Find cited papers", {"material_id": "BENZEN"}, [])

    assert result is None
    usage = manager.get_usage()
    assert usage
    assert usage[0]["role"] == "retrieval_planner"
    assert "local_file_reader" in usage[0]["tools"]
