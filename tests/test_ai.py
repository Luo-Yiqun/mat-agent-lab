from mat_agent.ai import AgentManager


def test_agent_manager_disabled_skips_calls():
    manager = AgentManager(enable_ai=False)

    result = manager.answer_question("What is the optical gap?", [])

    assert result is None
    usage = manager.get_usage()
    assert usage
    assert usage[0]["status"] == "skipped"

