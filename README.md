# 11766-mat-agent-lab
11766 LLM Applications course project: Multi-Agent Framework for Autonomous Computational Materials Analysis.

This repo contains a runnable application skeleton for the simplified proposal flow:

`Input normalization -> Orchestrator -> Retrieval -> Validation -> QA or Simulation -> Deliverables`

## Project Layout

- `src/mat_agent/`: orchestration, routing, AI agents, validators, and CLI
- `LiteratureReview/`: preserved legacy literature-review workflow
- `config.json`: local AI gateway config, kept out of git
- `run-artifacts/`: per-run state snapshots and deliverables

## AI Agents

The new app uses AI only where it adds value and keeps deterministic validators in charge of control:

- `retrieval_planner` uses `gpt-5-mini` for retrieval planning and lightweight critique
- `qa_agent` uses `claude-sonnet-4-20250514-v1:0` for evidence-grounded answer synthesis
- `simulation_planner` uses `gpt-5` for structured simulation planning
- deterministic routing, evidence validation, input gating, and result gating remain non-LLM

If AI is unavailable or disabled, the app falls back to deterministic behavior and records that in the deliverable.

## Agent Tools

Each AI agent is given a constrained tool/context bundle rather than arbitrary repo access:

- `retrieval_planner` in [src/mat_agent/pipelines/retrieval.py](c:/Users/18000/OneDrive/Desktop/11766-mat-agent-lab/src/mat_agent/pipelines/retrieval.py)
  tools: `local_file_reader`, `pdf_text_extractor`, `legacy_literature_review_handoff`
- `qa_agent` in [src/mat_agent/pipelines/qa.py](c:/Users/18000/OneDrive/Desktop/11766-mat-agent-lab/src/mat_agent/pipelines/qa.py)
  tools: `grounded_evidence_bundle`, `citation_pack`
- `simulation_planner` in [src/mat_agent/pipelines/simulation.py](c:/Users/18000/OneDrive/Desktop/11766-mat-agent-lab/src/mat_agent/pipelines/simulation.py)
  tools: `retrieval_context_bundle`, `software_hint_selector`, `human_approval_gate`

These are not API-side function-calling tools yet. They are explicit context/tool contracts enforced by the application layer and recorded in `deliverable.json` under `agent_usage`.

## Config

Create a local root-level `config.json` that matches `config.example.json`. The new AI layer reads:

- `api_key`
- `base_url`

## CLI Usage

```powershell
$env:PYTHONPATH='src'
python -m mat_agent --task "Extract cited papers for BENZEN" --material-id BENZEN --json
```

```powershell
$env:PYTHONPATH='src'
python -m mat_agent --task "Prepare a VASP single-point calculation" --material-id BENZEN --software vasp
```

To disable AI and force the deterministic fallback:

```powershell
$env:PYTHONPATH='src'
python -m mat_agent --task "Extract cited papers for BENZEN" --material-id BENZEN --no-ai
```

## Testing

```powershell
$env:PYTHONPATH='src'
pytest
```
