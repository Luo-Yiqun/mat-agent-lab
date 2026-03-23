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

- `qa_agent` uses `claude-sonnet-4-20250514-v1:0` for evidence-grounded answer synthesis
- `simulation_planner` uses `gpt-5` for structured simulation planning
- deterministic routing, evidence validation, input gating, and result gating remain non-LLM

If AI is unavailable or disabled, the app falls back to deterministic behavior and records that in the deliverable.

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
