# 11766-mat-agent-lab
11766 LLM Applications course project: Multi-Agent Framework for Autonomous Computational Materials Analysis.

This repo now contains a runnable application skeleton for the simplified proposal flow:

`Input normalization -> Orchestrator -> Retrieval -> Validation -> QA or Simulation -> Deliverables`

## Project Layout

- `src/mat_agent/`: new application package with routing, orchestration, validators, and deliverable generation
- `LiteratureReview/`: preserved legacy literature-review workflow and utilities
- `run-artifacts/`: generated state and deliverables for local runs

## What Works

- Deterministic routing between `qa` and `simulation`
- State tracking and JSON deliverables for every run
- Retrieval over local papers and structure files
- Legacy `LiteratureReview` environment detection and handoff metadata
- Dry-run simulation planning with an explicit approval gate

## What Is Preserved

The existing `LiteratureReview` code is kept as the legacy retrieval backend. Its existing functions were not revised. The new application wraps it behind an adapter so the rest of the repo can evolve without rewriting that workflow.

## CLI Usage

Run the new package directly from the repo root:

```powershell
$env:PYTHONPATH='src'
python -m mat_agent --task "Extract cited papers for BENZEN" --material-id BENZEN --json
```

```powershell
$env:PYTHONPATH='src'
python -m mat_agent --task "Prepare a VASP single-point calculation" --material-id BENZEN --software vasp
```

Each run writes:

- `run-artifacts/<run_id>/state.json`
- `run-artifacts/<run_id>/deliverable.json`

## Legacy Notes

- `LiteratureReview/config.json` is now a placeholder and should be populated locally before using the legacy scripts.
- `LiteratureReview/config.example.json` documents the expected shape.
- If the previously committed OpenAI key was real, it should be rotated.

## Testing

```powershell
$env:PYTHONPATH='src'
pytest
```
