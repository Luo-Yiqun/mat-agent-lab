# mat-agent-lab
11766 LLM Applications course project: Multi-Agent Framework for Autonomous Computational Materials Analysis.

This repo contains a runnable application skeleton for the simplified proposal flow:

`Input normalization -> Orchestrator -> Retrieval -> Validation -> QA or Simulation -> Deliverables`

## Project Layout

- `src/mat_agent/`: orchestration, routing, AI agents, validators, and CLI
- `LiteratureReview/`: preserved legacy literature-review workflow
- `config.json`: local AI gateway config, kept out of git
- `run-artifacts/`: per-run state snapshots plus `deliverable.json` and `deliverable.txt`

## AI Agents

The new app uses AI only where it adds value and keeps deterministic validators in charge of control:

- `retrieval_planner` uses `gpt-5-mini` for retrieval planning and lightweight critique
- `qa_agent` uses `claude-sonnet-4-20250514-v1:0` for evidence-grounded answer synthesis
- `simulation_planner` uses `gpt-5` for structured simulation planning
- deterministic routing, evidence validation, input gating, and result gating remain non-LLM

If AI is unavailable or disabled, the app falls back to deterministic behavior and records that in the deliverable.

## Agent Tools

Each AI agent is given a constrained tool/context bundle rather than arbitrary repo access:

- `retrieval_planner` in `src/mat_agent/pipelines/retrieval.py`
  tools: `local_file_reader`, `pdf_text_extractor`, `legacy_literature_review_backend`
- `qa_agent` in `src/mat_agent/pipelines/qa.py`
  tools: `grounded_evidence_bundle`, `citation_pack`
- `simulation_planner` in `src/mat_agent/pipelines/simulation.py`
  tools: `retrieval_context_bundle`, `software_hint_selector`, `human_approval_gate`

These are not API-side function-calling tools yet. They are explicit context/tool contracts enforced by the application layer and recorded in `deliverable.json` under `agent_usage`.

## Config

Create a local root-level `config.json` that matches `config.example.json`. The new AI layer reads:

- `api_key`
- `base_url`

## CLI Usage

The CLI supports five main workflow shapes.

### 1. CSD reference code => cited papers

This routes through retrieval and QA, and uses the legacy `LiteratureReview` cited-paper backend. It reads cached results from `LiteratureReview/citing.json` first and, when the legacy environment is available, can fall through to the original `CCDCCitingPaper` workflow.

```powershell
$env:PYTHONPATH='src'
python -m mat_agent `
  --task "Find cited papers for CSD reference code BENZEN" `
  --material-id BENZEN `
  --route qa
```

Legacy backend prerequisites for live lookup:

- CCDC Python API
- Selenium
- Chrome/Chromedriver available to the legacy workflow

### 2. CSD reference code => FHI-aims SPE calculation scripts

This routes through simulation planning and writes placeholder FHI-aims single-point files into `run-artifacts/<run_id>/generated/`.

```powershell
$env:PYTHONPATH='src'
python -m mat_agent `
  --task "Generate FHI-aims single-point energy calculation scripts" `
  --material-id BENZEN `
  --software fhi-aims `
  --route simulation
```

Generated files currently include:

- `geometry.in`
- `control.in`
- `run_fhi_aims.sh`

### 3. CSD reference code + question => answer

This keeps the material identifier in context and answers through the QA route.

```powershell
$env:PYTHONPATH='src'
python -m mat_agent `
  --task "What are the cited papers and likely optical-gap evidence for BENZEN?" `
  --material-id BENZEN `
  --route qa `
  --json
```

### 4. Papers in PDF + question => answer

This loads one or more local PDFs and prints a direct answer plus supporting figure/table/sentence references.

```powershell
$env:PYTHONPATH='src'
python -m mat_agent `
  --task "What is the reported band gap in these papers?" `
  --paper .\papers\paper1.pdf `
  --paper .\papers\paper2.pdf `
  --route qa
```

### 5. Input generation => QE or FHI-aims files

There are two related input-generation entry points:

- `python -m mat_agent` for structure-conditioned simulation script generation
- `python -m mat_agent.input_file_generator` for prompt-only few-shot input generation over local examples

Structure file => QE SPE calculation scripts:

```powershell
$env:PYTHONPATH='src'
python -m mat_agent `
  --task "Generate Quantum ESPRESSO single-point energy calculation scripts" `
  --structure .\structures\benzene.cif `
  --software "quantum espresso" `
  --route simulation
```

Generated files currently include:

- `qe_scf.in`
- `run_qe.sh`

The CLI prints the generated file paths after the run completes.

Prompt requirements => FHI-aims or QE input file (few-shot):

This uses a standalone few-shot generator over local examples in `data/FHI-aims/*` and `data/QE/*`, and writes outputs into `test_generated_files/`.

```powershell
$env:PYTHONPATH='src'
python -m mat_agent.input_file_generator `
  --software "fhi-aims" `
  --prompt "Generate a tight single-point setup for an organic crystal with 4x4x4 k-grid."
```

```powershell
$env:PYTHONPATH='src'
python -m mat_agent.input_file_generator `
  --software "qe" `
  --prompt "Create an SCF input for a periodic system with 60 Ry cutoff and dense k-point mesh." `
  --output-name qe_custom.in
```

To disable AI and force the deterministic fallback for any of the above:

```powershell
$env:PYTHONPATH='src'
python -m mat_agent --task "Extract cited papers for BENZEN" --material-id BENZEN --no-ai
```

## Testing

```powershell
$env:PYTHONPATH='src'
pytest
```
