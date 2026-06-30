# Project Name

One paragraph: what question this project answers, and what the core method is.

## Setup

```bash
uv venv
source .venv/bin/activate
uv pip install -e ".[dev]"
pre-commit install
```

Add inference-stack dependencies (vLLM, transformers, torch, etc.) to the
`inference` extra in `pyproject.toml` as needed, then:

```bash
uv pip install -e ".[dev,inference]"
```

## Repo structure

```
configs/        experiment configs and model lists, one YAML per experiment
data/raw/       untouched source data, read-only
data/processed/ cleaned/filtered data, produced by scripts in src/data/
data/splits/    fixed question IDs, seeds, eval splits saved as files
src/data/       dataset loading and preprocessing
src/generation/ model inference, prompting, sampling
src/metrics/    metric computation and downstream analysis (regressions, stats)
src/utils/      seeding, logging, shared helpers
scripts/        entry points that call into src/, one per pipeline stage
results/raw/    generation outputs, append-only, one subfolder per run
results/tables/ final tables for the paper
results/figures/ final figures for the paper
tests/          tests for metric computation logic
docs/           experimental design and pre-registered decisions
```

## Running an experiment

```bash
python scripts/run_generation.py --config configs/experiment1.yaml
python scripts/run_metrics.py --config configs/experiment1.yaml
```

## Conventions

- `results/raw/` is append-only. Rerunning writes to a new timestamped
  subfolder rather than overwriting prior results.
- Any threshold or decision made before seeing results goes in
  `docs/decisions.md`, dated.
- `data/raw/` is never edited directly.
- Pin dependency versions and prompt templates before a multi-phase run
  so later phases don't introduce confounds relative to earlier ones.

## Experimental design

See `docs/experimental_design.md` and `docs/decisions.md`.
