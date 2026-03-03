# Meal Analysis README

## Project Goal

This project supports a healthcare client by evaluating whether LLM-based vision workflows can (1) correctly identify the food shown in an image and (2) accurately infer likely ingredients from that image.

- **Evaluation Results** are summarized in `evaluation_results.ipynb` at the project root.

## Current Model Picks

Based on the latest evaluation run:

| results will be updated here shortly

## Eval Platform

This repository uses a custom Python evaluation harness because it is the most direct fit for this project's vision-to-structured-output workflow and agent-by-agent scoring.

- `run_evals` executes each configured `(agent, model)` evaluation job, sends inputs through the pipeline, and writes raw outputs to `outputs/results.csv`.
- `score_evals` compares those raw outputs against ground truth, computes row-level and aggregate metrics, and writes scored reports (`results_scored.csv` and `agent_model_summary.csv`).

## Setup Instructions

**Step 0.0** - Add an OpenAI API key in an `.env` file at the project root:

```bash
OPENAI_API_KEY=<your OpenAI API key goes here>
```

**Step 0.1** - Install dependencies and activate the conda environment:

```bash
conda env create -f environment.yml
conda activate healthevals
```

**Step 1** - Run evaluations (outputs are written to `outputs/results.csv`):

```bash
python evals/run_evals.py
```

For a smaller run, sample a subset:

```bash
python evals/run_evals.py --num-samples 10
```

**Step 2** - Score evaluations (writes `results_scored.csv` and `agent_model_summary.csv`):

```bash
python evals/score_evals.py
```

### Output Files

- `results.csv`: Raw model outputs for each evaluation job.
- `results_scored.csv`: Row-level scoring details against ground truth.
- `agent_model_summary.csv`: Aggregated performance by `(agent, model)`.

**Agent: Meal Analysis**
Performs food recognition and ingredient inference.

## Future Improvements

- Expand dataset coverage to improve food and ingredient matching confidence.
- Run evaluation and scoring jobs in parallel to reduce total runtime.
- Continue tuning prompts and scoring logic for ingredient-level accuracy.
