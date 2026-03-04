# Meal Analysis README

## At-a-Glance Summary

1. **LLMs used and why:** I evaluated OpenAI GPT models across pipeline roles and currently use `gpt-4.1` for `mealAnalysis` because it delivered the strongest score/quality performance for this dataset.
2. **Prompting and evaluation approach:** Prompting is structured per agent, then evaluated with a custom harness where `run_evals` generates raw model outputs and `score_evals` grades them against ground truth into row-level and aggregate metrics.
3. **Tradeoffs and limitations:** Ingredient-level inference remains the hardest task.

## Project Goal

This project supports a healthcare client by evaluating whether LLM-based vision workflows can (1) correctly identify the food shown in an image and (2) accurately infer likely ingredients from that image.

## Current Model Picks

Based on the latest evaluation run:

- **Current pick: `gpt-4.1`**
- **Why:** highest official `mealAnalysis` `eval_score` in this run: **44.96** (`72/72` scored rows).
- **Concise diagnostic vs `gpt-5.2-pro`:**
  - Official `gpt-5.2-pro` score is **30.81**, including `6/72` quota-failure rows.
  - Ignoring those failures, `gpt-5.2-pro` rises to **33.61**, still below `gpt-4.1`.
  - On parse-ok rows, ignoring text-quality component, `gpt-4.1` still leads (**64.23** vs **48.02**).
  - `gpt-5.2-pro` trails on key components: recommendation alignment (**53.03%** vs **66.67%**), macros (**50.47** vs **74.05**), and semantic ingredients (**20.49** vs **33.20**).

Conclusion: for this dataset and prompt setup, the gap is not just quota noise; `gpt-4.1` is currently the better `mealAnalysis` choice.

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
