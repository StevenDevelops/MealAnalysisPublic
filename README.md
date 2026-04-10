# Meal Analysis README

## At-a-Glance Summary

Have you ever wondered which foundational OpenAI model, given images of food, can most accurately what the food is? How about its ingredients? How about with the least latency? 
I implemented a custom Python harness to evaluate just that, evaluating various OpenAI GPT models. Given food images, I score the meal and ingredients inference accuracy.

1. **LLMs used and why:** I evaluated OpenAI GPT models across pipeline roles and `gpt-4.1` has the strongest score/quality performance for this dataset.
2. **Prompting and evaluation approach:** Prompting is structured per model, then evaluated with a custom harness where `run_evals` generates raw model outputs and `score_evals` grades them against ground truth into row-level and aggregate metrics.
3. **Tradeoffs and limitations:** Ingredient-level inference remains the hardest task.

## Project Goal

This project supports a healthcare client by evaluating whether LLM-based vision workflows can do:
1. Meal inference - model can correctly classify or identify the food shown in an image
2. Ingredients inference - model can accurately infer likely ingredients from that image
> Each model's outputs is scored against a ground truth or "answer key" using semantic matching of ingredients e.g. "aubergine" and "eggplant" are the same ingredient

## Current Model Picks

These models were tested:

- `gpt-4o`
- `gpt-4.1-mini`
- `gpt-4.1`
- `gpt-5-mini`
- `gpt-5`
- `gpt-5.2-pro`

Based on the latest evaluation run:

| model | eval_score | meal_inference_score | ingredients_inference_score | p50_latency_ms |
|---|---:|---:|---:|---:|
| gpt-4.1 | 70.15 | 50.00 | 51.10 | 3921.73 |
| gpt-5 | 64.69 | 43.06 | 51.92 | 5874.02 |
| gpt-4o | 60.40 | 47.22 | 40.13 | 7381.98 |
| gpt-4.1-mini | 59.25 | 36.11 | 42.42 | 4527.30 |
| gpt-5.2-pro | 59.05 | 30.56 | 46.89 | 23961.46 |
| gpt-5-mini | 52.67 | 33.33 | 44.45 | 5330.76 |

- **Current pick: `gpt-4.1`**
- **Why:** highest overall `eval_score`, strong `meal_inference_score`, near-highest `ingredients_inference_score`, and the lowest `p50_latency_ms` in this comparison set.
- **How `eval_score` is computed (brief):** weighted meal composite = `50% recommendation exact` + `30% text quality` + `20% average(macros_score, ingredients_accuracy)`.

Conclusion: for this dataset and prompt setup, `gpt-4.1` is the best overall model choice.

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

For faster runs, increase parallel workers (default is `10`):

```bash
python evals/run_evals.py --max-workers 20
```

For a smaller run, sample a subset:

```bash
python evals/run_evals.py --num-samples 10
```

**Step 2** - Score evaluations (writes `results_scored.csv` and `agent_model_summary.csv`):

```bash
python evals/score_evals.py
```

For faster scoring, increase workers for row scoring and ingredient matcher batches:

```bash
python evals/score_evals.py --score-workers 20 --ingredients-matcher-workers 20
```

### Output Files

- `results.csv`: Raw model outputs for each evaluation job.
- `results_scored.csv`: Row-level scoring details against ground truth.
- `agent_model_summary.csv`: Aggregated performance by `(agent, model)`.

**Agent: Meal Analysis**
Performs food recognition and ingredient inference.

## Future Improvements

- Expand dataset coverage to improve food and ingredient matching confidence.
- Parallelize more of the scoring path to reduce total runtime.
- Continue tuning prompts and scoring logic for ingredient-level accuracy.
