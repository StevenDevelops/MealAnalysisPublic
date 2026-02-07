# context.md — OAI Meal Analysis Takehome (for Codex)

## How to use this context
- This file is the “single source of truth” for our plan, risks, and implementation approach.
- **Also read the official assignment instructions** at: `myinfo/context.md` (relative to repo root).
- The user will pass the entire `myinfo/` folder into Codex on each interaction. Treat `myinfo/context.md` as authoritative for requirements and schemas.

---

## 1) Assignment summary (what we must deliver)
We are designing and evaluating a **3-agent meal analysis pipeline** using a provided dataset of meal images + ground-truth JSON outputs.

### Agents
1. **Input Guardrails Agent (`guardrailCheck`)**
   - From an uploaded image, output booleans:
     - `is_food`
     - `no_pii`
     - `no_humans`
     - `no_captcha`

2. **Inference Agent (`mealAnalysis`)** — Vision → Structured JSON
   - From a meal image, output JSON containing:
     - `is_food` (boolean)
     - `recommendation` (green | yellow | orange/red)
     - `guidance_message`, `meal_title`, `meal_description` (neutral, no medical claims)
     - `macros` numeric fields: calories, carbohydrates, fats, proteins
     - `ingredients[]`: `{ name, impact }` where impact ∈ {green, yellow, orange/red}

3. **Output Guardrails Agent (`safetyChecks`)**
   - Detect/block unsafe output text:
     - Emotional/judgmental language
     - Risky substitutions / medication changes
     - Treatment recommendations
     - Medical diagnosis

### Evals requirements (core)
- Build multimodal evals (image + text).
- Compute agent-level metrics and overall composite score (run-level).
- Report **P50 end-to-end latency (ms)** and token usage.
- Produce a per-agent table (models tested, eval score, avg input/output tokens, P50 latency).
- Provide architecture diagram + rationale + recommended next steps for a pilot.

---

## 2) Key discussion points, pitfalls, and risk areas (IMPORTANT)

### High Risk, Liability Warning
The ground truth is compromised. The standard at which we measure model accuracy is **compromised (degenerate dataset)**, so evaluation scores are **incomplete** and may not reflect real-world safety/compliance performance, creating potential legal and health risk.

- **Risk Area 1 — Missing negative examples**
  - No/insufficient ground truths for failed Safety Checks and Guardrail Checks (e.g., `is_food: false`, `no_humans: false`, unsafe medical advice, judgmental language).
- **Risk Area 2 — Noisy or incorrect labels**
  - Some images appear to include human hands but are labeled `no_humans: true`, suggesting inconsistent labeling.
- **Risk Area 3 — Composite scoring can hide catastrophic failures**
  - The requested 20/30/50 composite can still score “high” even if a guardrail or safety check fails, which is unacceptable for healthcare-like contexts.

### Mitigations (condensed)
Proceed with the provided evals as a benchmark, but clearly document that safety/guardrail quality is under-measured; for a real pilot, expand and clean the dataset and treat safety/guardrails as a hard compliance gate (near-100% threshold) rather than a weighted average.

---

## 3) Plan of attack (implementation approach)
We will implement a **pure Python eval harness + OpenAI SDK** (no Promptfoo/LangChain required).

### Coding artifacts (recommended)
- `src/eval_agents.py` — runs model calls + scoring; writes CSV
- `src/analyze_evals.py` — reads CSV; computes aggregated metrics + summary tables
- `notebooks/analysis.ipynb` (optional) — plots charts from CSV (latency distributions, score comparisons)

### Part 1 — Eval harness (core script)
Goal: For every `(agent, model, example)` → call model, score it, dump a CSV row.

High-level components:
- **Config & dataset loading**
  - Define `AGENTS` and `MODELS`
  - Load images and ground-truth JSON pairs
- **Agent runners**
  - `guardrailCheck`: image → guardrail booleans
  - `mealAnalysis`: image → meal JSON (recommendation, text, macros, ingredients)
  - `safetyChecks`: text → safety booleans
  - Each runner records latency + token usage and returns parsed JSON
- **Metric functions**
  - Guardrails/safety: exact boolean match (pass/fail)
  - MealAnalysis: recommendation correctness, `is_food` correctness, macros score (MAPE→0–100), ingredients accuracy (0–100), text quality (judge or heuristic), plus weighted composite with normalization
- **CSV output**
  - One row per agent-model-example, including latency and token counts

Output: `eval_results.csv`.

### Part 2 — Analysis script (CSV → metrics & tables)
Goal: Turn `eval_results.csv` into the numbers required by the assignment.

High-level outputs:
- Per-agent/per-model:
  - Eval score
  - Avg input tokens
  - Avg output tokens
  - P50 latency (ms)
- Run-level:
  - Overall composite score (20% guardrails, 50% meal analysis, 30% safety)
  - P50 end-to-end latency (ms)

This is where we decide the recommended model per agent.

### Part 3 — Visualization (optional; CSV → charts)
Goal: Produce simple charts for storytelling and debugging.
- Latency distributions (histograms) per agent/model
- Score comparisons (bar charts) per agent/model

---

## 4) Model selection approach (initial wave)
We will start with a pragmatic “small vs big” comparison per agent (wave 1) to balance time vs signal:
- Vision guardrails: fast model vs strong model
- Vision meal analysis: fast model vs strong model
- Text safety: cheapest/fast vs slightly stronger small

Then choose the best model per agent from the evaluated set (mix-and-match is allowed).

(Exact model names should be verified/updated per `myinfo/context.md` requirements and current OpenAI model availability in the environment.)

---

## 5) Notes for Codex (implementation guardrails)
- Keep the harness generic and configurable: agents/models lists should be easy to change.
- Log **per-example**:
  - latency_ms
  - input_tokens / output_tokens
  - key sub-metrics needed for composites
- Fail gracefully on JSON parse errors:
  - Record an error flag + set metric scores to 0 for that row (and keep going).
- Keep outputs deterministic:
  - Use low temperature
  - Force JSON-only output via schema/prompt rules
- Don’t over-engineer:
  - No orchestration frameworks
  - No web services
  - No tool-calling “agents” beyond the three LLM calls