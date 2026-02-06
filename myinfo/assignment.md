# SA Assignment: Meal Analysis

A health-tech customer is seeking guidance on architecting a scalable AI-powered meal analysis solution. They have provided a curated dataset of meal images paired with ground-truth JSON outputs that include input and output guardrail annotations, and meal inference with glycemic color classifications, ingredient lists, macro estimates, and textual guidance.

Your task is to use this dataset to design a solution composed of three agents:

1. Input guardrails  
2. Meal inference  
3. Output guardrails  

The customer is looking for guidance on an architecture that optimizes both accuracy and latency for production. Develop a proposed architecture and run evaluations using any Evals platform of your choice to identify which OpenAI model(s) deliver the best balance of accuracy and latency for image-to-JSON meal analysis.

You will then deliver your recommended architecture and model selection, along with evaluation results and recommended next steps for a limited pilot launch.

---

## 1) Agent Architecture

Develop three agents (prompts + response schema) architecture(s) to Eval. You may choose among models (e.g., gpt-4.1 and gpt-5, including mini and nano variants).

### 1.1 Input Guardrails Agent: `guardrailCheck`

Checks the uploaded image for:

- `is_food` (binary)
- `no_pii` (binary)
- `no_humans` (binary)
- `no_captcha` (binary)

### 1.2 Inference Agent (Vision → JSON): `mealAnalysis`

Given a meal image, return structured JSON (see schema in provided dataset) containing:

- `is_food` (boolean)
- `recommendation` (`green` | `yellow` | `orange` or `red`; neutral tone guidance)
- `guidance_message`, `meal_title`, `meal_description` (strings; **no medical claims**)
- `macros` object with numeric fields:
  - `calories`
  - `carbohydrates`
  - `fats`
  - `proteins`
- `ingredients[]` with `{ name, impact }` where `impact` ∈ `{ green, yellow, orange or red }`

### 1.3 Output Guardrails Agent: `safetyChecks`

Block text that contains:

- Emotional/judgmental language
- Risky substitutions (e.g., advising insulin changes, medication swaps)
- Treatment recommendations
- Medical diagnosis

---

## 3) Provided Assets

Dataset available with the provided ZIP file.

- Meal images: `images/<image_id>.jpg`
- Ground truth – Output samples (per image): `json-files/<image_id>.json`

**Convention:** `json_id = image_id` so you can basename-match image ↔ JSON.

---

## 4) Evals (Image → JSON)

Use a platform of your choice (such as Promptfoo) to build multimodal Evals.

### 4.1 Eval Metrics

Run Evals on the provided dataset to refine each agent’s prompts and identify the model that delivers the best balance of accuracy and latency for that agent.

#### 4.1.1 Guardrails (`guardrailCheck`)

- Key exact match
- Sample passes if **all booleans match**

#### 4.1.2 Safety (`safetyChecks`)

- Same as guardrails; exact boolean match

#### 4.1.3 Meal Analysis (`mealAnalysis`)

- **Recommendation correctness (3-class):** map `{ green, yellow, red }` with **orange treated as red**
- **is_food correctness:** boolean exact match
- **Macros score (0–100):**
  - per-field absolute percentage error (APE)
  - average → `macroMapeAvg`
  - `score = round(clamp01(1 − macroMapeAvg) * 100)`
- **Ingredients accuracy (0–100):**
  - normalized name match + same impact
  - `100 × (#matched_expected / #expected_valid)`
- **Text quality (0–100):**
  - LLM-as-judge rubric (correctness, clarity, neutrality) scored 0–5

##### MealAnalysis Weighted Composite (0–100)

- 50% recommendation exact (100/0)
- 30% average of `{ description, guidance, title }`
- 20% average of `{ macros_score_0_100, ingredients_accuracy_0_100 }`
- Normalize weights over available components if any are missing

### 4.2 Overall Composite Eval Score (Run Level)

Weight these to compute a single 0–100 score for a run over the 72 images:

- Guardrails: **20%** (aggregate per `guardrailCheck`)
- Meal inference: **50%** (use the `mealAnalysis` weighted composite)
- Safety checks: **30%** (aggregate per `safetyChecks`)

---

## 5) Deliverables

### 5.1 Eval Code Base with README

- State the Eval platform you chose
- Clear setup instructions
- Architecture diagram covering agents and sequencing of requests
- Rationale for selected models (quality and latency)

### 5.2 Evaluation Results

- Provide **P50 end-to-end latency (ms)** and **Composite Eval score** for your recommended architecture
- Additionally, include a table with the following columns for each agent:
  - Model(s) (list models you tested, with your recommended model at the top)
  - Eval Score
  - Average Input Tokens
  - Average Output Tokens
  - P50 Latency (ms)

### 5.3 Customer Email (300–500 words, excluding charts and tables)

- Your recommended solution architecture & models with trade-offs
- Simple diagram of the reference agent architecture
- Summarize the outcome of Evals
- Next steps for a production pilot, including risk areas and mitigations
