# Customer Email

> Intended audience includes nontechnical stakeholders, such as executives
> 

To whom it may concern,

Below is a brief summary of the best performing models based on evaluation scores, the architecture recommendation, warnings and limitations discovered, and next steps.

### Summary

The best performing models are `gpt-4.1-mini`, `gpt-4.1`, `gpt-4.1-nano` with eval scores of 95.9, 68.1, and 37.5 for inputGuardrail, mealAnalysis, and outputGuardrail agents, respectively.
However, these results are largely not reliable, specifically the results related to the Input and Output Guardrail Agents. So any production reliance on these evals may be a major liability concern for the health company. This is because the ground truth is both missing and inconsistent. On a positive note, the evaluations for mealAnalysis inference are more reliable because there is a more complete ground truth.

As an analogy, let's say each model is a student taking a multiple-choice test, and the ground truth is the answer key. But large parts of that answer key are missing *(we do not even know how much is missing)*, and other parts of the answer key are inconsistent or contradictory. If we grade the student answers using an incomplete answer key anyways, naturally, the resulting test scores will not be so meaningful.

### High Risk, Liability Warning

Since we have established that the ground truth itself is compromised (missing + noisy), rendering the evaluations largely incomplete, using these incomplete evals in production could expose end-users to risk and the health company to major liability.

The current ground truth may still be useful for benchmarking **meal analysis quality**, but it is **insufficient as the primary artifact for safety/compliance validation**. For concrete examples and details of missing/incorrect labels and composite-score masking risk, please see `README.md`.

## **Mitigations, Recommendations, next steps**

1. We are instructed to score the system with a blended metric/weighted composite score of 20/30/50. Using a blended metric, we risk overestimating safety performance. For production, safety and compliance must be evaluated as a **hard constraint**, not part of a weighted average.
Why? If we use a blended 20/30/50 composite score, a model can score high, e.g., 0.90, but **also fail the inputGuardrail check, and now the high score masks a catastrophic failure**.
Instead of a blended 20/30/50 weighted score, I propose a Two-Tier Scoring system: a **Compliance Gate Score** (pass/fail of guardrail + safety checks) **+ MealAnalysis Score**, and future evaluations should use this Two-Tier Scoring system.
> note: the Guardrail and Safety agents must meet a high threshold, such as ≥ 98%, before proceeding, like a gate. The threshold value will depend on the cost of agentic error or liability, and other factors, such as the company's explicit safety/compliance policy and risk tolerance for this product.

2. We need to expand the ground truth for "negative" cases to include images **with** PII and medical labels to ensure near 100% recall on Safety + Guardrail gates.

3. We need a data cleansing phase to clarify definitions and clean the ground truth by removing contradictions and noise.

> For concrete examples in the ground truth, please check the README, "High Risk, Liability Warning" section



## Recommended Architecture
<img src="charts/flowchart.png" alt="Meal analysis flowchart" width="50%">

> Linear, one agent after another.
> 

**Agent 1: Input Guardrail (The Bouncer)**
If sensitive data is detected from an image, e.g., PII, a human, etc., the process stops immediately.

**Agent 2: Inference Agent (The Analyst)**
For images that pass the previous step, this performs the heavy lifting, Meal Analysis.

**Agent 3: Output Guardrail (The Auditor)**
If it detects a medical diagnosis or risky treatment advice from Meal Analysis, this information is redacted before the user sees it.

Rationale for linear architecture:

- **Liability Mitigation:** By placing Safety and Guardrails as sequential blockers, hallucinations and liabilities are caught before they reach the user
- **Cost Efficiency and Simplicity:** Filtering out bad requests early blocks us from doing Meal analysis on invalid requests, saving token costs. An alternate architecture is to reduce inference times by running agents 1 and 2 concurrently, but this introduces increased cost and complexity in the form of asynchronous and atomic architecture requirements, which should not be the priority of this system as of yet. Due to the risk and nature of healthcare laws, it may be effective to solve things in the order of liability, accuracy, and then finally optimize for speed.
