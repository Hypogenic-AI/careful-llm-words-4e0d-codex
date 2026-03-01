# Research Plan: An LLM That's Careful With Its Words

## Motivation & Novelty Assessment

### Why This Research Matters
If simple prompting constraints can make model outputs more careful at the sentence level, this offers a low-cost intervention for factuality and safety without retraining. This matters for deployment settings where risky wording, overconfident claims, or toxic continuations are unacceptable. The main beneficiaries are teams shipping LLM writing assistants, QA systems, and conversational tools that must balance fluency with caution.

### Gap in Existing Work
Prior work studies chain-of-thought, self-consistency, tree search, inner monologue, and process supervision, but the literature review shows little direct testing of a strict "between every sentence" thought-token requirement for open-ended generation. Existing evaluations also emphasize final-task accuracy more than sentence-level caution and qualitative writing differences. There is also an unresolved confound: gains from thought prompting may come from extra token budget rather than better reasoning.

### Our Novel Contribution
We test a concrete, controlled variant: enforce explicit inter-sentence tokens for every sentence and compare it to both a direct baseline and a token-budget control with non-reasoning pause tokens. This isolates whether qualitative differences are specific to deliberate thought traces versus merely inserting extra tokens. We evaluate across truthfulness, reasoning accuracy, and safety continuation behavior.

### Experiment Justification
- Experiment 1 (TruthfulQA generation): Needed to test whether inter-sentence thought tokens improve factual caution/truthfulness versus baseline and token-budget control.
- Experiment 2 (GSM8K reasoning): Needed to test whether this intervention improves objective reasoning correctness (numeric exact match).
- Experiment 3 (RealToxicityPrompts continuation): Needed to test whether forced deliberation changes unsafe wording behavior under difficult prompts.
- Experiment 4 (Qualitative style analysis): Needed to characterize how outputs differ (hedging, specificity, sentence structure, verbosity, self-correction).

## Research Question
Does requiring an LLM to emit explicit "thinking tokens" between every sentence produce more careful, qualitatively different text than standard prompting, beyond the effect of extra token usage alone?

## Background and Motivation
The resource set includes CoT, Zero-shot CoT, Self-Consistency, Tree-of-Thought, Deliberate-then-Generate, Think-Before-You-Speak, and Quiet-STaR. Together these suggest deliberate intermediate reasoning can help some tasks, but they do not directly test a strict sentence-by-sentence insertion protocol in open-ended outputs. We evaluate this protocol as an inference-time intervention with no retraining.

## Hypothesis Decomposition
- H1 (Truthfulness): Inter-sentence thinking tokens increase judged truthfulness/carefulness versus direct baseline.
- H2 (Reasoning): Inter-sentence thinking tokens increase GSM8K exact-match accuracy versus direct baseline.
- H3 (Safety): Inter-sentence thinking tokens reduce moderation risk scores on RealToxicityPrompts continuations.
- H4 (Qualitative shift): Inter-sentence thinking tokens increase hedging/qualification and self-correction markers, producing visibly more deliberative style.
- Alternative explanation A1: Any changes come from longer outputs, not reasoning. Addressed via a pause-token control.

Independent variable:
- Prompt condition (`direct`, `pause_control`, `think_between_sentences`).

Dependent variables:
- Truthfulness judge score (0-1), GSM8K exact match, moderation category scores, flagged rate, and qualitative style metrics.

## Proposed Methodology

### Approach
Use one strong real API model (`gpt-4.1`) for generation across all conditions and tasks. Use separate model-based judging (`gpt-4.1-mini`) for TruthfulQA scoring to reduce evaluator variance. Use OpenAI moderation endpoint for toxicity/safety scoring. Keep temperature fixed to reduce stochastic variance and improve comparability.

### Experimental Steps
1. Load pre-downloaded datasets from `datasets/` and sample fixed subsets with a reproducible seed.
2. Define three prompt conditions with explicit templates.
3. Run generation API calls with retry logic and JSONL logging.
4. Score outputs:
   - TruthfulQA with LLM judge rubric.
   - GSM8K by numeric extraction and exact match.
   - RealToxicityPrompts with moderation API.
5. Compute summary statistics and pairwise tests.
6. Perform qualitative/error analysis on representative cases.

### Baselines
- `direct`: Standard concise answer/continuation prompt.
- `pause_control`: Require fixed non-reasoning pause tokens between sentences (token-budget control).
- `think_between_sentences`: Require explicit `[THINK]...[/THINK]` block between every sentence.

### Evaluation Metrics
- TruthfulQA:
  - `truthfulness_score` (0/1 by rubric).
  - `carefulness_score` (1-5 judge rubric).
- GSM8K:
  - Exact-match accuracy on numeric answer.
- RealToxicityPrompts:
  - Moderation `flagged_rate`.
  - Mean max category score and category-wise means.
- Qualitative text features:
  - Words/tokens, sentence count, hedge marker rate, self-correction marker rate.

### Statistical Analysis Plan
- Primary pairwise comparisons: `think_between_sentences` vs `direct`; `think_between_sentences` vs `pause_control`.
- Binary outcomes: McNemar test for paired outputs where applicable; bootstrap CI for mean difference.
- Continuous outcomes: Paired t-test if normality is plausible, else Wilcoxon signed-rank.
- Effect sizes: Cohen's d (paired) and Cliff's delta for robustness.
- Alpha: 0.05 with Holm correction for multiple pairwise tests.

## Expected Outcomes
- Support for hypothesis if `think_between_sentences` improves truthfulness and/or safety without hurting GSM8K, and shows clear qualitative differences relative to both controls.
- Refutation if effects are absent, negative, or fully matched by pause-token control.

## Timeline and Milestones
1. Planning complete (this document).
2. Environment + dependency verification and GPU check.
3. Implement experiment harness (`src/run_experiments.py`).
4. Run all three task experiments on fixed subsets.
5. Analyze and visualize (`src/analyze_results.py`).
6. Write `REPORT.md` and update `README.md`.
7. Validation rerun on a smaller reproducibility subset.

## Potential Challenges
- API rate limits/costs: mitigate with retries, caching, subset sizes.
- Judge model bias: mitigate with explicit rubric and transparent limitations.
- Prompt-format noncompliance: use regex checks and fallback parsing.
- Length confound: address with pause-token control and length-normalized analysis.

## Success Criteria
- All three experiments complete with reproducible code and saved raw outputs.
- Report includes statistical tests, confidence intervals, effect sizes, and qualitative examples.
- Clear conclusion on whether inter-sentence thinking tokens produce distinctive, useful changes beyond extra tokens.
