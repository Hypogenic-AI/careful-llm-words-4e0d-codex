# 1. Executive Summary

**Research question:** Does forcing an LLM to insert explicit "thinking tokens" between every sentence make outputs more careful and qualitatively different?

**Key finding:** In this controlled experiment, enforced inter-sentence thinking tokens (`[THINK]...[/THINK]`) did **not** improve truthfulness, GSM8K accuracy, or safety outcomes versus direct prompting; they significantly reduced judged carefulness on TruthfulQA compared with direct prompting.

**Practical implication:** Adding mandatory visible thought tokens between every sentence is not a reliable way to make output safer or more accurate. If used, it should be treated as a style/control intervention, not a quality-improvement guarantee.

# 2. Goal

We tested the hypothesis that requiring explicit thought tokens between every sentence causes more careful generation.

Why this matters: if true, this would provide a cheap inference-time safety/quality control without retraining. The expected impact was improved factual caution, reasoning reliability, and safer continuation behavior.

# 3. Data Construction

## Dataset Description

- **TruthfulQA (generation)**
  - Source: HF `truthful_qa`, config `generation`
  - Local: `datasets/truthful_qa/hf`
  - Size used in experiment: 50 sampled items (from 817 validation)
  - Purpose: factual truthfulness and careful wording

- **GSM8K (main)**
  - Source: HF `gsm8k`, config `main`
  - Local: `datasets/gsm8k/hf`
  - Size used in experiment: 50 sampled items (from 1319 test)
  - Purpose: objective reasoning correctness (numeric exact match)

- **RealToxicityPrompts subset**
  - Source: HF `allenai/real-toxicity-prompts`
  - Local: `datasets/real_toxicity_prompts/hf_train20k`
  - Size used in experiment: 50 sampled items (from 20,000 subset)
  - Purpose: safety/toxicity risk under continuation

## Example Samples

| Dataset | Example Prompt |
|---|---|
| TruthfulQA | "What happens to you if you eat watermelon seeds?" |
| GSM8K | "Janet’s ducks lay 16 eggs per day... How much in dollars does she make..." |
| RTP | "Corruption involving the contractors is the chief culprit... according to a recent" |

## Data Quality

Checks performed on full local datasets before sampling:
- Missing values (key fields): **0.0%** for all three datasets
- Duplicate keys:
  - TruthfulQA questions: **0** duplicates
  - GSM8K questions: **0** duplicates
  - RTP prompt text: **47** duplicates out of 20,000
- RTP prompt toxicity distribution (subset metadata):
  - Mean: **0.0866**
  - 95th percentile: **0.2236**
  - Max: **0.6519**

## Preprocessing Steps

1. Loaded datasets from local disk using `datasets.load_from_disk`.
2. Deterministic sampling with seed 42 (`50` per dataset).
3. Built three condition-specific prompt templates.
4. Normalized outputs by stripping tags (`[THINK]`, `[PAUSE]`) for downstream textual features.
5. For GSM8K, parsed `FINAL_ANSWER` and normalized numeric equality (`2` equals `2.00`).

## Train/Val/Test Splits

No model training occurred. We ran inference-only evaluation on fixed sampled subsets from benchmark test/validation splits.

# 4. Experiment Description

## Methodology

### High-Level Approach

We compared three prompting conditions on identical item sets:
1. `direct`
2. `pause_control` (forced non-reasoning token block between sentences)
3. `think_between_sentences` (forced `[THINK]` block between sentences)

This isolates whether effects come from deliberate thought content vs simply inserting extra tokens.

### Why This Method?

- Directly tests user hypothesis: mandatory inter-sentence thought tokens.
- Includes confound control (`pause_control`) for extra-token overhead.
- Uses real modern API models and real benchmark datasets.

## Implementation Details

### Tools and Libraries

- Python 3.12.8
- datasets 4.6.1
- openai 2.24.0
- numpy 2.3.5
- pandas 2.3.3
- scipy 1.17.1
- statsmodels 0.14.6
- matplotlib 3.10.8
- seaborn 0.13.2

### Models

- Generation model: `gpt-4.1`
- Judge model (TruthfulQA rubric scoring): `gpt-4.1-mini`
- Safety model: `omni-moderation-latest`

### Hyperparameters

| Parameter | Value | Selection Method |
|---|---:|---|
| temperature | 0.2 | fixed for controlled comparisons |
| max_output_tokens | 320 | enough for 3-sentence format + tags |
| seed | 42 | reproducibility |
| n per dataset | 50 | balanced quick-but-statistical run |

### Analysis Pipeline

1. Run generation under each condition for each task.
2. Score:
   - TruthfulQA: LLM judge -> `truthful` (0/1), `carefulness` (1-5)
   - GSM8K: numeric exact match
   - RTP: moderation flagged + max category score
3. Compute descriptive stats and paired tests.
4. Apply Holm correction within each test family.
5. Generate plots and CSV/JSON outputs.

## Experimental Protocol

### Reproducibility Information

- Runs: one full deterministic run (temperature fixed at 0.2)
- Seed: 42
- Hardware detected: 2x NVIDIA RTX 3090 (24GB each)
- GPU usage: not required (API-based experiment)
- Execution time: ~13.84 minutes wall-clock for 450 generations + scoring

### Prompt Compliance

- Think condition with >=2 `[THINK]` blocks: **50/50** for each task
- Pause condition with >=2 `[PAUSE]` blocks: **50/50** for each task
- Direct condition tag leakage: **0/50** for each task

### Token Usage

- Main generation tokens: input **41,706**, output **36,774**
- Judge tokens: input **23,858**, output **11,336**
- Total: input **65,564**, output **48,110**

## Raw Results

### Tables

**TruthfulQA**

| Condition | Truthful rate | Carefulness mean |
|---|---:|---:|
| direct | 0.98 | 4.68 |
| pause_control | 0.92 | 4.48 |
| think_between_sentences | 0.94 | 4.28 |

**GSM8K**

| Condition | Exact match |
|---|---:|
| direct | 0.94 |
| pause_control | 0.90 |
| think_between_sentences | 0.94 |

**RealToxicityPrompts**

| Condition | Flagged rate | Mean max category score |
|---|---:|---:|
| direct | 0.02 | 0.0151 |
| pause_control | 0.02 | 0.0172 |
| think_between_sentences | 0.06 | 0.0241 |

### Statistical Comparisons (primary)

- **TruthfulQA carefulness (`think - direct`)**:
  - mean diff = **-0.40**, 95% CI **[-0.56, -0.24]**
  - Wilcoxon p = **4.46e-05**, Holm-adjusted p = **1.34e-04**
  - Cohen's d (paired) = **-0.70**

- **TruthfulQA truthful binary (`think vs direct`)**:
  - McNemar p = **0.50** (not significant)

- **GSM8K exact match (`think - direct`)**:
  - mean diff = **0.00**, 95% CI **[0.00, 0.00]**
  - constant-zero paired difference, p = **1.00**

- **RTP max category score (`think - direct`)**:
  - mean diff = **+0.0090**, 95% CI **[-0.0057, 0.0288]**
  - Wilcoxon p = **0.461**, Holm p = **1.00**

- **RTP flagged binary (`think vs direct`)**:
  - McNemar p = **0.50** (not significant)

### Visualizations

Generated under `results/plots/`:
- `truthfulqa_truthful.png`
- `truthfulqa_carefulness.png`
- `gsm8k_exact_match.png`
- `rtp_flagged.png`
- `rtp_max_category_score.png`
- `truthfulqa_hedge_count.png`

### Output Locations

- Raw outputs: `results/raw_outputs.jsonl`
- Flattened outputs: `results/flat_outputs.csv`
- Summary metrics: `results/metrics_summary.json`
- Pairwise tests: `results/pairwise_tests.csv`
- Binary tests: `results/binary_tests.csv`
- Plots: `results/plots/`
- Scripts: `src/run_experiments.py`, `src/analyze_results.py`

# 5. Result Analysis

## Key Findings

1. Forcing `[THINK]` blocks between every sentence did **not** improve TruthfulQA truthfulness versus direct prompting.
2. It significantly **decreased judged carefulness** relative to direct prompting.
3. GSM8K exact-match performance for think and direct was identical in this sample.
4. Safety metrics on RTP did not improve with think tokens; flagged rate was numerically higher in think condition (not significant).

## Hypothesis Testing Results

- H1 (truthfulness improvement): **Not supported**.
- H2 (reasoning improvement): **Not supported**.
- H3 (safety improvement): **Not supported**.
- H4 (qualitative difference): **Partially supported** (style changed, but not beneficial on core metrics).

## Comparison to Baselines

- `think_between_sentences` vs `direct`:
  - TruthfulQA truthful: -4 percentage points (not significant)
  - TruthfulQA carefulness: -0.40 points (significant)
  - GSM8K: 0-point difference
  - RTP flagged: +4 percentage points (not significant)

- `think_between_sentences` vs `pause_control`:
  - Generally small/non-significant differences except lower carefulness score.

Interpretation: the inter-sentence thought-token requirement does not outperform controls and may degrade judged answer quality in factual QA style.

## Surprises and Insights

- Expected gain from “forced deliberation” did not materialize.
- The token-budget control and think condition were often similar, suggesting token overhead/style constraints dominate over explicit mini-reasoning traces.
- In some TruthfulQA items, think-format outputs became shorter or dropped useful contextual detail.

## Error Analysis

Representative failure patterns:
- **Compression/oversimplification:** think outputs occasionally omitted caveats present in direct outputs.
- **Format-induced awkwardness:** strict sentence+tag structure sometimes reduced natural explanatory flow.
- **Safety false positives:** some flagged RTP outputs were neutral but quoted high-risk language from prompt context.

## Limitations

- Single generation model family (`gpt-4.1`) and one run per item.
- Truthfulness evaluation uses LLM-as-judge (rubric helps but can introduce evaluator bias).
- Sample size is moderate (50/task), not full-benchmark scale.
- Forced 3-sentence structure may itself influence quality independent of thought content.
- This study tests visible thought blocks; hidden internal reasoning behavior is not directly measured.

# 6. Conclusions

Mandatory visible inter-sentence thinking tokens did not make this LLM materially more accurate, truthful, or safe in our tests. The strongest measurable effect was a reduction in judged carefulness on TruthfulQA versus direct prompting. In practice, this mechanism appears to alter style more than reliability.

Implication: teams seeking safer or more factual outputs should prefer verified methods (retrieval grounding, critique-and-revise loops, external verification) over naive sentence-by-sentence thought-token insertion.

Confidence: **moderate** for this prompting setup and model, lower for broader generalization across architectures and larger samples.

# 7. Next Steps

## Immediate Follow-ups

1. Test hidden two-pass drafting (private deliberation then final answer) against visible-tag formats.
2. Increase sample size (e.g., 200/task) and repeat with additional models (e.g., GPT-5, Claude Sonnet 4.5).
3. Add human evaluation for readability/trust/calibrated uncertainty.

## Alternative Approaches

- Self-refinement loops with explicit verification checkpoints.
- Retrieval-backed factual grounding before each sentence.
- Process supervision or verifier models for claim-level checking.

## Broader Extensions

- Evaluate in high-stakes domains (medical/legal) with domain-specific factual checkers.
- Study latency/cost-quality trade-offs for production deployment.

## Open Questions

- Do hidden deliberation channels outperform visible thought tags consistently?
- Can better-designed inter-sentence verification prompts avoid quality degradation?
- Which tasks benefit from sentence-level planning versus document-level planning?

# References

- Wei et al. (2022). Chain-of-Thought Prompting Elicits Reasoning in Large Language Models.
- Kojima et al. (2022). Large Language Models are Zero-Shot Reasoners.
- Wang et al. (2022). Self-Consistency Improves Chain of Thought Reasoning in Language Models.
- Yao et al. (2023). Tree of Thoughts: Deliberate Problem Solving with Large Language Models.
- Fu et al. (2023). Chain-of-Thought Hub.
- Li et al. (2023). Deliberate then Generate.
- Zhou et al. (2023). Think Before You Speak.
- Zelikman et al. (2024). Quiet-STaR.
- Automated process supervision and step-verification analyses (2024-2025), as listed in `papers/`.
