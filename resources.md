# Resources Catalog

## Summary
This document catalogs resources gathered for the project "An LLM That's Careful With Its Words," including papers, datasets, and code repositories.

## Papers
Total papers downloaded: 10

| Title | Authors | Year | File | Key Info |
|-------|---------|------|------|----------|
| Chain-of-Thought Prompting Elicits Reasoning in Large Language Models | Wei et al. | 2022 | papers/2201.11903_cot_prompting_elicits_reasoning.pdf | Foundational CoT prompting results |
| Large Language Models are Zero-Shot Reasoners | Kojima et al. | 2022 | papers/2205.11916_zero_shot_cot.pdf | Zero-shot "step by step" trigger baseline |
| Self-Consistency Improves Chain of Thought Reasoning in Language Models | Wang et al. | 2022 | papers/2203.11171_self_consistency_cot.pdf | Multi-path CoT decoding baseline |
| Tree of Thoughts: Deliberate Problem Solving with Large Language Models | Yao et al. | 2023 | papers/2305.10601_tree_of_thoughts.pdf | Deliberate search over thought units |
| Chain-of-Thought Hub | Fu et al. | 2023 | papers/2305.17306_cot_hub.pdf | Benchmark/task hub for reasoning |
| Deliberate then Generate | Li et al. | 2023 | papers/2305.19835_deliberate_then_generate.pdf | Two-stage planning then generation |
| Think Before You Speak | Zhou et al. | 2023 | papers/2311.07445_think_before_you_speak_inner_monologue.pdf | Inner monologue for communication quality |
| Quiet-STaR | Zelikman et al. | 2024 | papers/2403.09629_quiet_star_think_before_speaking.pdf | Learned thought tokens before output |
| Improve Mathematical Reasoning by Automated Process Supervision | (arXiv 2406.06592) | 2024 | papers/2406.06592_automated_process_supervision_reasoning.pdf | Process-level supervision |
| Do We Need to Verify Step by Step? | (arXiv 2502.10581) | 2025 | papers/2502.10581_verify_step_by_step_process_supervision.pdf | Counterpoint on step-level verification |

See `papers/README.md` for detailed descriptions.

## Datasets
Total datasets downloaded: 3

| Name | Source | Size | Task | Location | Notes |
|------|--------|------|------|----------|-------|
| TruthfulQA (generation) | HuggingFace | 817 | Truthfulness generation eval | datasets/truthful_qa/hf | Good for careful factual wording |
| GSM8K (main) | HuggingFace | 7,473 train / 1,319 test | Math reasoning | datasets/gsm8k/hf | Standard CoT benchmark |
| RealToxicityPrompts (subset) | HuggingFace | 20,000 subset of 99,442 | Safety/careful continuation | datasets/real_toxicity_prompts/hf_train20k | Subset saved for local efficiency |

See `datasets/README.md` for detailed descriptions and download instructions.

## Code Repositories
Total repositories cloned: 3

| Name | URL | Purpose | Location | Notes |
|------|-----|---------|----------|-------|
| chain-of-thought-hub | https://github.com/FranxYao/chain-of-thought-hub | Reasoning benchmark scripts and leaderboard context | code/chain-of-thought-hub/ | Useful for benchmark setup |
| tree-of-thought-llm | https://github.com/princeton-nlp/tree-of-thought-llm | Deliberate thought search baseline | code/tree-of-thought-llm/ | Official ToT implementation |
| self-refine | https://github.com/madaan/self-refine | Iterative self-feedback/refinement baseline | code/self-refine/ | Strong careful-generation comparison |

See `code/README.md` for detailed repository notes.

## Resource Gathering Notes

### Search Strategy
- Attempted paper-finder script first; service endpoint was unavailable/hanging.
- Performed manual arXiv API search across multiple targeted queries.
- Supplemented with Semantic Scholar API and Papers with Code search pages.
- Prioritized papers tightly aligned with explicit reasoning traces and deliberate generation.

### Selection Criteria
- Direct relevance to explicit intermediate reasoning/thought in LLM generation.
- Foundational impact (high citations) plus recent methodological developments.
- Practical utility for building baseline experiments and reproducible evaluation.

### Challenges Encountered
- Paper-finder localhost service unavailable in this environment.
- Semantic Scholar rate-limited some follow-up queries (HTTP 429).
- Several methods require external API keys for full runtime replication.

### Gaps and Workarounds
- No single benchmark directly measures "thinking token between every sentence" behavior.
- Workaround: combine TruthfulQA, GSM8K, and RealToxicityPrompts to evaluate factual care, reasoning quality, and safe wording.

## Recommendations for Experiment Design

1. Primary dataset(s): TruthfulQA + GSM8K + RealToxicityPrompts subset.
2. Baseline methods: direct generation, zero-shot CoT, few-shot CoT, self-consistency, ToT/self-refine variants.
3. Evaluation metrics: exact-match accuracy, truthfulness/factual error rate, toxicity/safety rate, length-normalized quality metrics.
4. Code to adapt/reuse: `tree-of-thought-llm` for deliberate inference scaffolding, `self-refine` for multi-pass edits, `chain-of-thought-hub` for benchmark conventions.

## Research Execution Log (2026-03-01)

### What Was Executed
- Implemented controlled experiment harness: `src/run_experiments.py`
- Implemented analysis/statistics pipeline: `src/analyze_results.py`
- Ran real API experiments with:
  - Generation model: `gpt-4.1`
  - Judge model: `gpt-4.1-mini`
  - Moderation model: `omni-moderation-latest`
- Conditions tested:
  - `direct`
  - `pause_control`
  - `think_between_sentences`
- Dataset sample sizes: 50 TruthfulQA + 50 GSM8K + 50 RTP per condition

### Generated Outputs
- `results/raw_outputs.jsonl` (450 rows)
- `results/flat_outputs.csv`
- `results/metrics_summary.json`
- `results/pairwise_tests.csv`
- `results/binary_tests.csv`
- `results/plots/*.png`
- `REPORT.md` and updated `README.md`

### High-Level Outcome
- Inter-sentence visible thought-token forcing did not improve truthfulness, reasoning accuracy, or safety in this run, and reduced judged carefulness on TruthfulQA relative to direct prompting.
