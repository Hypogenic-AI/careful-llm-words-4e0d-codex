# Literature Review: An LLM That's Careful With Its Words

## Review Scope

### Research Question
Does forcing explicit intermediate reasoning/thinking tokens between generated sentences change output quality and behavior (carefulness, truthfulness, reasoning fidelity, and safety)?

### Inclusion Criteria
- Studies on chain-of-thought, deliberate generation, inner monologue, or process supervision in LLMs
- Empirical results on text quality, reasoning, or safety-style outcomes
- Prefer papers with public PDF and reproducible setup details

### Exclusion Criteria
- Unrelated multimodal-only methods without textual reasoning relevance
- Papers with no methodological details or no usable evaluation setup

### Time Frame
- Primarily 2022 to 2025 (plus latest available records encountered during search)

### Sources
- arXiv API search (primary)
- Semantic Scholar API search (supplementary)
- Papers with Code search pages (supplementary)

## Search Log

| Date | Query | Source | Results | Notes |
|------|-------|--------|---------|-------|
| 2026-03-01 | "LLM explicit chain-of-thought between every sentence careful wording" | local paper-finder script | unavailable | localhost service hung; switched to manual search |
| 2026-03-01 | chain-of-thought / deliberate / process supervision / think-before-speaking queries | arXiv API | 65 candidates | selected 10 directly relevant papers |
| 2026-03-01 | chain of thought prompting language models | Semantic Scholar API | 5 returned | includes foundational CoT and Self-Consistency |
| 2026-03-01 | chain of thought, tree of thoughts, truthfulqa | Papers with Code | pages reachable | used for benchmark/code linkage |

## Screening Results

| Paper | Title Screen | Abstract Screen | Full-Text | Notes |
|-------|-------------|-----------------|-----------|-------|
| 2201.11903 | Include | Include | Include (deep) | Foundational CoT prompting |
| 2205.11916 | Include | Include | Include | Zero-shot CoT baseline |
| 2203.11171 | Include | Include | Include | Decoding strategy for CoT robustness |
| 2305.10601 | Include | Include | Include | Deliberate search over thoughts |
| 2305.17306 | Include | Include | Include | Evaluation benchmark hub |
| 2305.19835 | Include | Include | Include (deep) | Deliberate-then-generate framework |
| 2311.07445 | Include | Include | Include (deep) | Inner monologue before response |
| 2403.09629 | Include | Include | Include (deep) | Learned latent thought tokens before speaking |
| 2406.06592 | Include | Include | Include | Automated process supervision |
| 2502.10581 | Include | Include | Include | Analysis of step-by-step supervision assumptions |

## Key Papers

### Paper 1: Chain-of-Thought Prompting Elicits Reasoning in Large Language Models
- Authors: Wei et al.
- Year: 2022
- Source: arXiv (2201.11903)
- Key Contribution: Demonstrates large gains on multi-step reasoning from CoT exemplars.
- Methodology: Few-shot CoT prompts with intermediate reasoning traces.
- Datasets Used: GSM8K, StrategyQA, commonsense/math benchmarks.
- Results: Strong accuracy gains at larger model scales.
- Code Available: Partial benchmark assets; many downstream re-implementations.
- Relevance: Direct baseline for any explicit between-sentence thought requirement.

### Paper 2: Large Language Models are Zero-Shot Reasoners
- Authors: Kojima et al.
- Year: 2022
- Source: arXiv (2205.11916)
- Key Contribution: "Let's think step by step" unlocks zero-shot reasoning improvements.
- Methodology: Zero-shot prompting trigger for latent reasoning chain.
- Datasets Used: Arithmetic and symbolic reasoning tasks.
- Results: Significant jump over vanilla zero-shot prompting.
- Code Available: Community implementations exist.
- Relevance: Minimal intervention baseline for prompting explicit reasoning text.

### Paper 3: Self-Consistency Improves Chain of Thought Reasoning in Language Models
- Authors: Wang et al.
- Year: 2022
- Source: arXiv (2203.11171)
- Key Contribution: Improves CoT reliability via diverse reasoning samples + answer aggregation.
- Methodology: Multi-sample CoT decoding with consistency voting.
- Datasets Used: GSM8K and other reasoning benchmarks.
- Results: Better robustness than single-path CoT.
- Code Available: Widely reproduced.
- Relevance: Strong comparison point for "careful wording" through inference-time compute.

### Paper 4: Tree of Thoughts: Deliberate Problem Solving with Large Language Models
- Authors: Yao et al.
- Year: 2023
- Source: arXiv (2305.10601)
- Key Contribution: Search-based reasoning over thought units instead of single left-to-right chain.
- Methodology: Generate/evaluate/select thoughts (BFS/DFS style).
- Datasets Used: Game of 24, creative writing, crosswords.
- Results: Improves success on tasks requiring planning.
- Code Available: Yes (official repo cloned).
- Relevance: Closely aligned with enforced intermediate thought constraints.

### Paper 5: Chain-of-Thought Hub
- Authors: Fu et al.
- Year: 2023
- Source: arXiv (2305.17306)
- Key Contribution: Aggregated reasoning benchmark hub and model comparison.
- Methodology: Evaluation suite across GSM8K/MATH/MMLU/BBH.
- Datasets Used: Major reasoning benchmarks.
- Results: Highlights persistent gap between frontier and smaller/open models.
- Code Available: Yes (repo cloned).
- Relevance: Provides practical benchmark set for this project.

### Paper 6: Deliberate then Generate
- Authors: Li et al.
- Year: 2023
- Source: arXiv (2305.19835)
- Key Contribution: Separate deliberate planning from final generation for text tasks.
- Methodology: Two-stage prompting framework.
- Datasets Used: MT, summarization, style transfer tasks.
- Results: Better coherence/faithfulness in several settings.
- Code Available: No official unified release identified in this pass.
- Relevance: Very close to "think before each sentence" framing.

### Paper 7: Think Before You Speak
- Authors: Zhou et al.
- Year: 2023
- Source: arXiv (2311.07445)
- Key Contribution: Uses inner monologue to improve communication behavior.
- Methodology: Prompted inner-monologue scaffolding for dialogue skills.
- Datasets Used: Conversation skill benchmarks/self-chat and human-bot evaluations.
- Results: Improvements in humanness/proactivity/engagingness-style metrics.
- Code Available: Prompt design details in appendices.
- Relevance: Directly targets "consider before speaking" behavior.

### Paper 8: Quiet-STaR
- Authors: Zelikman et al.
- Year: 2024
- Source: arXiv (2403.09629)
- Key Contribution: Trains model to generate internal thought tokens that improve next-token prediction.
- Methodology: Learned thought spans with specialized thought start/end tokens.
- Datasets Used: OpenWebMath and text modeling setups.
- Results: Better reasoning-like performance and improved predictive behavior.
- Code Available: Public paper; check upstream project releases for code updates.
- Relevance: Strong methodological analog to explicit intermediate thinking tokens.

### Paper 9: Improve Mathematical Reasoning in Language Models by Automated Process Supervision
- Authors: (arXiv 2406.06592)
- Year: 2024
- Source: arXiv
- Key Contribution: Automated process-level supervision for intermediate reasoning quality.
- Methodology: Process labeling/reward and reasoning optimization.
- Datasets Used: Math reasoning datasets.
- Results: Better step quality and final answers.
- Code Available: Partial artifacts.
- Relevance: Useful for judging whether explicit steps are actually useful or merely verbose.

### Paper 10: Do We Need to Verify Step by Step? Rethinking Process Supervision
- Authors: (arXiv 2502.10581)
- Year: 2025
- Source: arXiv
- Key Contribution: Re-examines assumptions behind mandatory step-level verification.
- Methodology: Theoretical and empirical analysis of process supervision regimes.
- Datasets Used: Math/reasoning settings.
- Results: Shows contexts where full stepwise verification is unnecessary or suboptimal.
- Code Available: Not confirmed in this pass.
- Relevance: Important counterpoint to the hypothesis.

## Deep Reading Notes (Chunked PDFs)
- Full chunk-by-chunk notes were recorded for four key papers:
  - `papers/pages/2201.11903/*`
  - `papers/pages/2305.19835/*`
  - `papers/pages/2311.07445/*`
  - `papers/pages/2403.09629/*`
- Consolidated notes file: `papers/deep_reading_chunk_notes.md`

## Common Methodologies
- Prompted explicit reasoning traces: used in CoT / Zero-shot CoT / Think-before-you-speak.
- Deliberate search/planning: used in Tree-of-Thought and deliberate-then-generate paradigms.
- Process supervision: used for improving intermediate reasoning correctness.
- Decoding-time robustness: self-consistency and voting over multiple rationale paths.

## Standard Baselines
- Direct answer prompting (no CoT)
- Few-shot CoT prompting
- Zero-shot CoT trigger prompting
- Self-consistency decoding
- Deliberate reasoning frameworks (ToT)

## Evaluation Metrics
- Task accuracy (GSM8K exact match / benchmark accuracy)
- Truthfulness metrics (TruthfulQA scoring protocol)
- Toxicity/safety behavior rates (RealToxicityPrompts continuation metrics)
- Qualitative communication quality (human evaluation dimensions where applicable)

## Datasets in the Literature
- GSM8K: widely used for multi-step reasoning accuracy.
- MATH/MMLU/BBH: common in reasoning benchmark hubs.
- TruthfulQA: useful for factual robustness/carefulness checks.
- RealToxicityPrompts: useful for safety and wording caution behaviors.

## Gaps and Opportunities
- Gap 1: Limited direct studies on forcing thought tokens between *every sentence* in open-ended generation.
- Gap 2: Many works optimize final accuracy, not wording-level caution/fidelity tradeoffs.
- Gap 3: Process transparency may conflict with faithfulness (reasoning text can be post-hoc).

## Recommendations for Our Experiment
- Recommended datasets:
  - TruthfulQA (carefulness/truthfulness)
  - GSM8K (reasoning correctness)
  - RealToxicityPrompts (safe wording under continuation)
- Recommended baselines:
  - Direct generation
  - Zero-shot CoT
  - Few-shot CoT
  - Self-consistency
  - Tree-of-Thought (where task-compatible)
- Recommended metrics:
  - Exact match/accuracy on reasoning tasks
  - Truthfulness score / false claim rate
  - Toxicity rate and refusal/hedging behavior analysis
  - Length-normalized quality measures to control verbosity inflation
- Methodological considerations:
  - Control for output length and token budget (forced thoughts may improve by spending more tokens).
  - Separate private-thought vs exposed-thought conditions.
  - Evaluate sentence-level factual precision, not only final-task correctness.
