# Cloned Repositories

## Repo 1: chain-of-thought-hub
- URL: https://github.com/FranxYao/chain-of-thought-hub
- Purpose: Benchmark and scripts for evaluating reasoning-heavy tasks with CoT prompting.
- Location: `code/chain-of-thought-hub/`
- Key files:
  - `code/chain-of-thought-hub/readme.md`
  - `code/chain-of-thought-hub/gsm8k/`
  - `code/chain-of-thought-hub/MMLU/`
  - `code/chain-of-thought-hub/BBH/`
- Notes:
  - Provides task organization and scripts for major reasoning benchmarks (GSM8K, MMLU, BBH, etc.).
  - Useful as evaluation reference and baseline task setup.

## Repo 2: tree-of-thought-llm
- URL: https://github.com/princeton-nlp/tree-of-thought-llm
- Purpose: Official Tree-of-Thoughts implementation for deliberate multi-step search over candidate thoughts.
- Location: `code/tree-of-thought-llm/`
- Key files:
  - `code/tree-of-thought-llm/run.py`
  - `code/tree-of-thought-llm/scripts/`
  - `code/tree-of-thought-llm/src/tot/`
  - `code/tree-of-thought-llm/requirements.txt`
- Installation requirements discovered:
  - Python package + OpenAI API key (`OPENAI_API_KEY`)
  - Dependencies in `requirements.txt` (includes `openai`, `sympy`, `numpy`, `pandas`, etc.)
- Notes:
  - Strong baseline for explicit deliberation versus direct generation.
  - Relevant for testing whether structured intermediate thought improves sentence-level output quality.

## Repo 3: self-refine
- URL: https://github.com/madaan/self-refine
- Purpose: Iterative self-feedback and refinement framework for improving generated outputs.
- Location: `code/self-refine/`
- Key files:
  - `code/self-refine/README.md`
  - `code/self-refine/src/`
  - `code/self-refine/data/`
- Installation requirements discovered:
  - Uses `prompt-lib` plus task-specific dependencies described in README.
  - Requires model API access depending on selected task setup.
- Notes:
  - Good baseline for multi-pass careful generation and self-critique loops.
  - Can be adapted to compare explicit per-sentence thinking-token constraints against post-hoc refinement.

## Quick Validation Summary
- Cloning succeeded for all three repositories.
- README inspection completed for each repository.
- Full execution was not run end-to-end because API keys/model endpoints are required for most meaningful runs.
