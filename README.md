# An LLM That's Careful With Its Words

Controlled API study testing whether forcing explicit thinking tokens between every sentence changes LLM behavior. We compared three prompting conditions (`direct`, `pause_control`, `think_between_sentences`) on TruthfulQA, GSM8K, and RealToxicityPrompts.

## Key Findings

- Forced inter-sentence `[THINK]` tokens did **not** improve TruthfulQA truthfulness (0.94 vs 0.98 direct).
- It **reduced** judged carefulness on TruthfulQA (4.28 vs 4.68 direct, significant).
- GSM8K exact match was unchanged for think vs direct (both 0.94).
- RealToxicityPrompts safety did not improve; flagged rate was numerically higher for think (0.06 vs 0.02 direct, not significant).
- Conclusion: mandatory visible thought tokens changed style more than reliability.

See full details in [REPORT.md](REPORT.md).

## Reproduce

```bash
# from workspace root
source .venv/bin/activate

# run experiments (real API calls)
python src/run_experiments.py --n-truthfulqa 50 --n-gsm8k 50 --n-rtp 50 --temperature 0.2 --max-output-tokens 320

# analyze and generate plots/tables
python src/analyze_results.py
```

## File Structure

- `planning.md`: research plan and motivation/novelty assessment
- `src/run_experiments.py`: experiment harness with API calls
- `src/analyze_results.py`: statistics and plots
- `results/raw_outputs.jsonl`: raw model outputs and per-item metadata
- `results/metrics_summary.json`: aggregate metrics + tests
- `results/pairwise_tests.csv`, `results/binary_tests.csv`: statistical tests
- `results/plots/`: figures used in report
- `REPORT.md`: full scientific report
