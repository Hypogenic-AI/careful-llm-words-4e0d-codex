# Downloaded Datasets

This directory contains datasets for experiments on whether explicit intermediate reasoning/thinking changes output quality and behavior. Data files are not committed to git due to size.

## Dataset 1: TruthfulQA (generation)

### Overview
- Source: HuggingFace `truthful_qa` (config: `generation`)
- Size: 817 validation questions
- Format: HuggingFace dataset
- Task: Truthfulness / factual robustness in generation
- Splits: validation (817)
- License: See dataset card on HuggingFace
- Local location: `datasets/truthful_qa/hf`

### Download Instructions

Using HuggingFace (recommended):
```python
from datasets import load_dataset
truth = load_dataset("truthful_qa", "generation")
truth.save_to_disk("datasets/truthful_qa/hf")
```

### Loading the Dataset
```python
from datasets import load_from_disk
dataset = load_from_disk("datasets/truthful_qa/hf")
```

### Sample Data
- `datasets/truthful_qa/samples/samples.json`

### Notes
- Useful for checking whether forced reasoning traces improve truthful responses or just verbosity.

## Dataset 2: GSM8K (main)

### Overview
- Source: HuggingFace `gsm8k` (config: `main`)
- Size: 7,473 train / 1,319 test
- Format: HuggingFace dataset
- Task: Multi-step mathematical reasoning
- Splits: train, test
- License: See dataset card on HuggingFace
- Local location: `datasets/gsm8k/hf`

### Download Instructions

Using HuggingFace (recommended):
```python
from datasets import load_dataset
gsm = load_dataset("gsm8k", "main")
gsm.save_to_disk("datasets/gsm8k/hf")
```

### Loading the Dataset
```python
from datasets import load_from_disk
dataset = load_from_disk("datasets/gsm8k/hf")
```

### Sample Data
- `datasets/gsm8k/samples/samples.json`

### Notes
- Good for measuring reasoning accuracy differences between normal decoding and explicit per-sentence thinking-token prompting/constraints.

## Dataset 3: RealToxicityPrompts (subset)

### Overview
- Source: HuggingFace `allenai/real-toxicity-prompts`
- Size: 99,442 total train examples (downloaded subset: 20,000)
- Format: HuggingFace dataset
- Task: Safety and carefulness under prompt continuation
- Splits: train (subset saved)
- License: See dataset card on HuggingFace
- Local location: `datasets/real_toxicity_prompts/hf_train20k`

### Download Instructions

Using HuggingFace (recommended):
```python
from datasets import load_dataset
rtp = load_dataset("allenai/real-toxicity-prompts")
subset = rtp["train"].select(range(20000))
subset.save_to_disk("datasets/real_toxicity_prompts/hf_train20k")
```

### Loading the Dataset
```python
from datasets import load_from_disk
dataset = load_from_disk("datasets/real_toxicity_prompts/hf_train20k")
```

### Sample Data
- `datasets/real_toxicity_prompts/samples/samples.json`

### Notes
- Subset used to keep local footprint manageable while enabling quick ablation runs.
- Full dataset size is recorded in `datasets/dataset_summary.json`.
