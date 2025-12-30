# SOFT

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![arXiv](https://img.shields.io/badge/arXiv-2506.10424-b31b1b.svg)](https://arxiv.org/pdf/2506.10424)

This is the implementation of
[SOFT: Selective Data Obfuscation for Protecting LLM Fine-tuning against Membership Inference Attacks](https://arxiv.org/pdf/2506.10424) (USENIX Security'25).


## Table of Contents

- [Code Structure](#code-structure)
- [Quick Start](#quick-start)
- [Dataset Information](#dataset-information)
- [Data Obfuscation](#data-obfuscation)
- [Evaluation](#evaluation)
- [Citation](#citation)
- [Acknowledgments](#acknowledgments)

## Code Structure

```
mia_llms_benchmark/
├── README.md                       # This file
├── environment.yml                 # Conda environment specification
├── config_finetune.yaml            # Training configuration
├── config_auc_tpr.yaml             # Evaluation configuration
├── finetune.py                     # Main fine-tuning script
├── main.py                         # Evaluation script
├── utils.py                        # Utility functions
├── data/
│   ├── obfuscation.py              # Obfuscation implementations
│   └── prepare.py                  # Dataset loading and tokenization
├── attacks/                        # MIA attack implementations
│   ├── __init__.py                 
│   ├── loss.py                     
│   ├── ratio.py                    
│   ├── mink.py                     
│   ├── minkplusplus.py             
│   ├── zlib.py                     
│   ├── lowercase.py                
│   ├── recall.py                   
│   ├── conrecall.py                
│   ├── bag_of_words.py             
│   ├── ensemble_classifier.py      
│   └── utils.py                    # Attack utilities
└── output/                         # Evaluation results
```

## Quick Start

### 1. Install Dependencies
```bash
# Create python environment
conda env create -f environment.yml
conda activate mia
```

### 2. Fine-tune Model with Defense
```bash
# Single GPU training
python finetune.py --config config_finetune.yaml --select_ratio X

# Multi-GPU training with DeepSpeed
deepspeed --num_gpus=8 finetune.py --config config_finetune.yaml --select_ratio X
```

### 3. Evaluate Privacy Protection
The metrics include AUC-ROC, TPR@0.1FPR, and TPR@0.01FPR.
```bash
python main.py \
    -c config_auc_tpr.yaml \
    --run-all \
    --output "./output/" \
    --target-model "checkpoints/Llama-3.2-X/epoch-X" \
    --dataset "arxiv" \
    --split "ngram_13_0.8"
```

## Dataset Information

### Original Dataset
- **Source**: [iamgroot42/mimir](https://huggingface.co/datasets/iamgroot42/mimir)
- **Description**: Curated subset of The Pile dataset with membership labels
- **Splits**: Various n-gram and threshold combinations (e.g., `ngram_13_0.8`)
- **Domains**: ArXiv papers, Wikipedia, GitHub code, PubMed, and more

### Example of Obfuscated Dataset
- **Source**: [LLM-MIA/editing-syn-pr0.5-mimir-arxiv-ngram_13_0.8](https://huggingface.co/datasets/LLM-MIA/editing-syn-pr0.5-mimir-arxiv-ngram_13_0.8)
- **Description**: Paraphrased version of the ArXiv subset using advanced text transformation
- **Usage**: Ready-to-use obfuscated data for immediate training

## Data Obfuscation

### Generate Your Own Obfuscated Data

The `data/obfuscation.py` module provides tools to create obfuscated datasets:

```bash
# Set up environment variables
export OPENAI_API_KEY="your-api-key"
export HF_TOKEN="your-huggingface-token"

# Using OpenAI API for paraphrasing
python data/obfuscation.py
```

### Obfuscation Prompts

The framework supports different prompts for various content types:

**Text Paraphrasing Prompt:**
```python
message = [
    {"role": "system", "content": "You are a helpful text rewriting assistant."},
    {"role": "user", "content":
     f"Rewrite the following paragraph by replacing every word with an alternative term that does not share the same root or spelling. Preserve the same meaning and sentence structure as much as possible.\n\"\"\"\n{original_text}\n\"\"\""},
]
```

**Code Obfuscation Prompt:**
```python
message = f"Rewrite the following code so it preserves the same functionality and flow, but changes all variable names, function names, and comments. Maintain the same input-output behavior. Keep it in the same programming language.\n\"\"\"\n{original_text}\n\"\"\""
```

## Evaluation

### Available Attack Methods

The framework implements 10+ state-of-the-art MIA attacks:

| Attack Method | Description | Key Parameters |
|---------------|-------------|----------------|
| **Loss** | Basic loss-based attack | - |
| **Zlib** | Compression-based attack | - |
| **Lowercase** | Case-sensitivity attack | - |
| **Min-K% Prob** | Minimum k-probability attack | `k` |
| **Min-K%++** | Enhanced MinK with calibration | `k` |
| **Ratio** | Loss ratio with reference model | `reference_model_path` |
| **Bag of Words** | Feature-based ML attack | - |
| **ReCall** | Prefix-based recall attack | `n_shots`, `extra_non_member_dataset` |
| **CON-ReCall** | Conditional recall attack | `n_shots`, `extra_non_member_dataset` |
| **Ensemble** | Combined multiple attacks | - |

##### Custom Evaluation

```bash
# Evaluate specific attacks only
python main.py \
    -c config_auc_tpr.yaml \
    --attacks "loss,ratio,mink" \
    --target-model "path/to/model" \
    --dataset "arxiv" \
    --split "ngram_13_0.8"
```

## Citation

If you use this framework in your research, please cite:

```bibtex
@inproceedings{zhang2025soft,
    title = {SOFT: Selective Data Obfuscation for Protecting LLM Fine-tuning against Membership Inference Attacks},
    author = {Zhang, Kaiyuan and Cheng, Siyuan and Guo, Hanxi and Chen, Yuetian and Su, Zian and An, Shengwei and Du, Yuntao and Fleming, Charles and Kundu, Ashish and Zhang, Xiangyu and Li, Ninghui},
    booktitle = {34th USENIX Security Symposium (USENIX Security 25)},
    year = {2025},
    address = {Seattle, WA},
    publisher = {USENIX Association}
}
```


## Acknowledgments

- [Mimir Dataset](https://huggingface.co/datasets/iamgroot42/mimir) for providing the evaluation benchmark
- [The Pile](https://pile.eleuther.ai/) for the underlying text corpus
- HuggingFace for the model and dataset hosting infrastructure
