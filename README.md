# Max Post-Training Tests

A project for experimenting with post-training techniques on language models, including fine-tuning, RLHF, and preference optimization.

## Overview

This repository contains scripts and utilities for running post-training experiments. The entry point handles argument parsing and orchestration, while the training module contains the core training loop logic.

## Project Structure

```
.
├── README.md          # This file
├── main.py            # Entry point — run experiments from here
└── train.py           # Core training logic
```

## Setup

```bash
pip install -r requirements.txt
```

## Usage

Run a training experiment:

```bash
python main.py --model gpt2 --dataset data/train.jsonl --epochs 3 --output-dir ./output
```

### Arguments

| Argument | Default | Description |
|---|---|---|
| `--model` | `gpt2` | Base model name or path |
| `--dataset` | `data/train.jsonl` | Path to training data (JSONL) |
| `--epochs` | `3` | Number of training epochs |
| `--batch-size` | `8` | Per-device batch size |
| `--lr` | `5e-5` | Learning rate |
| `--output-dir` | `./output` | Directory to save checkpoints |

## Training Data Format

Each line in the JSONL file should be a JSON object with a `text` field (for causal LM fine-tuning) or `prompt`/`response` fields (for instruction tuning):

```json
{"text": "Your training example here."}
{"prompt": "Instruction", "response": "Expected response"}
```
