#!/usr/bin/env python3
"""
main.py — Entry point for post-training experiments.

Usage:
    python main.py --model gpt2 --dataset data/train.jsonl
"""

import argparse
import logging
import sys

from train import TrainingConfig, train


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a post-training experiment.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", default="gpt2", help="Base model name or path")
    parser.add_argument("--dataset", default="data/train.jsonl", help="Path to training data (JSONL)")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Per-device batch size")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--max-length", type=int, default=512, help="Max token length per example")
    parser.add_argument("--warmup-steps", type=int, default=100, help="LR warmup steps")
    parser.add_argument("--save-steps", type=int, default=500, help="Checkpoint save frequency")
    parser.add_argument("--output-dir", default="./output", help="Directory to save checkpoints")
    parser.add_argument("--resume-from", default=None, help="Resume training from a checkpoint path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args()


def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)

    config = TrainingConfig(
        model_name=args.model,
        dataset_path=args.dataset,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        max_length=args.max_length,
        warmup_steps=args.warmup_steps,
        save_steps=args.save_steps,
        seed=args.seed,
    )

    train(config, resume_from=args.resume_from)


if __name__ == "__main__":
    main()
