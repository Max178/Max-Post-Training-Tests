#!/usr/bin/env python3
"""
main.py — Entry point for post-training experiments.

Usage:
    python main.py --model gpt2 --dataset data/train.jsonl
"""

import argparse
import logging
import sys

from posttrain_transformer.train import TrainingConfig, train


def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def main() -> None:
    print('hi')

if __name__ == "__main__":
    main()
