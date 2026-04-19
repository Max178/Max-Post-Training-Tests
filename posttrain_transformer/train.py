"""
train.py — Core training logic for post-training experiments.
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    model_name: str
    dataset_path: str
    output_dir: str
    epochs: int = 3
    batch_size: int = 8
    learning_rate: float = 5e-5
    max_length: int = 512
    warmup_steps: int = 100
    save_steps: int = 500
    seed: int = 42


def load_dataset(path: str) -> list[dict]:
    """Load a JSONL dataset from disk."""
    data = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    logger.info(f"Loaded {len(data)} examples from {path}")
    return data


def build_model_and_tokenizer(model_name: str):
    """Load the base model and tokenizer."""
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        raise ImportError("Install transformers: pip install transformers")

    logger.info(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name)
    return model, tokenizer


def train(config: TrainingConfig, resume_from: Optional[str] = None):
    """Run the training loop."""
    import torch

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(config.dataset_path)
    model, tokenizer = build_model_and_tokenizer(config.model_name)

    try:
        from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling
        from datasets import Dataset
    except ImportError:
        raise ImportError("Install dependencies: pip install transformers datasets")

    # Format examples as plain text if they have prompt/response fields
    def format_example(example: dict) -> dict:
        if "text" in example:
            return {"text": example["text"]}
        prompt = example.get("prompt", "")
        response = example.get("response", "")
        return {"text": f"{prompt}\n{response}"}

    formatted = [format_example(ex) for ex in dataset]
    hf_dataset = Dataset.from_list(formatted)

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=config.max_length,
            padding="max_length",
        )

    tokenized = hf_dataset.map(tokenize, batched=True, remove_columns=["text"])

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=config.epochs,
        per_device_train_batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        warmup_steps=config.warmup_steps,
        save_steps=config.save_steps,
        logging_steps=50,
        seed=config.seed,
        resume_from_checkpoint=resume_from,
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=data_collator,
    )

    logger.info("Starting training...")
    trainer.train(resume_from_checkpoint=resume_from)

    logger.info(f"Saving final model to {output_dir}")
    trainer.save_model()
    tokenizer.save_pretrained(str(output_dir))

    return trainer
