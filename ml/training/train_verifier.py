from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from datasets import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from ml import config

LABEL2ID = {"Refuted": 0, "NotEnoughEvidence": 1, "Supported": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}


def _read_rows(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        label = obj["label"]
        label_id = int(label) if isinstance(label, int) else LABEL2ID[str(label)]
        rows.append(
            {
                "premise": str(obj["premise"]),
                "hypothesis": str(obj["hypothesis"]),
                "label": label_id,
            }
        )
    if not rows:
        raise ValueError(f"No rows found in {path}")
    return rows


def _compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = float((preds == labels).mean())
    macro_f1 = 0.0
    for c in (0, 1, 2):
        tp = int(((preds == c) & (labels == c)).sum())
        fp = int(((preds == c) & (labels != c)).sum())
        fn = int(((preds != c) & (labels == c)).sum())
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        macro_f1 += 2 * precision * recall / max(precision + recall, 1e-8)
    macro_f1 /= 3.0
    return {"accuracy": acc, "macro_f1": macro_f1}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train stance verifier transformer.")
    parser.add_argument("--train-path", type=str, required=True, help="JSONL with premise,hypothesis,label")
    parser.add_argument("--val-path", type=str, required=True, help="JSONL with premise,hypothesis,label")
    parser.add_argument("--model-name", type=str, default=config.VERIFIER_MODEL_NAME)
    parser.add_argument("--output-dir", type=str, default=str(config.PROJECT_ROOT / "ml" / "models" / "verifier"))
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-5)
    args = parser.parse_args()

    train_ds = Dataset.from_list(_read_rows(Path(args.train_path)))
    val_ds = Dataset.from_list(_read_rows(Path(args.val_path)))

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=3,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    def tokenize_batch(batch):
        return tokenizer(batch["premise"], batch["hypothesis"], truncation=True)

    train_ds = train_ds.map(tokenize_batch, batched=True)
    val_ds = val_ds.map(tokenize_batch, batched=True)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_steps=20,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=_compute_metrics,
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    metrics = trainer.evaluate()
    print(json.dumps({k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))}, indent=2))


if __name__ == "__main__":
    main()
