"""
train.py — Fine-tune DistilBERT for phishing URL detection.
Run: python train.py
"""

import numpy as np
from datasets import load_dataset
from sklearn.metrics import classification_report
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

# Config:
MODEL_NAME = "distilbert-base-uncased"
OUTPUT_DIR = "phishing-detector-model"
MAX_LENGTH = 128
EPOCHS = 3
TRAIN_BATCH = 32
EVAL_BATCH = 64

def load_data():
    print("📦 Loading dataset...")
    dataset = load_dataset("shawhin/phishing-site-classification")
    print(f"   Train: {len(dataset['train'])} samples")
    print(f"   Test:  {len(dataset['test'])} samples")
    print(f"   Columns: {dataset['train'].column_names}")
    return dataset


def get_text_column(dataset):
    """Auto-detect the column containing URLs/text."""
    candidates = ["url", "text", "URL", "Text", "domain", "query"]
    cols = dataset["train"].column_names
    for c in candidates:
        if c in cols:
            return c
    # Fall back to first non-label column
    return [c for c in cols if c != "label"][0]


def tokenize_dataset(dataset, tokenizer):
    text_col = get_text_column(dataset)
    print(f"Tokenizing using column: '{text_col}'...")

    def tokenize(batch):
        return tokenizer(
            batch[text_col],
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH,
        )

    tokenized = dataset.map(tokenize, batched=True)
    # Dataset already uses "labels" column name, no rename needed
    tokenized.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    return tokenized


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    report = classification_report(
        labels,
        preds,
        target_names=["legit", "phishing"],
        output_dict=True,
        zero_division=0,
    )
    return {
        "accuracy": report["accuracy"],
        "f1_phishing": report["phishing"]["f1-score"],
        "precision_phishing": report["phishing"]["precision"],
        "recall_phishing": report["phishing"]["recall"],
    }


def main():
    # Load tokenizer & model
    print(f"Loading model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

    # Data
    dataset = load_data()
    tokenized = tokenize_dataset(dataset, tokenizer)

    # Training args
    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=TRAIN_BATCH,
        per_device_eval_batch_size=EVAL_BATCH,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_phishing",
        logging_steps=50,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["test"],
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
    )

    # Train
    print("\nStarting training...\n")
    trainer.train()

    # Final eval
    print("\nFinal evaluation:")
    metrics = trainer.evaluate()
    for k, v in metrics.items():
        print(f"   {k}: {v:.4f}" if isinstance(v, float) else f"   {k}: {v}")

    # Save
    print(f"\n💾 Saving model to ./{OUTPUT_DIR}")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("Done! Run predict.py to test your model.")


if __name__ == "__main__":
    main()