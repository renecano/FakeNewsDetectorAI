# =============================================================
#  FakeNewsDetectorAI — training/train.py
#  Fine-tuning de DistilBERT para detección de fake news
#  Incluye: dataset propio, métricas, guardado del modelo
# =============================================================

import os
import sys
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, f1_score
)

import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    DataCollatorWithPadding,
)

# ── Paths ────────────────────────────────────────────────────
ROOT        = Path(__file__).resolve().parent.parent
DATA_PATH   = ROOT / "data" / "train_dataset.csv"
MODEL_OUT   = ROOT / "models" / "fakenews_model"
LOG_DIR     = ROOT / "models" / "logs"

MODEL_OUT.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_DIR / "training.log"),
    ]
)
logger = logging.getLogger("Trainer")

# ── Configuración ────────────────────────────────────────────
BASE_MODEL   = "distilbert-base-uncased"   # Rápido y efectivo
MAX_LENGTH   = 512
BATCH_SIZE   = 8
EPOCHS       = 5
LR           = 2e-5
TEST_SIZE    = 0.2
SEED         = 42

LABEL2ID = {"REAL": 0, "FAKE": 1}
ID2LABEL = {0: "REAL", 1: "FAKE"}


# ── 1. Cargar y preparar datos ───────────────────────────────
def load_data(path: Path) -> tuple[Dataset, Dataset]:
    logger.info(f"Cargando dataset: {path}")
    df = pd.read_csv(path)

    # Validar columnas
    assert "text" in df.columns and "label" in df.columns, \
        "El CSV debe tener columnas 'text' y 'label'"

    df = df.dropna(subset=["text", "label"])
    df["label"] = df["label"].str.upper().str.strip()
    df = df[df["label"].isin(LABEL2ID.keys())]
    df["label_id"] = df["label"].map(LABEL2ID)

    logger.info(f"Total ejemplos: {len(df)}")
    logger.info(f"Distribución:\n{df['label'].value_counts().to_string()}")

    # Split
    train_df, test_df = train_test_split(
        df, test_size=TEST_SIZE, random_state=SEED, stratify=df["label"]
    )
    logger.info(f"Train: {len(train_df)} | Test: {len(test_df)}")

    train_ds = Dataset.from_pandas(train_df[["text", "label_id"]].reset_index(drop=True))
    test_ds  = Dataset.from_pandas(test_df[["text", "label_id"]].reset_index(drop=True))

    return train_ds, test_ds


# ── 2. Tokenización ──────────────────────────────────────────
def tokenize_dataset(train_ds, test_ds, tokenizer):
    def tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            padding=False,
            max_length=MAX_LENGTH,
        )

    train_ds = train_ds.map(tokenize, batched=True)
    test_ds  = test_ds.map(tokenize, batched=True)

    # Renombrar label_id → labels (requerido por Trainer)
    train_ds = train_ds.rename_column("label_id", "labels")
    test_ds  = test_ds.rename_column("label_id", "labels")

    train_ds.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    test_ds.set_format("torch",  columns=["input_ids", "attention_mask", "labels"])

    return train_ds, test_ds


# ── 3. Métricas ──────────────────────────────────────────────
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, predictions)
    f1  = f1_score(labels, predictions, average="weighted")
    return {"accuracy": round(acc, 4), "f1": round(f1, 4)}


# ── 4. Reporte detallado post-entrenamiento ──────────────────
def full_evaluation(model, tokenizer, test_ds):
    logger.info("\n" + "="*55)
    logger.info("EVALUACIÓN FINAL DEL MODELO")
    logger.info("="*55)

    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    all_preds, all_labels = [], []

    for i in range(len(test_ds)):
        sample = test_ds[i]
        input_ids      = sample["input_ids"].unsqueeze(0).to(device)
        attention_mask = sample["attention_mask"].unsqueeze(0).to(device)
        label          = sample["labels"].item()

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            pred = torch.argmax(outputs.logits, dim=-1).item()

        all_preds.append(pred)
        all_labels.append(label)

    # Classification report
    report = classification_report(
        all_labels, all_preds,
        target_names=list(LABEL2ID.keys()),
        digits=4
    )
    logger.info(f"\nClassification Report:\n{report}")

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    logger.info(f"\nConfusion Matrix:\n{cm}")
    logger.info(f"  Rows = Real labels | Cols = Predicted")

    # Ejemplos difíciles (donde el modelo falló)
    failures = [i for i, (p, l) in enumerate(zip(all_preds, all_labels)) if p != l]
    if failures:
        logger.info(f"\n⚠️  Ejemplos donde falló el modelo ({len(failures)}):")
        for idx in failures[:5]:
            text_preview = test_ds[idx]["input_ids"][:20]
            logger.info(f"  - Sample {idx}: pred={ID2LABEL[all_preds[idx]]} | real={ID2LABEL[all_labels[idx]]}")
    else:
        logger.info("\n✅ El modelo clasificó todos los ejemplos de test correctamente.")

    return all_preds, all_labels


# ── 5. Prueba con el ejemplo problemático ────────────────────
def test_hard_cases(model, tokenizer):
    hard_cases = [
        {
            "label": "FAKE",
            "text": "Scientists Confirm That Drinking Coffee Daily Completely Prevents Cancer. "
                    "A group of researchers from a European university has reportedly discovered that "
                    "drinking at least three cups of coffee per day can completely prevent cancer. "
                    "The study claims 100% protection rate against all forms of cancer. "
                    "The full study has not yet been published in any recognized scientific journal."
        },
        {
            "label": "REAL",
            "text": "A peer-reviewed study published in the New England Journal of Medicine found a "
                    "moderate association between daily coffee consumption and reduced risk of certain "
                    "liver conditions. Researchers noted that more studies are needed to establish causation."
        },
        {
            "label": "FAKE",
            "text": "URGENT!! Doctors DON'T WANT YOU TO KNOW THIS. Cure cancer with lemon. "
                    "SHARE BEFORE THEY DELETE THIS!"
        },
    ]

    logger.info("\n" + "="*55)
    logger.info("PRUEBA CON CASOS DIFÍCILES")
    logger.info("="*55)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    model.to(device)

    for case in hard_cases:
        inputs = tokenizer(
            case["text"], return_tensors="pt",
            truncation=True, max_length=MAX_LENGTH, padding=True
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)[0]
            pred_id = torch.argmax(probs).item()
            pred_label = ID2LABEL[pred_id]
            confidence = probs[pred_id].item()

        status = "✅" if pred_label == case["label"] else "❌"
        logger.info(
            f"\n{status} Esperado: {case['label']} | Predicho: {pred_label} ({confidence*100:.1f}%)"
            f"\n   Texto: {case['text'][:80]}..."
        )


# ── 6. Main ──────────────────────────────────────────────────
def main():
    logger.info(f"Device: {'GPU ✅' if torch.cuda.is_available() else 'CPU'}")
    logger.info(f"Modelo base: {BASE_MODEL}")

    # Cargar datos
    train_ds, test_ds = load_data(DATA_PATH)

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    train_ds, test_ds = tokenize_dataset(train_ds, test_ds, tokenizer)

    # Modelo
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL,
        num_labels=len(LABEL2ID),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    # Data collator
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Argumentos de entrenamiento
    training_args = TrainingArguments(
        output_dir=str(MODEL_OUT),
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=LR,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        logging_steps=10,
        report_to="none",
        seed=SEED,
        fp16=torch.cuda.is_available(),
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        processing_class=tokenizer,   # 'tokenizer' fue renombrado en transformers v5+
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    # Entrenar
    logger.info("\n🚀 Iniciando entrenamiento...")
    trainer.train()

    # Evaluación completa
    full_evaluation(model, tokenizer, test_ds)

    # Prueba con casos difíciles
    test_hard_cases(model, tokenizer)

    # Guardar modelo final
    logger.info(f"\n💾 Guardando modelo en: {MODEL_OUT}")
    model.save_pretrained(str(MODEL_OUT))
    tokenizer.save_pretrained(str(MODEL_OUT))
    logger.info("✅ Modelo guardado exitosamente.")
    logger.info(f"\nPara usar el modelo en predictor.py, cambia model_primary a:\n  '{MODEL_OUT}'")


if __name__ == "__main__":
    main()