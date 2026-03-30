# =============================================================
#  FakeNewsDetectorAI — data/prepare_dataset.py
#  Combina Fake.csv y True.csv en train_dataset.csv listo
#  para entrenar. Ejecutar desde la carpeta raíz del proyecto.
# =============================================================

import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

# ── Rutas ────────────────────────────────────────────────────
ROOT      = Path(__file__).resolve().parent.parent
DATA_DIR  = ROOT / "data"
FAKE_PATH = DATA_DIR / "Fake.csv"
REAL_PATH = DATA_DIR / "True.csv"
OUT_TRAIN = DATA_DIR / "train_dataset.csv"
OUT_TEST  = DATA_DIR / "test_dataset.csv"

# ── Config ───────────────────────────────────────────────────
SAMPLE_PER_CLASS = 5000   # Cuántos ejemplos usar por clase (5k c/u = 10k total)
                           # Sube a 10000 si quieres más precisión (tarda más)
TEST_SIZE        = 0.15   # 15% para evaluación
SEED             = 42
MAX_TEXT_CHARS   = 1800   # Truncar textos muy largos

def prepare():
    print("📂 Cargando archivos...")
    fake_df = pd.read_csv(FAKE_PATH)
    real_df = pd.read_csv(REAL_PATH)

    print(f"   Fake.csv: {len(fake_df):,} filas | columnas: {list(fake_df.columns)}")
    print(f"   True.csv: {len(real_df):,} filas | columnas: {list(real_df.columns)}")

    # ── Combinar title + text ────────────────────────────────
    def build_text(row):
        title = str(row.get("title", "")).strip()
        body  = str(row.get("text", "")).strip()
        combined = f"{title}. {body}" if title else body
        return combined[:MAX_TEXT_CHARS]

    fake_df["text_combined"] = fake_df.apply(build_text, axis=1)
    real_df["text_combined"] = real_df.apply(build_text, axis=1)

    # ── Asignar etiquetas ────────────────────────────────────
    fake_df["label"] = "FAKE"
    real_df["label"] = "REAL"

    # ── Samplear para balancear ──────────────────────────────
    n_fake = min(SAMPLE_PER_CLASS, len(fake_df))
    n_real = min(SAMPLE_PER_CLASS, len(real_df))

    fake_sample = fake_df[["text_combined", "label"]].sample(n=n_fake, random_state=SEED)
    real_sample = real_df[["text_combined", "label"]].sample(n=n_real, random_state=SEED)

    # ── Combinar y renombrar ─────────────────────────────────
    combined = pd.concat([fake_sample, real_sample], ignore_index=True)
    combined = combined.rename(columns={"text_combined": "text"})
    combined = combined.dropna(subset=["text", "label"])
    combined = combined[combined["text"].str.strip().str.len() > 50]
    combined = combined.sample(frac=1, random_state=SEED).reset_index(drop=True)

    # ── Split train / test ───────────────────────────────────
    train_df, test_df = train_test_split(
        combined, test_size=TEST_SIZE, random_state=SEED, stratify=combined["label"]
    )

    # ── Guardar ──────────────────────────────────────────────
    train_df.to_csv(OUT_TRAIN, index=False)
    test_df.to_csv(OUT_TEST,  index=False)

    print(f"\n✅ Dataset listo:")
    print(f"   Train: {len(train_df):,} ejemplos → {OUT_TRAIN}")
    print(f"   Test:  {len(test_df):,} ejemplos  → {OUT_TEST}")
    print(f"\n   Distribución train:")
    print(train_df["label"].value_counts().to_string())
    print(f"\n   Distribución test:")
    print(test_df["label"].value_counts().to_string())
    print(f"\n🚀 Ahora corre: python training/train.py")


if __name__ == "__main__":
    prepare()