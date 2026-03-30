# FakeNews Detector AI

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)
![Gradio](https://img.shields.io/badge/Gradio-4.0%2B-FF7C00?style=for-the-badge&logo=gradio&logoColor=white)
![Accuracy](https://img.shields.io/badge/Accuracy-99.9%25-22c55e?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-6366f1?style=for-the-badge)

**An AI-powered fake news and misinformation detection system built on RoBERTa Transformers and NLP heuristics.**

*Classifies news articles as Reliable, Doubtful, or Fake — including pseudoscientific content disguised as legitimate journalism.*

[Demo](#demo) · [Features](#features) · [Installation](#installation) · [How It Works](#how-it-works) · [Training](#training)

</div>

---

## The Problem

Misinformation spreads faster than corrections. A 2023 Reuters Institute study found that false news travels six times faster on social media than verified reporting. The challenge is not just obvious clickbait — the most dangerous misinformation mimics the language and structure of legitimate science.

Consider this headline:

> *"Scientists Confirm That Drinking Coffee Daily Completely Prevents Cancer. Researchers from a European university reportedly discovered a 100% protection rate. The full study has not yet been published in any recognized scientific journal."*

A human reader might hesitate. Most AI classifiers trained only on obvious fake news would mark it as **real** — because it uses words like "researchers", "university", and "study". FakeNewsDetectorAI catches it.

---

## Features

- **Three-class classification** — Reliable, Doubtful, or Fake with confidence scores
- **Pseudoscience detection** — catches fake academic language patterns (`"100% protection"`, `"not yet published"`, `"unnamed university"`)
- **Neural + heuristic fusion** — 80% RoBERTa model weight + 20% linguistic heuristics
- **Automatic fallback** — degrades gracefully if local model is unavailable
- **Editorial web interface** — clean, high-contrast UI built with Gradio
- **REST API included** — Gradio exposes `/run/predict` automatically
- **Extended justifications** — explains *why* a text is classified as misinformation
- **GPU/CPU support** — auto-detects available hardware

---

## Demo

<div align="center">

| Input | Classification | Confidence |
|-------|---------------|------------|
| *"Scientists confirm coffee prevents cancer 100%... not yet published"* | 🚫 Fake | 98.9% |
| *"URGENT!! Doctors DON'T WANT YOU TO KNOW THIS!!"* | 🚫 Fake | 99.8% |
| *"Fed raised rates 0.25% according to official statement"* | ✅ Reliable | 98.2% |
| *"Peer-reviewed study in NEJM found moderate association..."* | ✅ Reliable | 96.5% |

</div>

---

## Project Structure

```
FakeNewsDetectorAI/
│
├── app/
│   ├── labels.py          # Label definitions, linguistic signals, metadata
│   ├── preprocess.py      # Text cleaning, feature extraction, pseudoscience detection
│   ├── predictor.py       # RoBERTa inference engine + heuristic fusion
│   └── main.py            # Gradio web interface
│
├── data/
│   ├── prepare_dataset.py         # Merges Fake.csv + True.csv into training set
│   └── pseudoscience_examples.csv # Hand-crafted pseudoscience training examples
│
├── training/
│   └── train.py           # Fine-tuning script with evaluation and hard-case testing
│
├── models/
│   └── fakenews_model/    # Trained model (not included — see Training section)
│
└── requirements.txt
```

---

## How It Works

### Prediction Pipeline

```
Raw text input
      │
      ▼
[preprocess.py]
  • Remove URLs, mentions, emojis
  • Normalize unicode
  • Extract linguistic features
  • Detect pseudoscience patterns (regex)
  • Calculate alarm_score (0.0 – 1.0)
      │
      ▼
[predictor.py]
  • Tokenize with RoBERTa tokenizer
  • Run neural inference (12 attention layers)
  • Get softmax scores {REAL, DOUBTFUL, FAKE}
  • Fuse with heuristics (80/20 weighting)
      │
      ▼
[labels.py]
  • Map to final label
  • Apply confidence thresholds
  • Generate warnings if confidence is low
      │
      ▼
[main.py]
  • Render result card with verdict, bars, indicators
```

### Neural + Heuristic Fusion

The system combines two complementary approaches:

**RoBERTa** understands context through self-attention — it knows that *"not yet been published"* negates *"researchers discovered"* even when separated by several words.

**Heuristics** apply explicit rules:

```python
# Pseudoscience override — 2+ regex patterns = high confidence FAKE
if len(pseudoscience_hits) >= 2:
    scores["FAKE"] = max(scores["FAKE"], 0.75)

# Alarm fusion — 35% heuristic weight when alarm is high
elif alarm_score > 0.6:
    scores["FAKE"] = scores["FAKE"] * 0.65 + alarm_score * 0.35

# Credibility boost — reinforce REAL when verified sources detected
if len(real_signal_hits) >= 3 and alarm_score < 0.2:
    scores["REAL"] = scores["REAL"] * 0.80 + 0.9 * 0.20
```

### Pseudoscience Detection

The hardest cases are fake news articles that mimic scientific language. The system uses regex patterns specifically designed to catch them:

```python
PSEUDOSCIENCE_PATTERNS = [
    r"\b100\s*%\s*(protection|effectiveness|cure|success rate)",
    r"completely\s+(prevent|cure|reverse|eliminate)",
    r"(not|never)\s+(yet\s+)?(been\s+)?(published|peer.reviewed)",
    r"extend\s+life\s+expectancy\s+by\s+\d+\s+years",
    r"(unnamed|anonymous|undisclosed)\s+(university|researchers|scientists)",
    r"(secret(ly)?|hidden|suppressed)\s+(cure|treatment|study)",
]
```

---

## Installation

### Prerequisites

- Python 3.10+
- 4GB RAM minimum (8GB recommended)
- GPU optional but recommended for training

### Setup

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/FakeNewsDetectorAI.git
cd FakeNewsDetectorAI

# Create virtual environment
python -m venv venv

# Activate (Windows)
.\venv\Scripts\activate

# Activate (macOS / Linux)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## Training

The trained model is not included in this repository due to file size (~500MB). You have two options:

### Option A — Train locally (CPU, ~8 hours)

```bash
# 1. Download the dataset
#    Get Fake.csv and True.csv from:
#    https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset
#    Place both files in the data/ folder

# 2. Prepare the dataset
python data/prepare_dataset.py

# 3. Train
python training/train.py
```

### Option B — Train on Google Colab (GPU T4, ~12 minutes) ✅ Recommended

1. Open [Google Colab](https://colab.research.google.com)
2. Upload `FakeNewsDetector_RoBERTa.ipynb`
3. Set runtime to **GPU T4**: `Runtime → Change runtime type → T4 GPU`
4. Run all cells in order
5. Download the model ZIP from the last cell
6. Extract and place contents in `models/fakenews_model/`

### Training Results

```
Model: roberta-base (fine-tuned)
Dataset: ~8,500 news articles (balanced FAKE/REAL)
         + 30 pseudoscience examples
         + 15 verified scientific reporting examples

Epoch │ Train Loss │ Val Loss │ Accuracy │ F1
──────┼────────────┼──────────┼──────────┼────────
  1   │  0.008785  │ 0.008627 │  99.84%  │ 99.84%
  2   │  0.000578  │ 0.009790 │  99.84%  │ 99.84%
  3*  │  0.000352  │ 0.030879 │  99.38%  │ 99.37%

* EarlyStopping triggered at epoch 3 — best model from epoch 2 saved
```

---

## Running the App

```bash
cd app
python main.py
```

Open **http://localhost:7860** in your browser.

To generate a public shareable link (valid 72 hours):

```python
# In main.py, change:
demo.launch(share=False)
# To:
demo.launch(share=True)
```

### REST API

Gradio automatically exposes a REST API:

```bash
curl -X POST http://localhost:7860/run/predict \
  -H "Content-Type: application/json" \
  -d '{"data": ["YOUR NEWS TEXT HERE"]}'
```

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Core model | RoBERTa-base (HuggingFace Transformers) |
| Training framework | PyTorch + HuggingFace Trainer |
| Web interface | Gradio 4.x |
| NLP preprocessing | Custom regex + heuristics |
| Training infrastructure | Google Colab (NVIDIA T4 GPU) |
| Dataset | Kaggle Fake and Real News Dataset (~45K articles) |

---

## Dataset

Base training data from the [Fake and Real News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset) on Kaggle:

- **23,481** fake news articles
- **21,417** real news articles (Reuters)
- Sampled to 8,500 balanced examples for training efficiency

Additionally, 45 hand-crafted examples were added covering:
- Pseudoscientific claims with fake academic language
- Verified scientific reporting with proper sourcing

---

## Limitations

- Primarily trained on English-language news
- Dataset is focused on US political news (Reuters + political blogs)
- Performance may vary on non-political topics in Spanish
- Does not have access to real-time fact-checking databases

---

## Future Improvements

- Multilingual support (Spanish, Portuguese) with XLM-RoBERTa
- Integration with fact-checking APIs (Snopes, PolitiFact)
- FastAPI REST backend for production deployment
- Browser extension for real-time article scanning
- Confidence calibration with temperature scaling

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

<div align="center">
Built for Hackathon / Expo · 2025
</div>
