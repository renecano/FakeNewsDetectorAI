# =============================================================
#  FakeNewsDetectorAI — predictor.py  v2.1
#  Motor de predicción con fusión reforzada para pseudociencia
# =============================================================

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from transformers import pipeline, Pipeline

from labels import LABELS, ID_TO_LABEL, PROJECT_META
from preprocess import (
    clean_text, extract_linguistic_features,
    validate_input, features_summary
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("FakeNewsDetector")

ROOT = Path(__file__).resolve().parent.parent


@dataclass
class PredictionResult:
    label: str
    display: str
    confidence: float
    confidence_pct: str
    scores: Dict[str, float]
    features: Dict
    features_text: str
    inference_ms: float
    model_used: str
    raw_text_preview: str
    warning: Optional[str] = None
    error: Optional[str] = None


class FakeNewsDetector:

    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or PROJECT_META["model_primary"]
        self.pipe: Optional[Pipeline] = None
        self._loaded = False
        self._load_model()

    def _load_model(self):
        # Priorizar modelo fine-tuned local si existe
        local_model = ROOT / "models" / "fakenews_model"
        models_to_try = []

        if local_model.exists():
            models_to_try.append(str(local_model))
            logger.info(f"Modelo local encontrado: {local_model}")
        
        models_to_try += [
            PROJECT_META["model_fallback"],
            PROJECT_META["model_fallback2"],
        ]

        for model in models_to_try:
            try:
                logger.info(f"Cargando modelo: {model} ...")
                self.pipe = pipeline(
                    "text-classification",
                    model=model,
                    tokenizer=model,
                    top_k=None,
                    truncation=True,
                    max_length=PROJECT_META["max_tokens"],
                    device=0 if torch.cuda.is_available() else -1,
                )
                self.model_name = model
                self._loaded = True
                logger.info(f"✅ Modelo cargado: {model}")
                return
            except Exception as e:
                logger.warning(f"No se pudo cargar '{model}': {e}")

        logger.error("⚠️ Ningún modelo disponible. Usando modo heurístico puro.")

    def _map_label(self, model_label: str) -> str:
        label_upper = model_label.upper()
        if label_upper in ("FAKE", "0", "LABEL_0", "NEGATIVE", "NEG"):
            return "FAKE"
        if label_upper in ("REAL", "1", "LABEL_1", "TRUE", "POSITIVE", "POS"):
            return "REAL"
        return "DOUBTFUL"

    def _neural_predict(self, clean: str) -> Tuple[Dict[str, float], str]:
        if not self._loaded or self.pipe is None:
            return {"REAL": 0.33, "DOUBTFUL": 0.34, "FAKE": 0.33}, "heuristic_only"

        raw_results = self.pipe(clean)[0]
        scores: Dict[str, float] = {"REAL": 0.0, "DOUBTFUL": 0.0, "FAKE": 0.0}

        if len(raw_results) == 2:
            for r in raw_results:
                mapped = self._map_label(r["label"])
                scores[mapped] += r["score"]
            gap = abs(scores["REAL"] - scores["FAKE"])
            if gap < 0.35:
                transfer = (0.35 - gap) * 0.4
                scores["REAL"]     -= transfer / 2
                scores["FAKE"]     -= transfer / 2
                scores["DOUBTFUL"] += transfer
        else:
            for r in raw_results:
                mapped = self._map_label(r["label"])
                scores[mapped] = max(scores[mapped], r["score"])

        total = sum(scores.values()) or 1.0
        return {k: round(v / total, 4) for k, v in scores.items()}, self.model_name

    def _fuse_with_heuristics(
        self,
        scores: Dict[str, float],
        features: Dict,
    ) -> Dict[str, float]:
        """
        Fusión inteligente modelo + heurísticas.

        Lógica clave para pseudociencia:
        - Si alarm_score > 0.6  → el modelo cede 35% de peso a heurísticas
        - Si hay patrones pseudocientíficos → override directo hacia FAKE
        - Si hay múltiples señales reales verificables → refuerzo de REAL
        """
        alarm      = features.get("alarm_score", 0.0)
        pseudo     = features.get("pseudoscience_hits", [])
        fake_hits  = features.get("fake_signal_hits", [])
        real_hits  = features.get("real_signal_hits", [])

        # ── Override por pseudociencia fuerte ────────────────
        # Si hay 2+ patrones regex de pseudociencia, es casi certeza FAKE
        if len(pseudo) >= 2:
            scores["FAKE"]     = max(scores["FAKE"], 0.75)
            scores["REAL"]     = min(scores["REAL"], 0.15)
            scores["DOUBTFUL"] = 1.0 - scores["FAKE"] - scores["REAL"]

        elif len(pseudo) == 1 or alarm > 0.6:
            # Peso heurístico del 35%
            HW = 0.35
            scores["FAKE"]  = scores["FAKE"]  * (1 - HW) + alarm * HW
            scores["REAL"]  = scores["REAL"]  * (1 - HW * 0.5)

        elif len(fake_hits) >= 3:
            HW = 0.25
            scores["FAKE"]  = scores["FAKE"]  * (1 - HW) + 0.8 * HW
            scores["REAL"]  = scores["REAL"]  * (1 - HW)

        # ── Refuerzo de noticias reales ──────────────────────
        if len(real_hits) >= 3 and alarm < 0.2 and len(pseudo) == 0:
            HW = 0.20
            scores["REAL"] = scores["REAL"] * (1 - HW) + 0.9 * HW
            scores["FAKE"] = scores["FAKE"] * (1 - HW)

        # ── Re-normalizar ────────────────────────────────────
        scores = {k: max(v, 0.0) for k, v in scores.items()}
        total = sum(scores.values()) or 1.0
        return {k: round(v / total, 4) for k, v in scores.items()}

    def predict(self, text: str) -> PredictionResult:
        start = time.perf_counter()

        valid, err_msg = validate_input(text)
        if not valid:
            return PredictionResult(
                label="DOUBTFUL", display=LABELS["DOUBTFUL"].display_es,
                confidence=0.0, confidence_pct="—",
                scores={}, features={}, features_text="",
                inference_ms=0, model_used="—",
                raw_text_preview="", error=err_msg,
            )

        clean = clean_text(text)
        features = extract_linguistic_features(clean)
        feat_text = features_summary(features)

        scores, model_used = self._neural_predict(clean)
        scores = self._fuse_with_heuristics(scores, features)

        best_label = max(scores, key=scores.get)
        confidence = scores[best_label]
        label_info = LABELS[best_label]

        warning = None
        if confidence < label_info.confidence_threshold:
            warning = (
                f"⚠️ Confianza baja ({confidence*100:.1f}%). "
                "Contrasta con otras fuentes antes de sacar conclusiones."
            )

        elapsed_ms = (time.perf_counter() - start) * 1000

        return PredictionResult(
            label=best_label,
            display=label_info.display_es,
            confidence=confidence,
            confidence_pct=f"{confidence * 100:.1f}%",
            scores=scores,
            features=features,
            features_text=feat_text,
            inference_ms=round(elapsed_ms, 1),
            model_used=str(model_used).split("/")[-1],
            raw_text_preview=clean[:120] + ("…" if len(clean) > 120 else ""),
            warning=warning,
        )

    def predict_batch(self, texts: List[str]) -> List[PredictionResult]:
        return [self.predict(t) for t in texts]

    def model_info(self) -> Dict:
        return {
            "model": self.model_name,
            "loaded": self._loaded,
            "device": "GPU" if torch.cuda.is_available() else "CPU",
            "max_tokens": PROJECT_META["max_tokens"],
        }


_detector_instance: Optional[FakeNewsDetector] = None


def get_detector() -> FakeNewsDetector:
    global _detector_instance
    if _detector_instance is None:
        _detector_instance = FakeNewsDetector()
    return _detector_instance