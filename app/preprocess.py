# =============================================================
#  FakeNewsDetectorAI — preprocess.py  v2.2
#  Limpieza + features + señales con nombres en español
# =============================================================

import re
import string
import unicodedata
from typing import Dict, List, Tuple

from labels import FAKE_SIGNALS, REAL_SIGNALS, DOUBTFUL_SIGNALS, PSEUDOSCIENCE_PATTERNS

MAX_CHARS = 2000
MIN_WORDS = 10
EXCLAMATION_THRESHOLD = 2
CAPS_RATIO_THRESHOLD  = 0.3

# ── Traducciones de señales al español ───────────────────────
SIGNAL_TRANSLATIONS = {
    # Fake / alarmistas
    "completely prevents":      "previene completamente",
    "completely cures":         "cura completamente",
    "completely reverses":      "revierte completamente",
    "100% protection":          "protección al 100%",
    "100% effectiveness":       "efectividad del 100%",
    "zero risk":                "riesgo cero",
    "miracle cure":             "cura milagrosa",
    "secret cure":              "cura secreta",
    "miraculous":               "milagroso",
    "doctors don't want":       "los médicos no quieren",
    "big pharma hiding":        "farmacéuticas ocultando",
    "government is hiding":     "el gobierno oculta",
    "not yet been published":   "aún no publicado",
    "has not been published":   "no ha sido publicado",
    "has not yet been peer-reviewed": "sin revisión científica",
    "allegedly discovered":     "supuestamente descubierto",
    "reportedly discovered":    "presuntamente descubierto",
    "unnamed university":       "universidad sin nombre",
    "unnamed researchers":      "investigadores anónimos",
    "undisclosed institute":    "instituto no revelado",
    "anonymous source":         "fuente anónima",
    "fuentes anónimas":         "fuentes anónimas",
    "extend life expectancy by": "extiende la vida",
    "neutralizing cancer cells": "neutralizar células cancerígenas",
    "slow aging":               "retrasar el envejecimiento",
    "revolutionize preventive medicine": "revolucionar la medicina",
    "can completely":           "puede completamente",
    "whistleblower claims":     "denuncia de informante",
    "SHOCKING":                 "impactante",
    "YOU WON'T BELIEVE":        "no lo creerás",
    "SHARE BEFORE DELETED":     "comparte antes de que lo borren",
    "THEY DON'T WANT YOU TO KNOW": "no quieren que lo sepas",
    "BREAKING:":                "urgente",
    "URGENT":                   "urgente",
    "URGENTE":                  "urgente",
    "comparte antes de que lo borren": "comparte antes de que lo borren",
    "lo que ocultan":           "lo que ocultan",
    "nadie te dice":            "nadie te dice",
    # Real / verificables
    "according to":             "según",
    "según":                    "según",
    "de acuerdo con":           "de acuerdo con",
    "confirmed":                "confirmado",
    "confirmó":                 "confirmó",
    "announced":                "anunciado",
    "anunció":                  "anunció",
    "published in":             "publicado en",
    "publicado en":             "publicado en",
    "peer-reviewed":            "revisado por pares",
    "revisado por pares":       "revisado por pares",
    "clinical trial":           "ensayo clínico",
    "ensayo clínico":           "ensayo clínico",
    "randomized controlled":    "ensayo controlado aleatorio",
    "more studies are needed":  "se necesitan más estudios",
    "further research":         "investigación adicional",
    "official statement":       "declaración oficial",
    "declaración oficial":      "declaración oficial",
    "press conference":         "conferencia de prensa",
    "New England Journal":      "New England Journal",
    "The Lancet":               "The Lancet",
    "Nature Medicine":          "Nature Medicine",
    "WHO":                      "OMS",
    "CDC":                      "CDC",
    "NIH":                      "NIH",
    "NASA":                     "NASA",
    "Johns Hopkins":            "Johns Hopkins",
    "Harvard":                  "Harvard",
    "MIT":                      "MIT",
    "Stanford":                 "Stanford",
    "effect sizes":             "tamaño del efecto",
    "statistically significant": "estadísticamente significativo",
    "peer review":              "revisión científica",
    # Dudosas
    "se dice que":              "se dice que",
    "circula en redes":         "circula en redes sociales",
    "some experts say":         "algunos expertos dicen",
    "experts say":              "expertos afirman",
    "could revolutionize":      "podría revolucionar",
    "sources say":              "según fuentes",
    "unverified":               "sin verificar",
    "sin verificar":            "sin verificar",
    "supuestamente":            "supuestamente",
    "allegedly":                "presuntamente",
    "could potentially":        "podría potencialmente",
}

def translate_signal(signal: str) -> str:
    """Traduce una señal detectada al español."""
    key = signal.strip()
    return SIGNAL_TRANSLATIONS.get(key, SIGNAL_TRANSLATIONS.get(key.lower(), key))


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text[:MAX_CHARS]
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"https?://\S+|www\.\S+", " [URL] ", text)
    text = re.sub(r"@\w+", " [USUARIO] ", text)
    text = re.sub(r"#(\w+)", r" \1 ", text)
    text = re.sub(
        r"[\U00010000-\U0010ffff]|[\u2600-\u26FF]|[\u2700-\u27BF]",
        " ", text, flags=re.UNICODE
    )
    text = re.sub(r"[\r\n\t]+", " ", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()


def simple_tokenize(text: str) -> List[str]:
    text = text.lower().translate(str.maketrans("", "", string.punctuation))
    return [t for t in text.split() if len(t) > 1]


def detect_pseudoscience(text: str) -> List[str]:
    hits = []
    for pattern in PSEUDOSCIENCE_PATTERNS:
        if re.search(pattern, text, flags=re.IGNORECASE):
            hits.append(pattern)
    return hits


def extract_linguistic_features(raw_text: str) -> Dict:
    if not raw_text:
        return {}

    words = raw_text.split()
    total_words = max(len(words), 1)

    caps_words   = sum(1 for w in words if w.isupper() and len(w) > 2)
    caps_ratio   = caps_words / total_words
    exclamations = len(re.findall(r"[!¡]{2,}|[?¿]{2,}", raw_text))

    text_upper = raw_text.upper()
    fake_hits   = [s for s in FAKE_SIGNALS    if s.upper() in text_upper]
    real_hits   = [s for s in REAL_SIGNALS    if s.lower() in raw_text.lower()]
    doubt_hits  = [s for s in DOUBTFUL_SIGNALS if s.lower() in raw_text.lower()]
    pseudo_hits = detect_pseudoscience(raw_text)

    is_too_short = total_words < MIN_WORDS
    has_urls     = "[URL]" in raw_text

    pseudo_weight = min(len(pseudo_hits) * 0.4, 0.8)
    fake_weight   = min(len(fake_hits)   * 0.15, 0.5)
    real_penalty  = min(len(real_hits)   * 0.12, 0.4)
    caps_contrib  = (caps_ratio / CAPS_RATIO_THRESHOLD) * 0.2
    excl_contrib  = (exclamations / EXCLAMATION_THRESHOLD) * 0.2

    alarm_score = max(
        min(pseudo_weight + fake_weight + caps_contrib + excl_contrib - real_penalty, 1.0),
        0.0
    )

    return {
        "word_count":           total_words,
        "char_count":           len(raw_text),
        "caps_ratio":           round(caps_ratio, 3),
        "exclamation_abuse":    exclamations,
        "fake_signal_hits":     fake_hits,
        "real_signal_hits":     real_hits,
        "doubtful_signal_hits": doubt_hits,
        "pseudoscience_hits":   pseudo_hits,
        "is_too_short":         is_too_short,
        "has_urls":             has_urls,
        "alarm_score":          round(alarm_score, 3),
    }


def validate_input(text: str) -> Tuple[bool, str]:
    if not text or not text.strip():
        return False, "⚠️ El campo de texto está vacío."
    words = text.strip().split()
    if len(words) < MIN_WORDS:
        return False, (
            f"⚠️ Texto demasiado corto ({len(words)} palabras). "
            f"Ingresa al menos {MIN_WORDS} palabras para un análisis preciso."
        )
    if len(text) > MAX_CHARS * 2:
        return False, f"⚠️ Texto demasiado largo. Máximo {MAX_CHARS * 2} caracteres."
    return True, ""


def features_summary(features: Dict) -> str:
    lines = []
    if features.get("pseudoscience_hits"):
        lines.append(
            f"🔬 Patrones pseudocientíficos detectados ({len(features['pseudoscience_hits'])}): "
            "afirmaciones absolutas o estudios sin publicar"
        )
    if features.get("fake_signal_hits"):
        translated = [translate_signal(h) for h in features['fake_signal_hits'][:3]]
        lines.append(f"🔴 Lenguaje alarmista: {', '.join(translated)}")
    if features.get("real_signal_hits"):
        translated = [translate_signal(h) for h in features['real_signal_hits'][:3]]
        lines.append(f"🟢 Fuentes verificables: {', '.join(translated)}")
    if features.get("doubtful_signal_hits"):
        translated = [translate_signal(h) for h in features['doubtful_signal_hits'][:3]]
        lines.append(f"🟡 Lenguaje no confirmado: {', '.join(translated)}")
    if features.get("caps_ratio", 0) > CAPS_RATIO_THRESHOLD:
        lines.append(f"🔠 Alto uso de mayúsculas ({features['caps_ratio']*100:.0f}%)")
    if features.get("exclamation_abuse", 0) >= EXCLAMATION_THRESHOLD:
        lines.append("❗ Abuso de signos de exclamación")
    if features.get("is_too_short"):
        lines.append("📏 Texto muy corto — precisión reducida")
    return "\n".join(lines) if lines else "Sin señales heurísticas destacables."