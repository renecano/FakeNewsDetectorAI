# =============================================================
#  FakeNewsDetectorAI — labels.py  v2.1
# =============================================================

from dataclasses import dataclass
from typing import Dict, Tuple


@dataclass(frozen=True)
class Label:
    id: int
    name: str
    display_es: str
    description_es: str
    color_hex: str
    emoji: str
    confidence_threshold: float


LABELS: Dict[str, Label] = {
    "REAL": Label(
        id=0,
        name="REAL",
        display_es="✅ Confiable",
        description_es="El texto presenta características de periodismo verificable: "
                       "fuentes citadas, lenguaje neutral y afirmaciones respaldadas "
                       "por instituciones reconocidas.",
        color_hex="#22c55e",
        emoji="✅",
        confidence_threshold=0.65,
    ),
    "DOUBTFUL": Label(
        id=1,
        name="DOUBTFUL",
        display_es="⚠️ Dudosa",
        description_es="El texto mezcla datos verificables con afirmaciones sin respaldo. "
                       "Contrasta con fuentes adicionales antes de compartir.",
        color_hex="#f59e0b",
        emoji="⚠️",
        confidence_threshold=0.50,
    ),
    "FAKE": Label(
        id=2,
        name="FAKE",
        display_es="🚫 Falsa / Desinformación",
        description_es="El texto presenta patrones de desinformación: afirmaciones absolutas "
                       "(100%, completamente), fuentes vagas o inexistentes, lenguaje "
                       "alarmista o pseudocientífico sin publicación verificable.",
        color_hex="#ef4444",
        emoji="🚫",
        confidence_threshold=0.55,
    ),
}

ID_TO_LABEL: Dict[int, str] = {v.id: k for k, v in LABELS.items()}
LABEL_NAMES: Tuple[str, ...] = tuple(LABELS.keys())
NUM_LABELS: int = len(LABELS)


# ── Señales FAKE ─────────────────────────────────────────────
FAKE_SIGNALS = [
    # Clickbait / alarmismo
    "SHOCKING", "YOU WON'T BELIEVE", "SHARE BEFORE DELETED",
    "THEY DON'T WANT YOU TO KNOW", "URGENT", "URGENTE",
    "comparte antes de que lo borren", "lo que ocultan",
    # Pseudociencia — el caso más difícil
    "completely prevents", "completely cures", "completely reverses",
    "100% protection", "100% effectiveness", "zero risk",
    "miracle cure", "secret cure", "miraculous",
    "doctors don't want", "médicos no quieren",
    "big pharma hiding", "government is hiding",
    "not yet been published",        # estudio sin publicar = RED FLAG
    "has not been published",
    "has not yet been peer-reviewed",
    "allegedly discovered",
    "reportedly discovered",
    "unnamed university", "unnamed researchers",
    "undisclosed institute",
    "anonymous source", "fuentes anónimas",
    "extend life expectancy by",
    "neutralizing cancer cells",
    "slow aging",
    "revolutionize preventive medicine",
    "can completely",
    "whistleblower claims",
]

# ── Señales REAL ─────────────────────────────────────────────
REAL_SIGNALS = [
    "according to", "según", "de acuerdo con",
    "confirmed", "confirmó", "announced", "anunció",
    "published in", "publicado en",
    "peer-reviewed", "revisado por pares",
    "clinical trial", "ensayo clínico",
    "randomized controlled",
    "more studies are needed",       # humildad científica = señal positiva
    "further research is needed",
    "official statement", "declaración oficial",
    "press conference",
    "New England Journal", "The Lancet", "Nature Medicine",
    "WHO", "OMS", "CDC", "NIH", "NASA",
    "Johns Hopkins", "Harvard", "MIT", "Stanford",
    "effect sizes", "statistically significant",
    "peer review",
]

# ── Señales DOUBTFUL ─────────────────────────────────────────
DOUBTFUL_SIGNALS = [
    "se dice que", "circula en redes",
    "some experts say",              # vago, sin nombrar a quién
    "experts say",
    "could revolutionize",
    "sources say", "unverified", "sin verificar",
    "supuestamente", "allegedly",
    "could potentially",
]

# ── Patrones regex de pseudociencia ──────────────────────────
PSEUDOSCIENCE_PATTERNS = [
    r"\b100\s*%\s*(protection|effectiveness|cure|success rate)",
    r"completely\s+(prevent|cure|reverse|eliminate)",
    r"(not|never)\s+(yet\s+)?(been\s+)?(published|peer.reviewed)",
    r"extend\s+life\s+expectancy\s+by\s+\d+\s+years",
    r"(unnamed|anonymous|undisclosed)\s+(university|researchers|scientists)",
    r"(secret(ly)?|hidden|suppressed)\s+(cure|treatment|study)",
    r"(no|not\s+any)\s+(scientific\s+)?journal",
]

PROJECT_META = {
    "name": "FakeNewsDetectorAI",
    "version": "2.1.0",
    "author": "Tu nombre / Tu equipo",
    "model_primary": "models/fakenews_model",
    "model_fallback": "mrm8488/bert-tiny-finetuned-fake-news-detection",
    "model_fallback2": "distilbert-base-uncased-finetuned-sst-2-english",
    "languages": ["es", "en"],
    "max_tokens": 512,
    "description": "Detección de fake news y pseudociencia con Transformers + NLP heurístico",
}