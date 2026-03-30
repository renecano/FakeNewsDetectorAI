# =============================================================
#  FakeNewsDetectorAI — main.py  v3.1
#  Layout corregido + justificaciones en español extendidas
# =============================================================

import gradio as gr
from predictor import get_detector, PredictionResult
from labels import LABELS, PROJECT_META
from preprocess import translate_signal

CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=DM+Mono:wght@400;500&family=Lora:ital,wght@0,400;0,600;1,400&display=swap');

*, *::before, *::after { box-sizing: border-box; }

:root {
    --cream:    #F5F0E8;
    --ink:      #1A1410;
    --red:      #C0392B;
    --amber:    #D4870A;
    --green:    #1E7D45;
    --paper:    #EDE8DC;
    --rule:     #C8BFA8;
    --muted:    #7A6E60;
    --real-bg:  #EAF5EE;
    --doubt-bg: #FEF8EC;
    --fake-bg:  #FDECEA;
    --real-bdr: #A8D5B5;
    --doubt-bdr:#F5D98A;
    --fake-bdr: #F0AAAA;
}

body, .gradio-container {
    background: var(--cream) !important;
    font-family: 'Lora', Georgia, serif !important;
    color: var(--ink) !important;
}

/* ── Masthead ─────────────────────────────────────────────── */
.fn-masthead {
    border-bottom: 3px double var(--ink);
    padding: 1.8rem 0 1.2rem;
    text-align: center;
    background: var(--cream);
    max-width: 900px;
    margin: 0 auto;
}
.fn-masthead::before {
    content: '';
    display: block;
    border-top: 1px solid var(--ink);
    border-bottom: 1px solid var(--ink);
    height: 4px;
    margin-bottom: 1.2rem;
}
.fn-eyebrow {
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.25em;
    color: var(--muted);
    text-transform: uppercase;
    margin-bottom: 0.6rem;
}
.fn-title {
    font-family: 'Syne', sans-serif;
    font-size: clamp(2.8rem, 6vw, 4.8rem);
    font-weight: 800;
    letter-spacing: -3px;
    line-height: 0.95;
    color: var(--ink);
}
.fn-title-accent { color: var(--red); }
.fn-subtitle {
    font-style: italic;
    font-size: 1rem;
    color: var(--muted);
    margin-top: 0.6rem;
}
.fn-tagline {
    display: flex;
    align-items: center;
    gap: 1rem;
    margin-top: 0.8rem;
    color: var(--muted);
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.1em;
    max-width: 700px;
    margin-left: auto;
    margin-right: auto;
}
.fn-tagline::before, .fn-tagline::after {
    content: '';
    flex: 1;
    height: 1px;
    background: var(--rule);
}

/* ── Wrapper global ───────────────────────────────────────── */
.fn-wrapper {
    max-width: 900px;
    margin: 0 auto;
    padding: 2rem 1.5rem;
}

/* ── Labels de columna ────────────────────────────────────── */
.fn-col-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: var(--muted);
    border-bottom: 1px solid var(--rule);
    padding-bottom: 0.5rem;
    margin-bottom: 1rem;
}

/* ── Textarea ─────────────────────────────────────────────── */
label span, .gr-textbox label span {
    font-family: 'DM Mono', monospace !important;
    font-size: 0.68rem !important;
    letter-spacing: 0.15em !important;
    text-transform: uppercase !important;
    color: var(--muted) !important;
}
textarea {
    background: white !important;
    border: 1px solid var(--rule) !important;
    border-radius: 0 !important;
    color: var(--ink) !important;
    font-family: 'Lora', serif !important;
    font-size: 0.97rem !important;
    line-height: 1.75 !important;
    padding: 1rem !important;
    box-shadow: inset 0 1px 4px rgba(0,0,0,0.04) !important;
    transition: border-color 0.2s !important;
}
textarea:focus {
    border-color: var(--ink) !important;
    outline: none !important;
}

/* ── Buttons ──────────────────────────────────────────────── */
button {
    border-radius: 0 !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: 0.82rem !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    transition: all 0.15s !important;
    cursor: pointer !important;
}
button.primary {
    background: var(--ink) !important;
    color: var(--cream) !important;
    border: 2px solid var(--ink) !important;
    padding: 0.7rem 1.8rem !important;
}
button.primary:hover {
    background: var(--red) !important;
    border-color: var(--red) !important;
}
button.secondary {
    background: transparent !important;
    color: var(--muted) !important;
    border: 1px solid var(--rule) !important;
    padding: 0.7rem 1.2rem !important;
}
button.secondary:hover {
    color: var(--ink) !important;
    border-color: var(--ink) !important;
}

/* ── Separador entre secciones ────────────────────────────── */
.fn-divider {
    max-width: 900px;
    margin: 0 auto;
    border: none;
    border-top: 3px double var(--ink);
}

/* ── Examples ─────────────────────────────────────────────── */
.fn-examples-wrap {
    max-width: 900px;
    margin: 0 auto;
    padding: 1.5rem 1.5rem 2rem;
}
.fn-examples-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 1rem;
}

/* ── Footer ───────────────────────────────────────────────── */
.fn-footer {
    border-top: 1px solid var(--rule);
    padding: 0.8rem 1.5rem;
    max-width: 900px;
    margin: 0 auto;
    display: flex;
    justify-content: space-between;
    font-family: 'DM Mono', monospace;
    font-size: 0.62rem;
    color: var(--muted);
    letter-spacing: 0.05em;
}

/* ── Hide gradio noise ────────────────────────────────────── */
footer, .gr-prose { display: none !important; }
.gradio-container { padding: 0 !important; max-width: 100% !important; }
"""

# ── Justificaciones extendidas en español ─────────────────────
JUSTIFICATIONS = {
    "FAKE": {
        "pseudoscience": (
            "El texto utiliza vocabulario técnico y científico de manera superficial para aparentar "
            "credibilidad, pero carece de los elementos fundamentales del periodismo verificable: "
            "no cita ninguna revista científica reconocida, no nombra a los investigadores ni la "
            "institución responsable, y hace afirmaciones absolutas (como protección del 100% o "
            "eliminación completa) que contradicen el funcionamiento real de la investigación "
            "científica, donde los resultados siempre tienen márgenes de error y limitaciones."
        ),
        "alarmist": (
            "El texto emplea técnicas clásicas de desinformación: lenguaje alarmista en mayúsculas, "
            "llamados urgentes a compartir el contenido, y afirmaciones de que la información está "
            "siendo suprimida por autoridades o industrias. Este tipo de redacción está diseñada "
            "para generar una respuesta emocional inmediata que evite el análisis crítico del lector, "
            "y es característica de la desinformación viral en redes sociales."
        ),
        "general": (
            "El texto presenta múltiples indicadores de desinformación: ausencia de fuentes "
            "verificables, afirmaciones extraordinarias sin respaldo de instituciones reconocidas, "
            "y un estilo de redacción orientado a provocar reacción emocional en lugar de informar. "
            "Antes de compartir este contenido, se recomienda buscar cobertura del mismo tema en "
            "medios de comunicación establecidos o consultar bases de datos científicas como PubMed."
        ),
    },
    "DOUBTFUL": (
        "El texto mezcla elementos verificables con afirmaciones que no pueden confirmarse "
        "con las fuentes disponibles. Aunque parte de la información puede ser correcta, "
        "la ausencia de fuentes primarias claras, el uso de lenguaje vago como 'se dice que' "
        "o 'algunos expertos afirman', y la falta de contexto suficiente hacen imposible "
        "validar el contenido completo. Se recomienda contrastar esta información con al menos "
        "dos fuentes independientes y verificadas antes de tomar decisiones basadas en ella."
    ),
    "REAL": (
        "El texto presenta características propias del periodismo verificable: cita fuentes "
        "identificables, utiliza lenguaje neutral y preciso, y sus afirmaciones están respaldadas "
        "por declaraciones de instituciones reconocidas o investigaciones con metodología descrita. "
        "El uso de expresiones como 'según datos oficiales', 'confirmó' o referencias a "
        "publicaciones científicas con nombre propio son indicadores de contenido que puede "
        "ser rastreado y verificado de manera independiente."
    ),
}

def get_justification(result: PredictionResult) -> str:
    """Selecciona la justificación más apropiada según el tipo de noticia."""
    if result.label == "FAKE":
        pseudo = result.features.get("pseudoscience_hits", [])
        fake_h = result.features.get("fake_signal_hits", [])
        if len(pseudo) >= 1:
            return JUSTIFICATIONS["FAKE"]["pseudoscience"]
        elif len(fake_h) >= 2:
            return JUSTIFICATIONS["FAKE"]["alarmist"]
        else:
            return JUSTIFICATIONS["FAKE"]["general"]
    elif result.label == "DOUBTFUL":
        return JUSTIFICATIONS["DOUBTFUL"]
    else:
        return JUSTIFICATIONS["REAL"]


def build_result_html(result: PredictionResult) -> str:
    if result.error:
        return f"""
        <div style="padding:1.5rem;border:1px solid #C8BFA8;background:white">
            <p style="font-family:'DM Mono',monospace;font-size:.78rem;
                      color:#C0392B;letter-spacing:.05em">{result.error}</p>
        </div>"""

    color_map  = {"REAL": "#1E7D45", "DOUBTFUL": "#D4870A", "FAKE": "#C0392B"}
    bg_map     = {"REAL": "#EAF5EE", "DOUBTFUL": "#FEF8EC", "FAKE": "#FDECEA"}
    border_map = {"REAL": "#A8D5B5", "DOUBTFUL": "#F5D98A", "FAKE": "#F0AAAA"}
    lc  = color_map[result.label]
    lbg = bg_map[result.label]
    lbd = border_map[result.label]
    scores = result.scores

    def bar(k, label_es):
        pct = scores.get(k, 0) * 100
        clr = color_map[k]
        return f"""
        <div style="margin:.45rem 0">
            <div style="display:flex;justify-content:space-between;
                        font-family:'DM Mono',monospace;font-size:.7rem;
                        color:#7A6E60;margin-bottom:3px">
                <span>{label_es}</span>
                <span style="color:{clr};font-weight:500">{pct:.1f}%</span>
            </div>
            <div style="height:4px;background:#E8E0D0;overflow:hidden">
                <div style="height:100%;width:{pct:.1f}%;background:{clr}"></div>
            </div>
        </div>"""

    # Señales en español
    pseudo = result.features.get("pseudoscience_hits", [])
    fake_h = result.features.get("fake_signal_hits",   [])
    real_h = result.features.get("real_signal_hits",   [])
    doubt_h= result.features.get("doubtful_signal_hits",[])

    signals = []
    if pseudo:
        signals.append(
            f'<p style="margin:4px 0;font-size:.85rem;color:#4A4035;line-height:1.5">'
            f'🔬 <strong>Patrones pseudocientíficos ({len(pseudo)}):</strong> '
            f'afirmaciones absolutas o estudios sin publicar detectados</p>'
        )
    if fake_h:
        translated = [translate_signal(h) for h in fake_h[:3]]
        signals.append(
            f'<p style="margin:4px 0;font-size:.85rem;color:#4A4035;line-height:1.5">'
            f'🔴 <strong>Lenguaje alarmista:</strong> {", ".join(translated)}</p>'
        )
    if real_h:
        translated = [translate_signal(h) for h in real_h[:3]]
        signals.append(
            f'<p style="margin:4px 0;font-size:.85rem;color:#4A4035;line-height:1.5">'
            f'🟢 <strong>Fuentes verificables:</strong> {", ".join(translated)}</p>'
        )
    if doubt_h:
        translated = [translate_signal(h) for h in doubt_h[:2]]
        signals.append(
            f'<p style="margin:4px 0;font-size:.85rem;color:#4A4035;line-height:1.5">'
            f'🟡 <strong>Lenguaje no confirmado:</strong> {", ".join(translated)}</p>'
        )

    signals_block = ""
    if signals:
        signals_block = f"""
        <div style="padding:.9rem 1rem;background:white;
                    border:1px solid #E0D8C8;margin-bottom:1.2rem">
            <div style="font-family:'DM Mono',monospace;font-size:.62rem;
                        letter-spacing:.15em;text-transform:uppercase;
                        color:#7A6E60;margin-bottom:.6rem">Indicadores detectados</div>
            {''.join(signals)}
        </div>"""

    warning_block = ""
    if result.warning:
        warning_block = f"""
        <p style="font-family:'DM Mono',monospace;font-size:.7rem;color:#D4870A;
                  margin-top:.8rem;padding:.6rem;border:1px solid #F5D98A;
                  background:#FEF8EC">{result.warning}</p>"""

    # Nombre del modelo sin ruta completa
    model_name = result.model_used
    if "\\" in model_name or "/" in model_name:
        import os
        model_name = os.path.basename(model_name.rstrip("/\\"))

    justification = get_justification(result)

    return f"""
    <div style="font-family:'Lora',serif;color:#1A1410">

        <!-- Veredicto -->
        <div style="background:{lbg};border:1px solid {lbd};border-left:5px solid {lc};
                    padding:1.2rem 1.4rem;margin-bottom:1.5rem">
            <div style="font-family:'DM Mono',monospace;font-size:.62rem;
                        letter-spacing:.2em;text-transform:uppercase;
                        color:#7A6E60;margin-bottom:.35rem">Veredicto del sistema</div>
            <div style="font-family:'Syne',sans-serif;font-size:1.9rem;
                        font-weight:800;color:{lc};letter-spacing:-1px;
                        line-height:1">{LABELS[result.label].display_es}</div>
            <div style="font-family:'DM Mono',monospace;font-size:.75rem;
                        color:#7A6E60;margin-top:.35rem">
                Confianza: <strong style="color:{lc}">{result.confidence_pct}</strong>
            </div>
        </div>

        <!-- Justificación extendida -->
        <div style="margin-bottom:1.5rem;padding-bottom:1.5rem;
                    border-bottom:1px solid #C8BFA8">
            <div style="font-family:'DM Mono',monospace;font-size:.62rem;
                        letter-spacing:.15em;text-transform:uppercase;
                        color:#7A6E60;margin-bottom:.6rem">Análisis del contenido</div>
            <p style="font-style:italic;font-size:.93rem;line-height:1.85;color:#3A3028">
                {justification}
            </p>
        </div>

        <!-- Distribución -->
        <div style="margin-bottom:1.4rem">
            <div style="font-family:'DM Mono',monospace;font-size:.62rem;
                        letter-spacing:.15em;text-transform:uppercase;
                        color:#7A6E60;margin-bottom:.7rem">Distribución de probabilidad</div>
            {bar("REAL",     "Confiable")}
            {bar("DOUBTFUL", "Dudosa")}
            {bar("FAKE",     "Falsa")}
        </div>

        {signals_block}
        {warning_block}

        <!-- Meta -->
        <div style="display:flex;gap:1.5rem;flex-wrap:wrap;
                    padding-top:.9rem;border-top:1px solid #C8BFA8">
            <span style="font-family:'DM Mono',monospace;font-size:.62rem;color:#7A6E60">⚡ {result.inference_ms} ms</span>
            <span style="font-family:'DM Mono',monospace;font-size:.62rem;color:#7A6E60">🤖 {model_name}</span>
            <span style="font-family:'DM Mono',monospace;font-size:.62rem;color:#7A6E60">📝 {result.features.get('word_count','?')} palabras</span>
        </div>
    </div>"""


def analyze(text: str):
    return build_result_html(get_detector().predict(text))


EXAMPLES = [
    ["Scientists Confirm That Drinking Coffee Daily Completely Prevents Cancer. A group of researchers from a European university has reportedly discovered that drinking at least three cups of coffee per day can completely prevent cancer. The study claims 100% protection rate against all forms of cancer. The full study has not yet been published in any recognized scientific journal."],
    ["¡¡URGENTE!! ¡¡LOS MÉDICOS NO QUIEREN QUE SEPAS ESTO!! Cura el cáncer con jugo de limón. COMPARTE ANTES DE QUE LO BORREN. El gobierno está ocultando la verdad sobre esta cura milagrosa."],
    ["La Reserva Federal elevó las tasas de interés en 0.25 puntos porcentuales el miércoles, según un comunicado oficial emitido por el banco central. La decisión fue unánime entre los miembros votantes del Comité Federal de Mercado Abierto."],
    ["A peer-reviewed study published in the New England Journal of Medicine found a moderate association between daily coffee consumption and reduced risk of certain liver conditions. Researchers noted that more studies are needed to establish causation."],
]

with gr.Blocks(css=CUSTOM_CSS, title="FakeNews Detector AI", theme=gr.themes.Base()) as demo:

    gr.HTML(f"""
    <div style="max-width:900px;margin:0 auto;padding:1.8rem 1.5rem 0">
        <div class="fn-masthead">
            <div class="fn-eyebrow">Inteligencia Artificial · NLP · Transformers · v{PROJECT_META['version']}</div>
            <div class="fn-title">Fake<span class="fn-title-accent">News</span><br>Detector</div>
            <div class="fn-subtitle">Sistema de verificación de noticias basado en Inteligencia Artificial</div>
            <div class="fn-tagline">
                <span>RoBERTa · Heurísticas NLP · 99.9% Accuracy · Detección de pseudociencia</span>
            </div>
        </div>
    </div>
    """)

    with gr.Row():
        with gr.Column(scale=1):
            gr.HTML("""<div style="max-width:900px;margin:2rem auto 0;padding:0 1.5rem">
                <div class="fn-col-label">① Ingresa el texto a analizar</div>
            </div>""")
            input_text = gr.Textbox(
                label="Texto de la noticia",
                placeholder="Pega aquí el titular y cuerpo de la noticia que deseas verificar…",
                lines=11,
                max_lines=25,
            )
            with gr.Row():
                btn_analyze = gr.Button("Analizar →", variant="primary")
                btn_clear   = gr.Button("Limpiar", variant="secondary")

    with gr.Row():
        with gr.Column(scale=1):
            gr.HTML("""<div style="max-width:900px;margin:1.5rem auto 0;padding:0 1.5rem">
                <div class="fn-col-label">② Resultado del análisis</div>
            </div>""")
            output_html = gr.HTML(
                value="""<div style="max-width:900px;margin:0 auto;padding:0 1.5rem">
                    <div style="padding:2.5rem;border:1px dashed #C8BFA8;
                                text-align:center;background:white">
                        <div style="font-family:'DM Mono',monospace;font-size:.72rem;
                                    color:#C8BFA8;letter-spacing:.15em">
                            INGRESA UN TEXTO Y PRESIONA ANALIZAR
                        </div>
                    </div>
                </div>"""
            )

    gr.HTML('<hr class="fn-divider" style="max-width:900px;margin:1.5rem auto 0;border:none;border-top:3px double #1A1410"/>')

    gr.HTML('<div class="fn-examples-wrap"><div class="fn-examples-label">Ejemplos de prueba — haz clic para cargar</div></div>')

    gr.Examples(
        examples=EXAMPLES,
        inputs=input_text,
        outputs=output_html,
        fn=analyze,
        cache_examples=False,
    )

    gr.HTML(f"""
    <div class="fn-footer">
        <span>{PROJECT_META['name']} · v{PROJECT_META['version']}</span>
        <span>Modelo: fakenews_model (RoBERTa)</span>
        <span>Hackathon Demo · 2025</span>
    </div>
    """)

    btn_analyze.click(fn=analyze, inputs=input_text, outputs=output_html)
    btn_clear.click(fn=lambda: ("", ""), outputs=[input_text, output_html])
    input_text.submit(fn=analyze, inputs=input_text, outputs=output_html)


if __name__ == "__main__":
    print(f"\n{'='*55}")
    print(f"  FakeNewsDetectorAI v{PROJECT_META['version']}")
    print(f"  http://localhost:7860")
    print(f"{'='*55}\n")
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False, show_error=True)