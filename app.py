import gradio as gr
from transformers import pipeline

# ── Binary classifier ─────────────────────────────────────────────────────────
# Model: ConvNextV2 fine-tuned on oral histopathology (oral_normal vs oral_scc)
# Labels: "oral_normal" = healthy tissue, "oral_scc" = Oral Squamous Cell Carcinoma
MODEL_ID = "momogueye7/oral_cancer_detection"

binary_clf = None

def get_classifier():
    global binary_clf
    if binary_clf is None:
        binary_clf = pipeline("image-classification", model=MODEL_ID)
    return binary_clf

# ── TNM staging rule engine (pure Python, no ML) ──────────────────────────────
def compute_stage(t: str, n: str, m: str) -> str:
    """Map T/N/M codes to AJCC clinical stage per oral cavity cancer guidelines."""
    if m == "M1":
        return "Stage IVC\nDistant metastasis present. Any T, Any N, M1."
    if t == "T4b":
        return "Stage IVB\nVery advanced unresectable local disease. T4b, Any N, M0."
    if n == "N3":
        return "Stage IVB\nExtensive nodal involvement (>6 cm or extranodal extension). Any T, N3, M0."
    if n == "N2" or t == "T4a":
        return "Stage IVA\nModerately advanced local/regional disease. T4a (any N0-N1) or T1-T4a with N2, M0."
    if n == "N1":
        return "Stage III\nT1-T3 with single ipsilateral node ≤3 cm (N1), M0."
    if t == "T3":
        return "Stage III\nTumor >4 cm (T3), no nodal involvement (N0), M0."
    if t == "T2":
        return "Stage II\nTumor >2 cm and ≤4 cm (T2), N0, M0."
    if t == "T1":
        return "Stage I\nTumor ≤2 cm (T1), N0, M0."
    return "Stage 0\nCarcinoma in situ or no primary tumor (T0/Tis), N0, M0."


def detect(image):
    """Run binary classification; show staging panel only if cancer detected."""
    if image is None:
        return "Please upload an image.", gr.update(visible=False)

    results = get_classifier()(image)
    top = max(results, key=lambda x: x["score"])
    label = top["label"].upper()
    score = top["score"]

    # Model labels: "oral_normal" (healthy) or "oral_scc" (squamous cell carcinoma)
    is_cancer = "SCC" in label

    if is_cancer:
        msg = (
            f"CANCER DETECTED  (confidence: {score:.1%})\n\n"
            "Oral Squamous Cell Carcinoma identified in the histopathological image.\n"
            "Please enter clinical T, N, M values below to compute the AJCC stage."
        )
    else:
        msg = (
            f"NORMAL TISSUE  (confidence: {score:.1%})\n\n"
            "No malignancy detected in this histopathological image."
        )

    return msg, gr.update(visible=is_cancer)


def stage(t_val: str, n_val: str, m_val: str) -> str:
    """Extract T/N/M codes from dropdown labels and compute stage."""
    t = t_val.split(" —")[0].strip()
    n = n_val.split(" —")[0].strip()
    m = m_val.split(" —")[0].strip()
    result = compute_stage(t, n, m)
    return (
        f"TNM Classification:  {t},  {n},  {m}\n\n"
        f"{result}\n\n"
        "⚠  This is an educational tool. Staging must be confirmed by a qualified clinician."
    )


# ── Dropdown options ───────────────────────────────────────────────────────────
T_CHOICES = [
    "T0 — No evidence of primary tumor",
    "Tis — Carcinoma in situ",
    "T1 — Tumor ≤2 cm",
    "T2 — Tumor >2 cm and ≤4 cm",
    "T3 — Tumor >4 cm",
    "T4a — Moderately advanced (resectable, invades jaw/deep muscles/skin)",
    "T4b — Very advanced (unresectable, invades masticator space/skull base/carotid)",
]
N_CHOICES = [
    "N0 — No regional lymph node metastasis",
    "N1 — Single ipsilateral node ≤3 cm, no extranodal extension",
    "N2 — 3–6 cm, or multiple/bilateral/contralateral nodes",
    "N3 — Node >6 cm or extranodal extension",
]
M_CHOICES = [
    "M0 — No distant metastasis",
    "M1 — Distant metastasis present",
]

# ── Gradio UI ─────────────────────────────────────────────────────────────────
with gr.Blocks(title="Oral Cancer Detection & Staging", theme=gr.themes.Soft()) as demo:

    gr.Markdown("""
    # Oral Cancer Detection & TNM Staging
    Upload a **histopathological image** (H&E stained oral tissue) for binary cancer detection.
    If cancer is identified, enter clinical TNM values to compute the **AJCC stage**.
    """)

    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(type="pil", label="Histopathological Image")
            analyze_btn = gr.Button("Analyze Image", variant="primary", size="lg")
        with gr.Column(scale=1):
            detection_output = gr.Textbox(
                label="Detection Result", lines=6, interactive=False
            )

    with gr.Group(visible=False) as staging_panel:
        gr.Markdown("---\n### TNM Staging — Enter Clinical Values")
        gr.Markdown(
            "Based on **clinical examination, imaging (CT/MRI), and biopsy** findings:"
        )
        with gr.Row():
            t_input = gr.Dropdown(
                choices=T_CHOICES, value=T_CHOICES[2], label="T — Tumor Size / Extent"
            )
            n_input = gr.Dropdown(
                choices=N_CHOICES, value=N_CHOICES[0], label="N — Regional Lymph Nodes"
            )
            m_input = gr.Dropdown(
                choices=M_CHOICES, value=M_CHOICES[0], label="M — Distant Metastasis"
            )
        stage_btn = gr.Button("Compute Stage", variant="secondary")
        stage_output = gr.Textbox(
            label="Stage Classification Result", lines=5, interactive=False
        )

    gr.Markdown(
        "---\n"
        "*Disclaimer: This tool is for **educational purposes only** and must not be used for clinical diagnosis.*"
    )

    # ── Event wiring ───────────────────────────────────────────────────────────
    analyze_btn.click(
        fn=detect,
        inputs=image_input,
        outputs=[detection_output, staging_panel],
        api_name=False,
    )
    stage_btn.click(
        fn=stage,
        inputs=[t_input, n_input, m_input],
        outputs=stage_output,
    )

demo.launch(server_name="0.0.0.0")
