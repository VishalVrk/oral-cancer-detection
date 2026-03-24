import os
import tempfile
from flask import Flask, request, jsonify, render_template
from gradio_client import Client, handle_file

app = Flask(__name__)

# ── HuggingFace Space used as the ML inference backend ───────────────────────
HF_SPACE = "vrkforever/oral-cancer-detection"
_hf_client = None

def get_client():
    """Lazy-init the Gradio client (reuses the connection)."""
    global _hf_client
    if _hf_client is None:
        _hf_client = Client(HF_SPACE)
    return _hf_client


# ── TNM staging rule engine (pure Python, no API call needed) ─────────────────
def compute_stage(t: str, n: str, m: str) -> dict:
    if m == "M1":
        return {"stage": "Stage IVC", "description": "Distant metastasis present. Any T, Any N, M1.", "color": "red"}
    if t == "T4b":
        return {"stage": "Stage IVB", "description": "Very advanced unresectable local disease. T4b, Any N, M0.", "color": "red"}
    if n == "N3":
        return {"stage": "Stage IVB", "description": "Extensive nodal involvement (>6 cm or extranodal extension). Any T, N3, M0.", "color": "red"}
    if n == "N2" or t == "T4a":
        return {"stage": "Stage IVA", "description": "Moderately advanced local/regional disease. T4a or N2, M0.", "color": "orange"}
    if n == "N1":
        return {"stage": "Stage III", "description": "T1–T3 with single ipsilateral node ≤3 cm (N1), M0.", "color": "orange"}
    if t == "T3":
        return {"stage": "Stage III", "description": "Tumor >4 cm (T3), no nodal involvement (N0), M0.", "color": "orange"}
    if t == "T2":
        return {"stage": "Stage II", "description": "Tumor >2 cm and ≤4 cm (T2), N0, M0.", "color": "yellow"}
    if t == "T1":
        return {"stage": "Stage I", "description": "Tumor ≤2 cm (T1), N0, M0.", "color": "yellow"}
    return {"stage": "Stage 0", "description": "Carcinoma in situ or no primary tumor (T0/Tis), N0, M0.", "color": "green"}


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/detect", methods=["POST"])
def detect():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]

    # Save to a temp file so gradio_client can upload it to the HF Space
    suffix = os.path.splitext(file.filename)[1] or ".jpg"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        file.save(tmp.name)
        tmp_path = tmp.name

    try:
        client = get_client()
        result = client.predict(handle_file(tmp_path), api_name="/predict")
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        os.unlink(tmp_path)


@app.route("/stage", methods=["POST"])
def stage():
    data = request.get_json()
    t = data.get("t", "T1")
    n = data.get("n", "N0")
    m = data.get("m", "M0")
    result = compute_stage(t, n, m)
    result["tnm"] = f"{t}, {n}, {m}"
    return jsonify(result)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
