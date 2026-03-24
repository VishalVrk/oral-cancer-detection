import gradio as gr
import torch
from transformers import pipeline

# Load model at startup (do this once)
pipe = pipeline(
    "image-classification",
    model="momogueye7/oral_cancer_detection",
    device="cuda" if torch.cuda.is_available() else "cpu"  # optional but better
)

def detect(image):
    if image is None:
        return {"error": "No image provided"}
    
    results = pipe(image)
    
    # Get top prediction
    top = max(results, key=lambda x: x["score"])
    
    is_cancer = "SCC" in top["label"].upper()
    
    return {
        "is_cancer": is_cancer,
        "label": top["label"],
        "confidence": round(top["score"] * 100, 1),
        "all_results": [
            {"label": r["label"], "score": round(r["score"] * 100, 1)}
            for r in results
        ]
    }

# Use Blocks (more reliable than gr.Interface with JSON output)
with gr.Blocks(title="Oral Cancer Detection") as demo:
    gr.Markdown("""
    # 🦷 Oral Cancer Detection
    Upload a **histopathological image** to detect Normal tissue vs **Oral Squamous Cell Carcinoma (OSCC)**.
    """)
    
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(
                type="pil",
                label="Upload Histopathological Image",
                height=400
            )
            btn = gr.Button("🔍 Detect Cancer", variant="primary", size="large")
        
        with gr.Column():
            output_json = gr.JSON(label="Detection Result")
    
    # Connect button
    btn.click(
        fn=detect,
        inputs=input_image,
        outputs=output_json
    )

# Launch configuration optimized for HF Spaces / Docker
demo.launch(
    server_name="0.0.0.0",
    server_port=7860,
    share=False,           # Usually False on Spaces
    show_error=True,
    debug=False            # Set True only for testing
)