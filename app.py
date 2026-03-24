import gradio as gr
from transformers import pipeline

binary_clf = pipeline("image-classification", model="momogueye7/oral_cancer_detection")

def detect(image):
    if image is None:
        return {"error": "No image provided"}
    results = binary_clf(image)
    top = max(results, key=lambda x: x["score"])
    is_cancer = "SCC" in top["label"].upper()
    return {
        "is_cancer": is_cancer,
        "label": top["label"],
        "confidence": round(top["score"] * 100, 1),
        "all_results": [{"label": r["label"], "score": round(r["score"] * 100, 1)} for r in results]
    }

demo = gr.Interface(
    fn=detect,
    inputs=gr.Image(type="pil", label="Upload Histopathological Image"),
    outputs=gr.JSON(label="Detection Result"),
    title="Oral Cancer Detection API",
    description="Upload a histopathological image to detect Normal tissue vs Oral Squamous Cell Carcinoma (OSCC).",
)

demo.launch()
