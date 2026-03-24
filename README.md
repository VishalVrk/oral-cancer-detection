---
title: Oral Cancer Detection and Staging
emoji: 🔬
colorFrom: blue
colorTo: red
sdk: gradio
sdk_version: "4.44.0"
app_file: app.py
pinned: false
license: mit
tags:
  - oral-cancer
  - histopathology
  - image-classification
  - OSCC
---

# Oral Cancer Detection & TNM Staging

An AI-powered web tool for **binary classification** of oral histopathological images
(Normal vs OSCC) followed by **clinical TNM-based staging** of oral cavity cancer.

## How to Use

1. Upload a **histopathological image** (H&E stained oral tissue slide)
2. Click **Analyze Image** — the model detects Normal tissue or Oral Squamous Cell Carcinoma (OSCC)
3. If cancer is detected, the **TNM Staging** panel appears
4. Select T, N, M values based on clinical examination / imaging / biopsy
5. Click **Compute Stage** to get the AJCC stage with explanation

## Model

| Component | Details |
|-----------|---------|
| Architecture | ConvNextV2 (87.7M parameters) via HuggingFace Transformers |
| Base model | [momogueye7/oral_cancer_detection](https://huggingface.co/momogueye7/oral_cancer_detection) |
| Labels | `oral_normal` (healthy tissue) vs `oral_scc` (Oral Squamous Cell Carcinoma) |
| Staging logic | Rule-based AJCC 8th edition TNM mapping (no second ML model needed) |

## TNM Staging Reference (Oral Cavity Cancer)

### T — Tumor
| Code | Description |
|------|-------------|
| T0 | No evidence of primary tumor |
| Tis | Carcinoma in situ |
| T1 | Tumor ≤2 cm |
| T2 | Tumor >2 cm and ≤4 cm |
| T3 | Tumor >4 cm |
| T4a | Moderately advanced — invades adjacent structures (resectable) |
| T4b | Very advanced — invades masticator space / skull base (unresectable) |

### N — Regional Lymph Nodes
| Code | Description |
|------|-------------|
| N0 | No regional lymph node metastasis |
| N1 | Single ipsilateral node ≤3 cm, no extranodal extension |
| N2 | 3–6 cm, or multiple / bilateral / contralateral nodes |
| N3 | Node >6 cm or extranodal extension |

### M — Distant Metastasis
| Code | Description |
|------|-------------|
| M0 | No distant metastasis |
| M1 | Distant metastasis present |

### Stage Groups
| Stage | TNM Criteria |
|-------|-------------|
| 0 | T0/Tis, N0, M0 |
| I | T1, N0, M0 |
| II | T2, N0, M0 |
| III | T3 N0 M0, or T1-T3 N1 M0 |
| IVA | T4a any N0-N1 M0, or T1-T4a N2 M0 |
| IVB | Any T N3 M0, or T4b any N M0 |
| IVC | Any T, Any N, M1 |

## Tech Stack

- **ML**: HuggingFace Transformers + EfficientNet-B0
- **UI**: Gradio
- **Hosting**: HuggingFace Spaces (free)
- **Training**: Google Colab (free T4 GPU)

## Disclaimer

This tool is for **educational purposes only** and must **not** be used for clinical diagnosis.
All staging decisions must be made by a qualified medical professional.
