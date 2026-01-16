# MindScope: Multimodal Clinical Risk Stratification

![Build Status](https://img.shields.io/github/actions/workflow/status/deshm084/MindScope_AI/ci_cd.yml?branch=main) ![Python](https://img.shields.io/badge/python-3.9%2B-blue) ![Framework](https://img.shields.io/badge/FastAPI-Production-green) ![License](https://img.shields.io/badge/license-MIT-green)

## 📋 Overview
MindScope is a **multimodal deep learning system** designed to detect early signals of mental health risk by fusing structured patient history with unstructured clinical notes.

While traditional models rely solely on survey checkboxes, MindScope uses a **Late-Fusion Architecture** to analyze the *context* of a patient's words (using DistilBERT) alongside their clinical metrics (TabNet).

### Key Metrics (Synthetic Data)
* **Validation Accuracy:** ~95.8%
* **F1-Score (Weighted):** ~0.955
* **Inference Latency:** <45ms (via ONNX/FastAPI)

---

## 🏗️ Architecture
The system processes data through two parallel "towers" before fusing representations for the final classification:

1.  **NLP Tower:** * **Input:** Unstructured text (e.g., "I feel hopeless and can't sleep.")
    * **Model:** `DistilBERT-base-uncased` (Fine-tuned).
    * **Output:** 768-dim semantic vector.
2.  **Tabular Tower:** * **Input:** Normalized features (Age, Sleep, Stress).
    * **Model:** Dense Network with **Batch Normalization**.
    * **Output:** 32-dim feature vector.
3.  **Fusion Layer:**
    * Concatenates vectors $[V_{nlp}, V_{tab}]$ -> Dense Layer -> Softmax -> Risk Probability.

---

## 🚀 Quick Start

### 1. Installation
```bash
pip install -r requirements.txt
2. Generate Data & TrainTo respect patient privacy, this repository generates synthetic "Gold Standard" data.Bash# Generate synthetic dataset
python -m src.data_gen

# Train the fusion model
python -m src.train
Best model saved to: models/mindscope_best.pth3. Run the Production API (Microservice)Launch the FastAPI server to generate real-time risk assessments.Bashuvicorn api.app:app --reload
Swagger UI: Open http://localhost:8000/docs to test the API interactively.Sample Request:JSON{
  "clinical_note": "Patient reports insomnia and feelings of worthlessness.",
  "age": 45,
  "sleep_hours": 3.5,
  "stress_level": 9,
  "family_history": 1
}
4. Run Unit TestsValidate model architecture and tensor shapes before deployment.Bashpytest tests/
🛠️ Engineering DecisionsDecisionRationaleLate FusionHandles mixed modalities (Text vs. Numbers) without forcing a joint feature space too early, allowing each tower to specialize.DistilBERTRetains 97% of BERT's performance while being 40% smaller—critical for low-latency clinical dashboards.Batch NormalizationStabilizes the learning of tabular features (Age 0-100 vs. Sleep 0-24) so they aren't overpowered by high-dimensional text gradients.Microservice (API)Inference logic is wrapped in FastAPI with Pydantic validation, making the model Docker-ready for Azure/AWS deployment.📂 Project StructurePlaintextMindScope_AI/
├── .github/workflows/   # CI/CD Pipeline (Automated Testing)
├── api/                 # FastAPI Microservice (Production Interface)
├── src/                 # Core ML Logic (Data Loading, Fusion Model, Training)
├── tests/               # Unit Tests (Pytest)
├── models/              # Saved PyTorch Weights
└── requirements.txt     # Dependencies
⚖️ Responsible AIDisclaimer: This tool is a research prototype for risk screening, not automated diagnosis. High-risk flags must always be reviewed by a human clinician.Data Privacy: No real patient data is included in this repository; all data is synthetically generated based on statistical distributions.📧 ContactSanskruti Sanjay Deshmukh LinkedIn | Email
