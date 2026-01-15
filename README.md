# MindScope: Multimodal Clinical Risk Stratification

![Build Status](https://img.shields.io/badge/build-passing-brightgreen) ![Python](https://img.shields.io/badge/python-3.9%2B-blue) ![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c) ![License](https://img.shields.io/badge/license-MIT-green)

## Overview
MindScope is a multimodal deep learning prototype that detects early signals of mental health risk by fusing structured patient history with unstructured clinical notes. It uses a late-fusion architecture: a transformer-based text encoder runs in parallel with a tabular MLP, and both representations are concatenated for final classification.

> This repo generates synthetic “gold standard” data to avoid using sensitive patient records.

## Architecture
- **NLP Tower:** DistilBERT → 768-d CLS embedding
- **Tabular Tower:** Normalized features → MLP + BatchNorm → 32-d embedding
- **Fusion Head:** concat([V_text, V_tab]) → MLP → 3-class logits (low/med/high)

## Key Metrics (Synthetic Data)
- **Validation Accuracy:** ~0.958
- **Validation F1 (Weighted):** ~0.955

> Metrics are reported on synthetic data generated with a fixed seed. Your numbers may vary slightly by environment.

## Quick Start

### Install
```bash
pip install -r requirements.txt
Generate Synthetic Data
bash
Copy code
python -m src.data_gen
Train
bash
Copy code
python -m src.train
Best checkpoint saved to models/mindscope_best.pth.

Engineering Decisions
Late fusion: handles mixed modalities without forcing a single joint feature space too early.

Tabular normalization: prevents scale dominance during fusion.

BatchNorm: stabilizes tabular learning and improves convergence.

Gradient clipping: prevents exploding gradients during fine-tuning.

Device-aware: uses CUDA when available.

Responsible Use
Research prototype only — not a diagnostic tool. Any high-risk flag must be reviewed by a clinician.

Contact
Sanskruti Sanjay Deshmukh — deshm084@umn.edu — https://www.linkedin.com/in/sanskruti-deshmukh233/
