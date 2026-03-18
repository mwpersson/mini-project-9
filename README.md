# Mini Project 9 — NLP Content Moderation with Transformers

COMP 9130 - Group 6
---

## Problem Description & Motivation

Social media platforms face an overwhelming volume of user-generated content that requires moderation. Manual review is expensive, slow, and psychologically taxing for human reviewers. This project builds a prototype automated content moderation classifier for SafeSpace AI, a startup providing moderation tooling to mid-size social networks and community forums.

The classifier categorizes tweets into three classes:

| Class | Label | Description |
|-------|-------|-------------|
| 0 | Hate Speech | Content targeting individuals or groups based on protected characteristics |
| 1 | Offensive Language | Rude or vulgar content that does not target a protected group |
| 2 | Neither | Acceptable content requiring no moderation action |

The critical challenge is distinguishing hate speech from offensive language. Platforms that fail to remove hate speech risk regulatory penalties, while over-censoring offensive-but-legal speech damages user trust. The project compares a TF-IDF + classical ML baseline against a fine-tuned ALBERT transformer, and designs a production moderation workflow with confidence-based routing.

---

## Dataset

**Twitter Hate Speech and Offensive Language Dataset** — Davidson et al. (2017)

| Detail | Info |
|--------|------|
| Source | [GitHub — t-davidson/hate-speech-and-offensive-language](https://github.com/t-davidson/hate-speech-and-offensive-language) |
| Direct CSV | [labeled_data.csv](https://raw.githubusercontent.com/t-davidson/hate-speech-and-offensive-language/master/data/labeled_data.csv) |
| Size | 24,783 tweets |
| Annotation | CrowdFlower crowd workers, majority-vote label |
| Class split | ~5.77% hate speech · ~77.43% offensive · ~16.80% neither |
| Paper | [Davidson et al., ICWSM 2017](https://arxiv.org/abs/1703.04009) |

> **Note:** This dataset contains offensive and hateful language. It is used professionally for the purpose of building harm-reduction tooling.

Data is small enough to be downloaded to the ```/data``` folder for convenience.

---

## Repository Structure

```
mini-project-9/
├── README.md
├── requirements.txt
├── figures/                       # Figures from notebooks
├── data/
│   └── labeled_data.csv           # Download before running
└── notebooks/
    ├── 00_full.ipynb              # Primary run: exploration, baseline, transformer (oversampling)
    └── 00_fullv2.ipynb            # Second run: weighted loss + balanced baselines + confidence analysis
```

---

## Setup & How to Run

### Requirements

- Python 3.9+
- A CUDA-capable GPU is strongly recommended for transformer fine-tuning. Training on CPU is possible but will be significantly slower (~hours per epoch vs. minutes on GPU).
- See `requirements.txt` for all dependencies.

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/mwpersson/mini-project-9
cd mini-project-9

# 2. Create and activate a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate        # macOS/Linux
.venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt

# GPU (CUDA 13.0) — installs PyTorch with CUDA support
pip install torch --index-url https://download.pytorch.org/whl/cu130

# CPU only — installs standard PyTorch
pip install torch
```

### Running the Notebooks

**Run the primary notebook — `00_full.ipynb`:**

This single notebook covers the full pipeline end-to-end:
1. Data loading, exploration, and class distribution visualization
2. Text preprocessing (URL/mention/hashtag removal, lowercasing)
3. Train/val/test split (70/15/15, stratified)
4. TF-IDF baseline — SVM, Logistic Regression, Random Forest (no class balancing)
5. ALBERT-base-v2 fine-tuning with random oversampling (3 epochs, lr=1e-5)
6. Evaluation: per-class metrics, confusion matrices, misclassification analysis

```bash
jupyter notebook notebooks/00_full.ipynb
# or
jupyter lab notebooks/00_full.ipynb
```

Run all cells top to bottom. The notebook will automatically download the ALBERT model weights from Hugging Face on first run (~50 MB).

> **Note:** A second notebook `00_fullv2.ipynb` is also included. It extends the primary run with class-balanced baselines (`class_weight='balanced'`), a weighted CrossEntropyLoss transformer, and confidence/coverage threshold analysis. It is not required to run `00_full.ipynb` first — both notebooks are self-contained.

---

## Results Summary

### Baseline vs. Transformer Comparison

All results on the held-out test set (3,718 samples, stratified split).

| Model | Accuracy | Macro F1 | Hate F1 | Hate Recall | Offensive F1 | Neither F1 |
|-------|----------|----------|---------|-------------|--------------|------------|
| SVM Baseline (no balancing) | 89.32% | 0.7066 | 0.3551 | 26.6% | 0.9370 | 0.8277 |
| SVM Baseline (balanced weights) | 89.59% | 0.7401 | 0.4239 | 39.7% | 0.9379 | 0.8584 |
| **ALBERT — Oversampling (v1)** | **89.75%** | **0.7304** | **0.3735** | **35.5%** | **0.9383** | **0.8796** |
| ALBERT — Weighted Loss (v2) | 85.88% | 0.7398 | 0.4223 | 74.3% | 0.9138 | 0.8833 |

**Key findings:**
- The ALBERT transformer (v1) achieves the highest overall accuracy (89.75%) and outperforms the unbalanced SVM baseline on macro F1 (+2.4 pp) and neither F1 (+5.2 pp).
- Adding class balancing to the SVM (v2) closes most of the macro F1 gap with the transformer.
- The weighted-loss transformer (v2) dramatically improves hate speech recall (35.5% → 74.3%) at the cost of lower accuracy and more false positives — a better fit for regulated environments where missing hate speech carries legal risk.
- All models struggle most with hate speech due to severe class imbalance (~5.77%) and vocabulary overlap with the offensive class.

---

## Team Member Contributions

| Member | Contributions |
|--------|---------------|
| Michael Persson | Data preprocessing pipeline · Exploratory data analysis · TF-IDF baseline models (SVM, LR, RF) · ALBERT transformer training loop · Oversampling strategy (v1) |
| Timothy Tan | Class-balanced baselines and weighted CrossEntropyLoss transformer (v2) · Confidence/coverage threshold analysis · Report · README · `requirements.txt` |

---

## References

- Davidson, T., Warmsley, D., Macy, M., & Weber, I. (2017). *Automated hate speech detection and the problem of offensive language.* ICWSM. https://arxiv.org/abs/1703.04009
- Lan, Z. et al. (2020). *ALBERT: A Lite BERT for Self-supervised Learning.* ICLR. https://arxiv.org/abs/1909.11942
- Hugging Face Transformers: https://huggingface.co/docs/transformers
- Scikit-learn TF-IDF: https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html