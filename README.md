# 🤖 AI/ML Internship – Advanced Tasks

> **Intern:** Khalid Rafique
> **Program:** AI/ML Engineering  Internship – DevelopersHub Corporation
> **Repository:** `ai-ml-internship-Advanced-Tasks-`
> **Due Date:** 25 May 2026

---

## 📋 Overview

This repository contains solutions to **3 out of 5 advanced AI/ML tasks** assigned as part of the DevelopersHub Corporation internship program. Each task demonstrates hands-on experience with cutting-edge machine learning and artificial intelligence techniques.

---

## ✅ Completed Tasks

### Task 1 – News Topic Classifier Using BERT
**Notebook:** [`Task1_News_Classifier_BERT.ipynb`](Task1_News_Classifier_BERT.ipynb)

**Objective:** Fine-tune a BERT transformer to classify news headlines into 4 categories: World, Sports, Business, Sci/Tech.

**Methodology:**
- Loaded the AG News dataset from Hugging Face (120K train / 7.6K test)
- Tokenized text using `bert-base-uncased` tokenizer (max_length=128)
- Fine-tuned BERT for sequence classification using Hugging Face `Trainer` API
- Evaluated with accuracy and weighted F1-score
- Deployed an interactive demo with Gradio

**Key Results:**
- Accuracy: ~92%+ on test set
- Weighted F1: ~92%+
- Live Gradio demo with public share link

**Tools:** HuggingFace Transformers, PyTorch, Gradio, Datasets

---

### Task 2 – End-to-End ML Pipeline with Scikit-learn
**Notebook:** [`Task2_ML_Pipeline_Customer_Churn.ipynb`](Task2_ML_Pipeline_Customer_Churn.ipynb)

**Objective:** Build a reusable, production-ready ML pipeline to predict customer churn using the Telco Churn Dataset.

**Methodology:**
- Loaded and cleaned the IBM Telco Customer Churn dataset (7,043 customers, 20 features)
- Built `sklearn.Pipeline` with `ColumnTransformer` for numeric scaling & categorical encoding
- Trained Logistic Regression and Random Forest classifiers
- Tuned hyperparameters with `GridSearchCV` (5-fold cross-validation)
- Exported the complete pipeline using `joblib` for production use

**Key Results:**
- Random Forest (tuned): ~81% accuracy, ~0.86 ROC AUC
- Top predictors: Contract type, Tenure, Monthly Charges
- Pipeline exported as `.pkl` — ready for deployment

**Tools:** Scikit-learn, Pandas, Matplotlib, Seaborn, joblib

---

### Task 5 – Auto Tagging Support Tickets Using LLM
**Notebook:** [`Task5_Auto_Tagging_Support_Tickets.ipynb`](Task5_Auto_Tagging_Support_Tickets.ipynb)

**Objective:** Automatically tag support tickets into categories using a Large Language Model with prompt engineering.

**Methodology:**
- Created a realistic dataset of 25 support tickets across 5 categories
- Used `google/flan-t5-base` (free, open-source, no API key needed)
- Implemented **Zero-Shot** classification — no examples provided
- Implemented **Few-Shot** classification — 1 labeled example per category
- Generated **Top-3 most probable tags** per ticket using multi-label scoring
- Compared zero-shot vs few-shot performance

**Key Results:**
- Few-shot consistently outperforms zero-shot
- Top-3 tagging captures ambiguous ticket overlap
- No API costs — runs fully on free Colab CPU

**Tools:** Hugging Face Transformers (Flan-T5), Pandas, Scikit-learn, Matplotlib

---

## 🚀 How to Run

### Google Colab (Recommended)
1. Open any `.ipynb` file from this repo
2. Click **Open in Colab** or upload manually to [colab.research.google.com](https://colab.research.google.com)
3. For Task 1: Switch to **GPU runtime** (`Runtime → Change runtime type → T4 GPU`)
4. Run all cells in order (`Runtime → Run all`)

### Local Jupyter
```bash
git clone https://github.com/kr8457/ai-ml-internship-Advanced-Tasks-.git
cd ai-ml-internship-Advanced-Tasks-
pip install -r requirements.txt
jupyter notebook
```

---

## 📦 Requirements

```
transformers>=4.35.0
torch>=2.0.0
datasets>=2.14.0
scikit-learn>=1.3.0
pandas>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
seaborn>=0.12.0
gradio>=4.0.0
joblib>=1.3.0
```

---

## 📁 Repository Structure

```
ai-ml-internship-Advanced-Tasks-/
│
├── Task1_News_Classifier_BERT.ipynb          # BERT news classification + Gradio demo
├── Task2_ML_Pipeline_Customer_Churn.ipynb    # Scikit-learn pipeline + GridSearchCV
├── Task5_Auto_Tagging_Support_Tickets.ipynb  # LLM-based ticket tagging (zero/few-shot)
│
└── README.md
```

---

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Hugging Face Transformers | BERT fine-tuning, Flan-T5 inference |
| Scikit-learn | ML pipelines, GridSearchCV |
| PyTorch | Deep learning backend |
| Gradio | Interactive model demo |
| Pandas / NumPy | Data processing |
| Matplotlib / Seaborn | Visualizations |
| joblib | Model serialization |

---

*Built for the DevelopersHub Corporation AI/ML Engineering Internship Program*
