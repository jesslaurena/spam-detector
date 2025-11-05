# Spam Detection Project

## 📋 Project Overview
This project implements and compares K-Nearest Neighbors (KNN) and Naive Bayes algorithms for email spam detection. The primary, high‑accuracy pipeline uses TF‑IDF + classifier (saved artifacts included). The app prefers TF‑IDF artifacts and falls back to the from‑scratch Multinomial Naive Bayes implementation. KNN is included as a from‑scratch NumPy baseline.

Labels:  
- 0 = ham (legitimate)  
- 1 = spam

---

## ⚙️ How to Run (Streamlit — copy/paste ready)

These instructions assume you have the repository locally and that model artifacts are already present under `NaiveBayes/models/`. Run every command from the project root (the folder that contains `app.py` and the `data/` directory).

### 1) Create & activate a virtual environment

- Windows (PowerShell)
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

- macOS / Linux (bash/zsh)
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2) Install dependencies
```bash
pip install -r requirements.txt
# ensure Streamlit and joblib are available
pip install streamlit joblib
```

### 3) Confirm trained artifacts exist (required for the app to use TF‑IDF model)
From repo root, list model files:
```bash
# POSIX (macOS / Linux)
ls NaiveBayes/models

# PowerShell (Windows)
Get-ChildItem NaiveBayes\models
```

The app looks for these files (preferred TF‑IDF flow):
- `NaiveBayes/models/tfidf_vectorizer.pkl` (TF‑IDF vectorizer)
- `NaiveBayes/models/logreg_tfidf.pkl` (TF‑IDF classifier)

Fallback Naive Bayes artifacts (optional):
- `NaiveBayes/models/nb_model.pkl`
- `NaiveBayes/models/vocab.json`

If any of the TF‑IDF artifacts are missing and you want to use TF‑IDF, train or add those files (see Training below). Otherwise the app will use the NB fallback automatically.

### 4) (Optional) Download data (if you need raw data or feature CSVs)
If you want to fetch the dataset and precomputed features:
```bash
python scripts/download_data.py
```
This places files into `data/`.

### 5) Start the Streamlit app (recommended)
Run from repo root:
```bash
python -m streamlit run app.py
```
The app opens at `http://localhost:8501` (Streamlit prints the exact URL in the terminal). The UI prefers the TF‑IDF model; if not found it uses Naive Bayes.

---

## ◦ Quick CLI checks

- Single text prediction (uses TF‑IDF if available):
```bash
python src/predict.py --text "Free entry in 2 a wkly comp to win FA Cup final"
```

- Force TF‑IDF paths if needed:
```bash
python src/predict.py --text "..." --tfidf-model NaiveBayes/models/logreg_tfidf.pkl --tfidf-vect NaiveBayes/models/tfidf_vectorizer.pkl
```

- Evaluate TF‑IDF model on labeled CSV:
```bash
python src/evaluate.py --data data/spam_assassin_preprocessed.csv --tfidf-model NaiveBayes/models/logreg_tfidf.pkl --tfidf-vect NaiveBayes/models/tfidf_vectorizer.pkl --output NaiveBayes/models/eval_tfidf.json
```

- Evaluate Naive Bayes (fallback) on labeled CSV:
```bash
python src/evaluate.py --data data/spam_assassin_preprocessed.csv --model NaiveBayes/models/nb_model.pkl --vocab NaiveBayes/models/vocab.json --pos-label 1 --output NaiveBayes/models/eval_nb.json
```

---

## ▶ Training (optional — only if you want to retrain)
The repo already contains trained models; retrain only if you need to update models.

- Train TF‑IDF + classifier (recommended for best accuracy)
```bash
python NaiveBayes/scripts/train_tfidf_logreg.py
# saves: NaiveBayes/models/tfidf_vectorizer.pkl, NaiveBayes/models/logreg_tfidf.pkl
```

- Train Naive Bayes from raw text:
```bash
python NaiveBayes/src/train.py --data data/spam_assassin_preprocessed.csv --config NaiveBayes/config.yaml --save_dir NaiveBayes/models
```

- Train Naive Bayes from precomputed Bag‑of‑Words:
```bash
python NaiveBayes/scripts/train_from_bow.py
```

Training scripts save model artifacts and `metrics.json` under `NaiveBayes/models/`.

---

## 📁 Dataset summary (what’s in `data/`)
- `spam_assassin_preprocessed.csv` — preprocessed raw texts (id, email_content, label)
- Precomputed feature files (ready-to-use for KNN or alternative pipelines):
  - `data/train_tfidf_features.csv`, `data/test_tfidf_features.csv` (TF‑IDF feature CSVs)
  - `data/train_bow_features.csv`, `data/test_bow_features.csv` (BoW features)
- Vocab files:
  - `data/tfidf_vocabulary.json`, `data/bow_vocabulary.json`

Dataset stats (for reference)
- Total emails (augmented): ~100,000  
- Train/Test split used: 80k / 20k  
- Feature dimension (precomputed): 5,000

---

## 🧠 Implementation notes
- Primary model: TF‑IDF vectorizer + classifier (high accuracy). Artifacts in `NaiveBayes/models/`.
- Naive Bayes: implemented from scratch — accepts sparse count dicts ({index: count}) and uses Laplace smoothing.
- KNN: implemented from scratch (NumPy). App uses TF‑IDF vectorizer to build query vectors for KNN.
- The app will automatically choose TF‑IDF over NB when TF‑IDF artifacts are present.

---

## ⚠️ Troubleshooting
- If `streamlit` is not found: either activate the venv or run:
  ```bash
  python -m streamlit run app.py
  ```
- If models are missing: verify `NaiveBayes/models/` contains the expected files listed above.
- If outputs look incorrect or identical every time:
  - Ensure the TF‑IDF vectorizer used to transform inputs matches the vectorizer used when training the TF‑IDF classifier.
  - If using NB, ensure `vocab.json` corresponds to the NB model (index mapping must match).
- If KNN is slow: KNN is heavier (nearest-neighbor search across many dense vectors). The app preloads the index and uses a vectorized implementation; reducing TF‑IDF `max_features` will reduce memory/time.

---

## ✅ Quick verification (one command)
From project root, after installing dependencies and confirming model files:
```bash
python -m streamlit run app.py
```
Open the app, paste one of these example messages and click Analyze:
- Ham: `Go until jurong point, crazy.. Available only in bugis n great world la e buffet...`
- Spam: `Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005. Text FA to 87121 to receive entry`

---

## Team
- Jess Anderson — data cleaning & frontend  
- Suchir Kolli — Naive Bayes, integration (you)  
- Isabella Pareja — KNN implementation


