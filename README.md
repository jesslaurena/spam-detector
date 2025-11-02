# Spam Detection Project

## 📋 Project Overview
This project implements and compares **K-Nearest Neighbors (KNN)** and **Naive Bayes** algorithms for email spam detection. The goal is to classify emails as spam (1) or ham (0) and compare the performance of both algorithms.

## How to Run

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Download Dataset

```bash
python scripts/download_data.py
```

### 3. Train Your Models

Make sure your models are trained first:

```bash
# Train Naive Bayes
cd NaiveBayes
python src/train.py --data ../data/spam_assassin_preprocessed.csv --config config.yaml --save_dir models
cd ..
```

TODO: Add KNN

### 4. Run the App

```bash
streamlit run app.py
```

The app will open automatically in your browser at `http://localhost:8501`


## 📊 Dataset Information

### **Main Dataset**
- **File**: `data/spam_assassin_preprocessed.csv`
- **Size**: 100,000 emails
- **Format**: CSV with columns `id`, `email_content`, `label`
- **Labels**: 0 = Ham (legitimate), 1 = Spam

### **Preprocessed Feature Files*
- **Bag of Words**: `data/train_bow_features.csv`, `data/test_bow_features.csv`
- **TF-IDF**: `data/train_tfidf_features.csv`, `data/test_tfidf_features.csv`
- **Vocabularies**: `data/bow_vocabulary.json`, `data/tfidf_vocabulary.json`
- **Format**: CSV with `label` column + 5000 feature columns

### **Dataset Statistics**
- **Total emails**: 100,000
- **Ham emails (0)**: 67,206 (67.2%)
- **Spam emails (1)**: 32,794 (32.8%)
- **Training set**: 80,000 samples
- **Test set**: 20,000 samples
- **Feature dimensions**: 5,000 features per email

### **Preprocessing Applied**
✅ **Text preprocessing**: Lowercase, removed special characters, cleaned whitespace
✅ **Feature extraction**: Bag of Words and TF-IDF with 5,000 most frequent words
✅ **Train/test split**: 80/20 split with stratified sampling