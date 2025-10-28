# Data Folder - Spam Detection Dataset

To download data:
```
pip install -r requirements.txt
python scripts/download_data.py
```

## 📁 What's in This Folder?

### **Original Data:**
- `spam_assassin.csv` - The original raw email dataset (5,798 emails)
- `spam_assassin_preprocessed.csv` - Cleaned text data (100,000 emails after augmentation)

### **Ready-to-Use Feature Files:** ⭐ **START HERE**
- `train_bow_features.csv` - Training data with Bag of Words features
- `test_bow_features.csv` - Test data with Bag of Words features  
- `train_tfidf_features.csv` - Training data with TF-IDF features
- `test_tfidf_features.csv` - Test data with TF-IDF features

### **Vocabulary Files:**
These are mostly for our reference to understand results and debug.
- `bow_vocabulary.json` - Word-to-index mapping for Bag of Words
- `tfidf_vocabulary.json` - Word-to-index mapping and IDF scores for TF-IDF

## 🚀 Quick Start Guide

### **Step 1: Load the Features**
```python
import csv

def load_features(filename):
    """Load preprocessed features from CSV file."""
    features = []
    labels = []
    
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)  # Skip header row
        
        for row in reader:
            labels.append(int(row[0]))      # First column = label (0=ham, 1=spam)
            features.append([float(x) for x in row[1:]])  # Rest = features
    
    return features, labels

# Load your data
X_train, y_train = load_features('data/train_bow_features.csv')
X_test, y_test = load_features('data/test_bow_features.csv')

print(f"Training samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")
print(f"Features per sample: {len(X_train[0])}")
```

### **Step 2: Understand the Data Format**

**Each CSV file has:**
- **Column 1**: Label (0 = Ham/legitimate email, 1 = Spam)
- **Columns 2-5001**: Features (5,000 numerical values per email)

**Example:**
```csv
label,feature_0,feature_1,feature_2,...,feature_4999
0,1,0,2,0,...,1
1,0,3,0,1,...,0
```

## 📊 Dataset Statistics

- **Total emails**: 100,000
- **Training set**: 80,000 emails
- **Test set**: 20,000 emails
- **Features per email**: 5,000
- **Spam ratio**: ~32% spam, ~68% ham

## 🔍 Understanding the Features

### **Bag of Words (BoW) Features:**
- **What it is**: Counts how many times each word appears in an email
- **Values**: Non-negative integers (0, 1, 2, 3, ...)
- **Example**: If "free" appears 3 times, that feature = 3
- **Good for**: Naive Bayes (count-based algorithms)

### **TF-IDF Features:**
- **What it is**: Term Frequency × Inverse Document Frequency
- **Values**: Non-negative decimals (0.0 to ~1.0)
- **Example**: "free" might have value 0.15 (rare word = higher weight)
- **Good for**: KNN (distance-based algorithms)