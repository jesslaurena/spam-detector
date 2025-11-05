# scripts/train_tfidf_logreg.py
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
import joblib

# load preprocessed raw text CSV (ensure column named 'email_content' or rename to 'text')
df = pd.read_csv("data/spam_assassin_preprocessed.csv", encoding="utf8")
if 'email_content' in df.columns:
    df = df.rename(columns={'email_content': 'text'})
df['text'] = df['text'].fillna("").astype(str)
y = df['label'].astype(int)

# split (or use existing split indices)
X_train_text, X_test_text, y_train, y_test = train_test_split(df['text'], y, test_size=0.2, random_state=42, stratify=y)

# TF-IDF vectorizer: unigrams + bigrams, trim rare terms, limit dim
vect = TfidfVectorizer(ngram_range=(1,2), min_df=5, max_features=5000, stop_words='english')
X_train = vect.fit_transform(X_train_text)
X_test  = vect.transform(X_test_text)

# classifier (use class_weight='balanced' if you want to emphasize recall)
clf = LogisticRegression(max_iter=1000, solver='saga', penalty='l2', C=1.0, class_weight=None)
clf.fit(X_train, y_train)

# evaluate
y_pred = clf.predict(X_test)
report = {
    "accuracy": metrics.accuracy_score(y_test, y_pred),
    "precision": metrics.precision_score(y_test, y_pred, pos_label=1, zero_division=0),
    "recall": metrics.recall_score(y_test, y_pred, pos_label=1, zero_division=0),
    "f1": metrics.f1_score(y_test, y_pred, pos_label=1, zero_division=0),
}
print(report)
print(metrics.classification_report(y_test, y_pred, digits=4))

# save artifacts
joblib.dump(clf, "NaiveBayes/models/logreg_tfidf.pkl")
joblib.dump(vect, "NaiveBayes/models/tfidf_vectorizer.pkl")