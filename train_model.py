import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
import scipy.sparse as sp

# Load dataset
df = pd.read_csv("AuthentiText_X_2026_AI_vs_Human_Detection_1K.csv")

# Features
text = df["content_text"]
labels = df["author_type"]

# TF-IDF for text
vectorizer = TfidfVectorizer(max_features=500)
X_text = vectorizer.fit_transform(text)

# Numeric features
numeric_features = df[[
    "perplexity_score",
    "burstiness_index",
    "syntactic_variability",
    "semantic_coherence_score",
    "lexical_diversity_ratio",
    "readability_grade_level",
    "generation_confidence_score"
]].fillna(0)

# Combine text + numeric
X = sp.hstack([X_text, numeric_features.values])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# Save model + vectorizer
with open("model.pkl", "wb") as f:
    pickle.dump((model, vectorizer), f)

print("✅ Model trained and saved as model.pkl")