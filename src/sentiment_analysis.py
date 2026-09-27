# ============================================================
# NLP-Based Price Complaint Analyzer
# Sentiment Analysis Module
# ============================================================
#
# Dataset:
#     310 Negative
#      30 Positive
#
# This module performs:
# 1. Load processed dataset
# 2. Validate sentiment labels
# 3. Analyze and visualize sentiment distribution
# 4. Stratified train/test split
# 5. TF-IDF feature extraction (unigrams + bigrams, sublinear TF)
# 6. Logistic Regression training with class balancing
# 7. Model evaluation & confusion matrix generation
# 8. Inference function with confidence scoring
# 9. Model and vectorizer persistence
#
# ============================================================

import os
import sys
import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

# ============================================================
# 1. PATH CONFIGURATION (Dynamic project root)
# ============================================================

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

DATASET_PATH = os.path.join(PROJECT_ROOT, "dataset", "processed_complaints.csv")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "sentiment_model.pkl")
VECTORIZER_PATH = os.path.join(MODEL_DIR, "sentiment_vectorizer.pkl")
REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports", "figures")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

# Import unified cleaning function from preprocessing
try:
    from src.preprocessing import clean_text
except ImportError:
    try:
        from preprocessing import clean_text
    except ImportError:
        import re

        def clean_text(text):
            text = str(text).lower()
            text = re.sub(r"http\S+|www\S+", " ", text)
            text = re.sub(r"\S+@\S+", " ", text)
            text = re.sub(r"<.*?>", " ", text)
            text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
            text = re.sub(r"\s+", " ", text)
            return text.strip()


# ============================================================
# 2. INFERENCE FUNCTIONS
# ============================================================

_cached_model = None
_cached_vectorizer = None


def load_sentiment_assets():
    """Load cached sentiment model and vectorizer."""
    global _cached_model, _cached_vectorizer
    if _cached_model is None or _cached_vectorizer is None:
        if not os.path.exists(MODEL_PATH) or not os.path.exists(VECTORIZER_PATH):
            raise FileNotFoundError(
                f"Sentiment model/vectorizer not found in {MODEL_DIR}. "
                "Run train_sentiment_model() first."
            )
        _cached_model = joblib.load(MODEL_PATH)
        _cached_vectorizer = joblib.load(VECTORIZER_PATH)
    return _cached_model, _cached_vectorizer


def predict_sentiment(text, model=None, vectorizer=None):
    """
    Predict sentiment for a new complaint text.
    Returns:
        sentiment (str): 'Positive' or 'Negative'
        confidence (float): highest probability score (0.0 to 1.0)
        probabilities (dict): probability breakdown per class
    """
    if model is None or vectorizer is None:
        model, vectorizer = load_sentiment_assets()

    cleaned = clean_text(text)
    text_vector = vectorizer.transform([cleaned])

    prediction = model.predict(text_vector)[0]
    probs = model.predict_proba(text_vector)[0]
    prob_dict = {cls: float(prob) for cls, prob in zip(model.classes_, probs)}
    confidence = float(max(probs))

    return prediction, confidence, prob_dict


# ============================================================
# 3. MODEL TRAINING PIPELINE
# ============================================================

def train_sentiment_model(dataset_path=DATASET_PATH, save_figures=True, show_figures=False, save_model=True):
    """
    Trains the sentiment classification model, evaluates performance,
    generates reports, and persists model & vectorizer artifacts.
    """
    print("=" * 70)
    print("SENTIMENT ANALYSIS - TRAINING PIPELINE")
    print("=" * 70)

    if not os.path.exists(dataset_path):
        raise FileNotFoundError(
            f"Dataset not found at: {dataset_path}\n"
            "Run preprocessing.py first."
        )

    df = pd.read_csv(dataset_path)
    print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")

    required_columns = ["clean_text", "sentiment"]
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' missing from dataset.")

    # Drop missing and standardize labels
    df = df.dropna(subset=required_columns).copy()
    df["sentiment"] = df["sentiment"].astype(str).str.strip().str.capitalize()

    allowed = {"Positive", "Negative"}
    invalid = set(df["sentiment"].unique()) - allowed
    if invalid:
        raise ValueError(f"Unexpected sentiment labels found: {invalid}")

    sentiment_counts = df["sentiment"].value_counts()
    print("\nSentiment distribution:")
    print(sentiment_counts)

    # Visualize distribution
    if save_figures or show_figures:
        plt.figure(figsize=(7, 5))
        sns.countplot(
            data=df,
            x="sentiment",
            order=["Negative", "Positive"],
            hue="sentiment",
            palette={"Negative": "#d9534f", "Positive": "#5cb85c"},
            legend=False
        )
        plt.title("Sentiment Distribution", fontsize=14, fontweight="bold")
        plt.xlabel("Sentiment", fontsize=11)
        plt.ylabel("Number of Complaints", fontsize=11)
        plt.tight_layout()
        if save_figures:
            dist_fig = os.path.join(REPORTS_DIR, "sentiment_distribution.png")
            plt.savefig(dist_fig, dpi=300)
            print(f"Saved distribution plot: {dist_fig}")
        if show_figures:
            plt.show()
        plt.close()

    # Train/Test Split
    X = df["clean_text"]
    y = df["sentiment"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    print(f"\nTraining samples: {len(X_train)} | Testing samples: {len(X_test)}")

    # TF-IDF Feature Extraction
    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        sublinear_tf=True
    )
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    # Train Logistic Regression with Balanced Class Weights
    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42
    )
    model.fit(X_train_tfidf, y_train)

    # Evaluate
    y_pred = model.predict(X_test_tfidf)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nModel Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")

    print("\nClassification Report:")
    report = classification_report(y_test, y_pred, labels=["Negative", "Positive"], zero_division=0)
    print(report)

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred, labels=["Negative", "Positive"])
    if save_figures or show_figures:
        plt.figure(figsize=(6, 5))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Negative", "Positive"],
            yticklabels=["Negative", "Positive"],
            cbar=False
        )
        plt.title("Sentiment Classification - Confusion Matrix", fontsize=13, fontweight="bold")
        plt.xlabel("Predicted Sentiment", fontsize=11)
        plt.ylabel("Actual Sentiment", fontsize=11)
        plt.tight_layout()
        if save_figures:
            cm_fig = os.path.join(REPORTS_DIR, "sentiment_confusion_matrix.png")
            plt.savefig(cm_fig, dpi=300)
            print(f"Saved confusion matrix: {cm_fig}")
        if show_figures:
            plt.show()
        plt.close()

    # Save Model & Vectorizer
    if save_model:
        joblib.dump(model, MODEL_PATH)
        joblib.dump(vectorizer, VECTORIZER_PATH)
        print(f"\nSaved trained model: {MODEL_PATH}")
        print(f"Saved vectorizer:    {VECTORIZER_PATH}")

    # Test Sample Queries
    print("\n" + "=" * 70)
    print("TESTING SAMPLE QUERIES")
    print("=" * 70)
    test_samples = [
        "The food is too expensive and the quantity is very small.",
        "The price is reasonable and the food quality is good.",
        "I am very unhappy with the high price of the sandwich.",
        "The canteen food is affordable and good.",
        "The price of this item is extremely high."
    ]
    for text in test_samples:
        pred, conf, probs = predict_sentiment(text, model, vectorizer)
        print(f"Input:      \"{text}\"")
        print(f"Prediction: {pred} ({conf:.1%} confidence)\n")

    print("=" * 70)
    print("SENTIMENT ANALYSIS TRAINING COMPLETED")
    print("=" * 70)
    return model, vectorizer, accuracy


if __name__ == "__main__":
    train_sentiment_model(save_figures=True, show_figures=False)