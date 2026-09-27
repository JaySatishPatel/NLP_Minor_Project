# ============================================================
# NLP-Based Price Complaint Analyzer
# Complaint Classification Module
# ============================================================
#
# Purpose:
# Classify a canteen complaint into one of five pricing categories:
# 1. High Price
# 2. Price Mismatch
# 3. Quantity-Price Concern
# 4. Price Increase
# 5. General Price Complaint
#
# Input:
#     dataset/processed_complaints.csv
#
# Output:
#     models/complaint_classifier.pkl
#     models/complaint_vectorizer.pkl
#     reports/figures/*.png
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
MODEL_PATH = os.path.join(MODEL_DIR, "complaint_classifier.pkl")
VECTORIZER_PATH = os.path.join(MODEL_DIR, "complaint_vectorizer.pkl")
REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports", "figures")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

# Import shared preprocessing pipeline
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


def load_classifier_assets():
    """Load cached complaint classifier model and vectorizer."""
    global _cached_model, _cached_vectorizer
    if _cached_model is None or _cached_vectorizer is None:
        if not os.path.exists(MODEL_PATH) or not os.path.exists(VECTORIZER_PATH):
            raise FileNotFoundError(
                f"Complaint classifier model/vectorizer not found in {MODEL_DIR}. "
                "Run train_complaint_classifier() first."
            )
        _cached_model = joblib.load(MODEL_PATH)
        _cached_vectorizer = joblib.load(VECTORIZER_PATH)
    return _cached_model, _cached_vectorizer


def predict_category(text, model=None, vectorizer=None):
    """
    Predict the pricing category for a complaint text.
    Returns:
        category (str): predicted category
        confidence (float): highest class probability (0.0 to 1.0)
        probabilities (dict): probability breakdown across all categories
    """
    if model is None or vectorizer is None:
        model, vectorizer = load_classifier_assets()

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

def train_complaint_classifier(dataset_path=DATASET_PATH, save_figures=True, show_figures=False, save_model=True):
    """
    Trains the multi-class complaint classification model, evaluates metrics,
    saves evaluation charts, and persists model & vectorizer.
    """
    print("=" * 70)
    print("PRICE COMPLAINT CLASSIFIER - TRAINING PIPELINE")
    print("=" * 70)

    if not os.path.exists(dataset_path):
        raise FileNotFoundError(
            f"Dataset not found at: {dataset_path}\n"
            "Run preprocessing.py first."
        )

    df = pd.read_csv(dataset_path)
    print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")

    required_columns = ["clean_text", "category"]
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' missing from dataset.")

    df = df.dropna(subset=required_columns).copy()
    df = df.drop_duplicates(subset=["clean_text"]).copy()

    category_counts = df["category"].value_counts()
    print("\nCategory distribution:")
    print(category_counts)

    # Visualize Category Distribution
    if save_figures or show_figures:
        plt.figure(figsize=(10, 5))
        sns.countplot(
            data=df,
            x="category",
            order=category_counts.index,
            hue="category",
            palette="Set2",
            legend=False
        )
        plt.title("Complaint Category Distribution", fontsize=14, fontweight="bold")
        plt.xlabel("Category", fontsize=11)
        plt.ylabel("Number of Complaints", fontsize=11)
        plt.xticks(rotation=25, ha="right")
        plt.tight_layout()
        if save_figures:
            dist_fig = os.path.join(REPORTS_DIR, "complaint_category_distribution.png")
            plt.savefig(dist_fig, dpi=300)
            print(f"Saved category distribution chart: {dist_fig}")
        if show_figures:
            plt.show()
        plt.close()

    # Train/Test Split
    X = df["clean_text"]
    y = df["category"]

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

    # Train Logistic Regression Classifier
    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42
    )
    model.fit(X_train_tfidf, y_train)

    print("\nLearned Classes:")
    for cls in model.classes_:
        print(f" - {cls}")

    # Evaluate Model
    y_pred = model.predict(X_test_tfidf)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nModel Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")

    print("\nClassification Report:")
    report = classification_report(y_test, y_pred, zero_division=0)
    print(report)

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
    if save_figures or show_figures:
        plt.figure(figsize=(9, 7))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=model.classes_,
            yticklabels=model.classes_
        )
        plt.title("Complaint Category - Confusion Matrix", fontsize=13, fontweight="bold")
        plt.xlabel("Predicted Category", fontsize=11)
        plt.ylabel("Actual Category", fontsize=11)
        plt.xticks(rotation=25, ha="right")
        plt.yticks(rotation=0)
        plt.tight_layout()
        if save_figures:
            cm_fig = os.path.join(REPORTS_DIR, "complaint_confusion_matrix.png")
            plt.savefig(cm_fig, dpi=300)
            print(f"Saved confusion matrix: {cm_fig}")
        if show_figures:
            plt.show()
        plt.close()

    # Save Model & Vectorizer
    if save_model:
        joblib.dump(model, MODEL_PATH)
        joblib.dump(vectorizer, VECTORIZER_PATH)
        print(f"\nSaved trained classifier: {MODEL_PATH}")
        print(f"Saved vectorizer:         {VECTORIZER_PATH}")

    # Test Sample Queries
    print("\n" + "=" * 70)
    print("TESTING SAMPLE COMPLAINTS")
    print("=" * 70)
    test_complaints = [
        "The price of the samosa is too high.",
        "I was charged 50 rupees but the menu says 40.",
        "The quantity is very small for this price.",
        "The price of tea has increased this month.",
        "The canteen should review its food prices."
    ]
    for complaint in test_complaints:
        category, conf, probs = predict_category(complaint, model, vectorizer)
        print(f"Complaint:  \"{complaint}\"")
        print(f"Category:   {category} ({conf:.1%} confidence)\n")

    print("=" * 70)
    print("COMPLAINT CLASSIFIER TRAINING COMPLETED")
    print("=" * 70)
    return model, vectorizer, accuracy


if __name__ == "__main__":
    train_complaint_classifier(save_figures=True, show_figures=False)