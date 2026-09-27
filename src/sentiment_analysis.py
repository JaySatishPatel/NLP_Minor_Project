# ============================================================
# NLP-Based Price Complaint Analyzer
# Sentiment Analysis Module
# ============================================================
#
# Dataset:
#     330 Negative
#      30 Positive
#
# This module performs:
#
# 1. Load processed dataset
# 2. Validate sentiment labels
# 3. Analyze sentiment distribution
# 4. Train/test split
# 5. TF-IDF feature extraction
# 6. Logistic Regression with class balancing
# 7. Model evaluation
# 8. Confusion matrix
# 9. Test new complaints
# 10. Save trained model and vectorizer
#
# ============================================================


# ============================================================
# 1. IMPORT LIBRARIES
# ============================================================

import os
import re
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


# ============================================================
# 2. FILE PATHS
# ============================================================

DATASET_PATH = "dataset/processed_complaints.csv"

MODEL_DIR = "models"

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "sentiment_model.pkl"
)

VECTORIZER_PATH = os.path.join(
    MODEL_DIR,
    "sentiment_vectorizer.pkl"
)


# Create model directory if it doesn't exist
os.makedirs(MODEL_DIR, exist_ok=True)


# ============================================================
# 3. LOAD PROCESSED DATASET
# ============================================================

if not os.path.exists(DATASET_PATH):

    raise FileNotFoundError(
        f"Dataset not found at: {DATASET_PATH}\n"
        "Run the preprocessing script first."
    )


df = pd.read_csv(DATASET_PATH)


print("=" * 70)
print("SENTIMENT ANALYSIS - DATASET LOADED")
print("=" * 70)

print("\nDataset shape:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nFirst 5 records:")
print(df.head())


# ============================================================
# 4. CHECK REQUIRED COLUMNS
# ============================================================

required_columns = [
    "clean_text",
    "sentiment"
]

for column in required_columns:

    if column not in df.columns:

        raise ValueError(
            f"Required column '{column}' "
            "is missing from the dataset."
        )


print("\nRequired columns are present.")


# ============================================================
# 5. CHECK MISSING VALUES
# ============================================================

print("\n" + "=" * 70)
print("MISSING VALUE CHECK")
print("=" * 70)

print(df[required_columns].isnull().sum())


# Remove missing text/sentiment records
df = df.dropna(
    subset=["clean_text", "sentiment"]
).copy()


# ============================================================
# 6. STANDARDIZE SENTIMENT LABELS
# ============================================================

df["sentiment"] = (
    df["sentiment"]
    .astype(str)
    .str.strip()
    .str.capitalize()
)


# ============================================================
# 7. CHECK SENTIMENT LABELS
# ============================================================

print("\n" + "=" * 70)
print("SENTIMENT LABELS")
print("=" * 70)

print(
    df["sentiment"]
    .value_counts()
)


# Only Positive and Negative are expected
allowed_sentiments = {
    "Positive",
    "Negative"
}

invalid_sentiments = set(
    df["sentiment"].unique()
) - allowed_sentiments


if invalid_sentiments:

    raise ValueError(
        f"Unexpected sentiment labels found: "
        f"{invalid_sentiments}"
    )


# ============================================================
# 8. SENTIMENT DISTRIBUTION
# ============================================================

sentiment_counts = (
    df["sentiment"]
    .value_counts()
)

print("\nSentiment distribution:")
print(sentiment_counts)


sentiment_percentage = (
    df["sentiment"]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)

print("\nSentiment percentage:")
print(sentiment_percentage)


# ============================================================
# 9. VISUALIZE SENTIMENT DISTRIBUTION
# ============================================================

plt.figure(figsize=(8, 5))

sns.countplot(
    data=df,
    x="sentiment",
    order=["Negative", "Positive"]
)

plt.title(
    "Sentiment Distribution"
)

plt.xlabel("Sentiment")
plt.ylabel("Number of Complaints")

plt.tight_layout()

plt.show()


# ============================================================
# 10. PREPARE INPUT AND TARGET
# ============================================================

X = df["clean_text"]
y = df["sentiment"]


print("\n" + "=" * 70)
print("INPUT / TARGET")
print("=" * 70)

print("Input feature: clean_text")
print("Target label : sentiment")


# ============================================================
# 11. TRAIN-TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.20,

    random_state=42,

    # Important because the dataset is imbalanced
    stratify=y
)


print("\n" + "=" * 70)
print("TRAIN / TEST SPLIT")
print("=" * 70)

print("Training samples:", len(X_train))
print("Testing samples :", len(X_test))

print("\nTraining sentiment distribution:")
print(y_train.value_counts())

print("\nTesting sentiment distribution:")
print(y_test.value_counts())


# ============================================================
# 12. TF-IDF FEATURE EXTRACTION
# ============================================================

vectorizer = TfidfVectorizer(

    # Maximum number of features
    max_features=5000,

    # Unigrams + bigrams
    ngram_range=(1, 2),

    # Give more importance to informative terms
    sublinear_tf=True
)


# Fit TF-IDF only on training data
X_train_tfidf = vectorizer.fit_transform(
    X_train
)


# Transform test data
X_test_tfidf = vectorizer.transform(
    X_test
)


print("\n" + "=" * 70)
print("TF-IDF")
print("=" * 70)

print(
    "Training TF-IDF shape:",
    X_train_tfidf.shape
)

print(
    "Testing TF-IDF shape:",
    X_test_tfidf.shape
)


# ============================================================
# 13. TRAIN SENTIMENT CLASSIFICATION MODEL
# ============================================================
#
# class_weight="balanced" is important because:
#
# Negative = 330
# Positive = 30
#
# Without balancing, the model may strongly favor
# the Negative class.
#
# ============================================================

model = LogisticRegression(

    max_iter=1000,

    class_weight="balanced",

    random_state=42
)


model.fit(
    X_train_tfidf,
    y_train
)


print("\n" + "=" * 70)
print("MODEL TRAINING")
print("=" * 70)

print(
    "Logistic Regression model trained successfully."
)

print(
    "Class balancing: ENABLED"
)


# ============================================================
# 14. MAKE PREDICTIONS
# ============================================================

y_pred = model.predict(
    X_test_tfidf
)


# ============================================================
# 15. MODEL ACCURACY
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)


print("\n" + "=" * 70)
print("MODEL ACCURACY")
print("=" * 70)

print(
    f"Accuracy: {accuracy:.4f}"
)

print(
    f"Accuracy: {accuracy * 100:.2f}%"
)


# ============================================================
# 16. CLASSIFICATION REPORT
# ============================================================

print("\n" + "=" * 70)
print("CLASSIFICATION REPORT")
print("=" * 70)

print(
    classification_report(
        y_test,
        y_pred,
        labels=["Negative", "Positive"],
        zero_division=0
    )
)


# ============================================================
# 17. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(

    y_test,

    y_pred,

    labels=[
        "Negative",
        "Positive"
    ]
)


plt.figure(figsize=(7, 6))

sns.heatmap(

    cm,

    annot=True,

    fmt="d",

    cmap="Blues",

    xticklabels=[
        "Negative",
        "Positive"
    ],

    yticklabels=[
        "Negative",
        "Positive"
    ]
)

plt.title(
    "Sentiment Classification - Confusion Matrix"
)

plt.xlabel(
    "Predicted Sentiment"
)

plt.ylabel(
    "Actual Sentiment"
)

plt.tight_layout()

plt.show()


# ============================================================
# 18. FUNCTION TO PREPROCESS NEW TEXT
# ============================================================

def preprocess_new_text(text):

    """
    Apply the same basic preprocessing to new
    complaint text before prediction.
    """

    text = str(text)

    # Lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(
        r"http\S+|www\S+",
        " ",
        text
    )

    # Remove email addresses
    text = re.sub(
        r"\S+@\S+",
        " ",
        text
    )

    # Remove HTML tags
    text = re.sub(
        r"<.*?>",
        " ",
        text
    )

    # Remove special characters
    text = re.sub(
        r"[^a-zA-Z0-9\s]",
        " ",
        text
    )

    # Normalize spaces
    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# ============================================================
# 19. SENTIMENT PREDICTION FUNCTION
# ============================================================

def predict_sentiment(text):

    """
    Predict sentiment for a new complaint.

    Returns:
        sentiment
        confidence
    """

    # Preprocess input
    clean_text = preprocess_new_text(
        text
    )

    # Convert text to TF-IDF
    text_vector = vectorizer.transform(
        [clean_text]
    )

    # Predict sentiment
    prediction = model.predict(
        text_vector
    )[0]

    # Get probabilities
    probabilities = model.predict_proba(
        text_vector
    )[0]

    # Highest probability
    confidence = max(
        probabilities
    )

    return prediction, confidence


# ============================================================
# 20. TEST NEW COMPLAINTS
# ============================================================

print("\n" + "=" * 70)
print("TESTING NEW COMPLAINTS")
print("=" * 70)


test_complaints = [

    "The food is too expensive and the quantity is very small.",

    "The price is reasonable and the food quality is good.",

    "I am very unhappy with the high price of the sandwich.",

    "The canteen food is affordable and good.",

    "The price of this item is extremely high."
]


for complaint in test_complaints:

    sentiment, confidence = predict_sentiment(
        complaint
    )

    print("\nComplaint:")
    print(complaint)

    print(
        "Predicted Sentiment:",
        sentiment
    )

    print(
        f"Confidence: {confidence:.2%}"
    )


# ============================================================
# 21. SAVE TRAINED MODEL
# ============================================================

joblib.dump(
    model,
    MODEL_PATH
)


# ============================================================
# 22. SAVE TF-IDF VECTORIZER
# ============================================================

joblib.dump(
    vectorizer,
    VECTORIZER_PATH
)


print("\n" + "=" * 70)
print("MODEL SAVED")
print("=" * 70)

print(
    "Model:",
    MODEL_PATH
)

print(
    "Vectorizer:",
    VECTORIZER_PATH
)


# ============================================================
# 23. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("SENTIMENT ANALYSIS COMPLETED")
print("=" * 70)

print(
    "\nDataset:",
    len(df),
    "records"
)

print(
    "Negative:",
    (df["sentiment"] == "Negative").sum()
)

print(
    "Positive:",
    (df["sentiment"] == "Positive").sum()
)

print(
    f"Model Accuracy: {accuracy * 100:.2f}%"
)

print(
    "\nThe sentiment-analysis model is ready "
    "for integration with the complaint classifier."
)