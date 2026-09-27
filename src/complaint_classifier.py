# ============================================================
# NLP-Based Price Complaint Analyzer
# Complaint Classification Module
# ============================================================
#
# Purpose:
# Classify a canteen complaint into one of the following:
#
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
    "complaint_classifier.pkl"
)

VECTORIZER_PATH = os.path.join(
    MODEL_DIR,
    "complaint_vectorizer.pkl"
)


# Create models directory if it doesn't exist
os.makedirs(MODEL_DIR, exist_ok=True)


# ============================================================
# 3. LOAD PROCESSED DATASET
# ============================================================

if not os.path.exists(DATASET_PATH):

    raise FileNotFoundError(
        f"Dataset not found at: {DATASET_PATH}\n"
        "Run preprocessing.py first."
    )


df = pd.read_csv(DATASET_PATH)


print("=" * 70)
print("PRICE COMPLAINT CLASSIFIER")
print("=" * 70)

print("\nDataset loaded successfully.")

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
    "category"
]

for column in required_columns:

    if column not in df.columns:

        raise ValueError(
            f"Required column '{column}' "
            "is missing from the dataset."
        )


print("\nRequired columns are present.")


# ============================================================
# 5. REMOVE MISSING VALUES
# ============================================================

print("\n" + "=" * 70)
print("MISSING VALUE CHECK")
print("=" * 70)

print(
    df[required_columns].isnull().sum()
)


df = df.dropna(
    subset=required_columns
).copy()


# ============================================================
# 6. REMOVE DUPLICATE RECORDS
# ============================================================

duplicate_count = df.duplicated(
    subset=["clean_text"]
).sum()

print("\nDuplicate complaints:", duplicate_count)


df = df.drop_duplicates(
    subset=["clean_text"]
).copy()


# ============================================================
# 7. CHECK CATEGORY DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("COMPLAINT CATEGORY DISTRIBUTION")
print("=" * 70)

category_counts = (
    df["category"]
    .value_counts()
)

print(category_counts)


# ============================================================
# 8. CATEGORY DISTRIBUTION VISUALIZATION
# ============================================================

plt.figure(figsize=(10, 6))

sns.countplot(
    data=df,
    x="category",
    order=category_counts.index
)

plt.title(
    "Complaint Category Distribution"
)

plt.xlabel(
    "Complaint Category"
)

plt.ylabel(
    "Number of Complaints"
)

plt.xticks(
    rotation=30
)

plt.tight_layout()

plt.show()


# ============================================================
# 9. CHECK CATEGORY BALANCE
# ============================================================

category_percentage = (
    df["category"]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)

print("\nCategory percentage:")
print(category_percentage)


# ============================================================
# 10. PREPARE INPUT AND TARGET
# ============================================================

X = df["clean_text"]

y = df["category"]


print("\n" + "=" * 70)
print("INPUT AND TARGET")
print("=" * 70)

print("Input  : clean_text")
print("Target : category")


# ============================================================
# 11. TRAIN-TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.20,

    random_state=42,

    # Maintain category proportions
    stratify=y
)


print("\n" + "=" * 70)
print("TRAIN / TEST SPLIT")
print("=" * 70)

print(
    "Training samples:",
    len(X_train)
)

print(
    "Testing samples:",
    len(X_test)
)

print("\nTraining category distribution:")
print(y_train.value_counts())

print("\nTesting category distribution:")
print(y_test.value_counts())


# ============================================================
# 12. TF-IDF FEATURE EXTRACTION
# ============================================================

vectorizer = TfidfVectorizer(

    # Maximum number of features
    max_features=5000,

    # Use individual words and two-word combinations
    ngram_range=(1, 2),

    # Improve representation of frequently occurring terms
    sublinear_tf=True
)


# Fit only on training data
X_train_tfidf = vectorizer.fit_transform(
    X_train
)


# Transform testing data
X_test_tfidf = vectorizer.transform(
    X_test
)


print("\n" + "=" * 70)
print("TF-IDF FEATURE EXTRACTION")
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
# 13. TRAIN COMPLAINT CLASSIFICATION MODEL
# ============================================================
#
# Logistic Regression is a strong baseline for
# text classification with TF-IDF features.
#
# class_weight="balanced" is included in case your
# complaint categories are not equally represented.
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
    "Complaint classification model "
    "trained successfully."
)

print(
    "\nClasses learned by the model:"
)

for class_name in model.classes_:
    print("-", class_name)


# ============================================================
# 14. MAKE TEST PREDICTIONS
# ============================================================

y_pred = model.predict(
    X_test_tfidf
)


# ============================================================
# 15. CALCULATE ACCURACY
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
        zero_division=0
    )
)


# ============================================================
# 17. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=model.classes_
)


plt.figure(figsize=(10, 8))

sns.heatmap(

    cm,

    annot=True,

    fmt="d",

    cmap="Blues",

    xticklabels=model.classes_,

    yticklabels=model.classes_
)


plt.title(
    "Complaint Category Classification - Confusion Matrix"
)

plt.xlabel(
    "Predicted Category"
)

plt.ylabel(
    "Actual Category"
)

plt.xticks(
    rotation=30
)

plt.yticks(
    rotation=0
)

plt.tight_layout()

plt.show()


# ============================================================
# 18. PREPROCESS NEW COMPLAINT
# ============================================================

def preprocess_new_complaint(text):
    """
    Apply the same basic preprocessing to a new
    complaint before classification.
    """

    text = str(text)

    # Convert to lowercase
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
# 19. PREDICTION FUNCTION
# ============================================================

def predict_category(text):
    """
    Predict the pricing-related category
    of a new complaint.

    Returns:
        category
        confidence
    """

    # Preprocess new complaint
    clean_text = preprocess_new_complaint(
        text
    )

    # Convert to TF-IDF
    text_vector = vectorizer.transform(
        [clean_text]
    )

    # Predict category
    prediction = model.predict(
        text_vector
    )[0]

    # Get probabilities
    probabilities = model.predict_proba(
        text_vector
    )[0]

    # Get highest probability
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

    "The price of the samosa is too high.",

    "I was charged 50 rupees but the menu says 40.",

    "The quantity is very small for this price.",

    "The price of tea has increased this month.",

    "The canteen should review its food prices."
]


for complaint in test_complaints:

    category, confidence = predict_category(
        complaint
    )

    print("\nComplaint:")
    print(complaint)

    print(
        "Predicted Category:",
        category
    )

    print(
        f"Confidence: {confidence:.2%}"
    )


# ============================================================
# 21. SAVE MODEL
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
    "Classifier:",
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
print("COMPLAINT CLASSIFICATION COMPLETED")
print("=" * 70)

print(
    "\nTotal records used:",
    len(df)
)

print(
    "Number of categories:",
    len(model.classes_)
)

print(
    f"Model Accuracy: {accuracy * 100:.2f}%"
)

print(
    "\nCategories:"
)

for category in model.classes_:
    print("-", category)

print(
    "\nThe complaint classification model "
    "is ready for integration."
)