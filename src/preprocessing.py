# ============================================================
# NLP-Based Price Complaint Analyzer
# EDA + Text Preprocessing + Dataset Preparation
# ============================================================
#
# Purpose:
# Prepare the existing complaint dataset for sentiment analysis
#
# Input:
#     dataset/complaints.csv
#
# Output:
#     dataset/processed_complaints.csv
#
# ============================================================


# ============================================================
# 1. IMPORT LIBRARIES
# ============================================================

import re
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from collections import Counter


# ============================================================
# 2. LOAD EXISTING DATASET
# ============================================================

DATASET_PATH = "dataset/complaints.csv"

if not os.path.exists(DATASET_PATH):
    raise FileNotFoundError(
        f"Dataset not found at: {DATASET_PATH}"
    )

df = pd.read_csv(DATASET_PATH)

print("=" * 60)
print("DATASET LOADED")
print("=" * 60)

print("\nFirst 5 records:")
print(df.head())


# ============================================================
# 3. BASIC DATASET INFORMATION
# ============================================================

print("\n" + "=" * 60)
print("DATASET INFORMATION")
print("=" * 60)

print("\nShape:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nData types:")
print(df.dtypes)

print("\nDataset information:")
df.info()


# ============================================================
# 4. CHECK REQUIRED COLUMNS
# ============================================================

required_columns = ["complaint", "category"]

for column in required_columns:

    if column not in df.columns:
        raise ValueError(
            f"Required column '{column}' is missing from dataset."
        )

print("\nRequired columns are present.")


# ============================================================
# 5. CHECK MISSING VALUES
# ============================================================

print("\n" + "=" * 60)
print("MISSING VALUE ANALYSIS")
print("=" * 60)

missing_values = df.isnull().sum()

print(missing_values)


# Remove rows where complaint or category is missing
df = df.dropna(
    subset=["complaint", "category"]
).copy()

print("\nDataset shape after removing missing values:")
print(df.shape)


# ============================================================
# 6. CHECK DUPLICATE COMPLAINTS
# ============================================================

print("\n" + "=" * 60)
print("DUPLICATE ANALYSIS")
print("=" * 60)

duplicate_count = df.duplicated(
    subset=["complaint"]
).sum()

print("Duplicate complaints:", duplicate_count)


# Remove duplicate complaints
df = df.drop_duplicates(
    subset=["complaint"]
).copy()

print("Dataset shape after removing duplicates:")
print(df.shape)


# ============================================================
# 7. CHECK CATEGORY DISTRIBUTION
# ============================================================

print("\n" + "=" * 60)
print("CATEGORY DISTRIBUTION")
print("=" * 60)

category_distribution = df["category"].value_counts()

print(category_distribution)


# ============================================================
# 8. CATEGORY DISTRIBUTION VISUALIZATION
# ============================================================

plt.figure(figsize=(10, 6))

sns.countplot(
    data=df,
    x="category",
    order=df["category"].value_counts().index
)

plt.title("Distribution of Complaint Categories")
plt.xlabel("Complaint Category")
plt.ylabel("Number of Complaints")

plt.xticks(rotation=30)
plt.tight_layout()
plt.show()


# ============================================================
# 9. CHECK CATEGORY PERCENTAGES
# ============================================================

category_percentage = (
    df["category"]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)

print("\nCategory percentages:")
print(category_percentage)


# ============================================================
# 10. TEXT PREPROCESSING FUNCTION
# ============================================================

def preprocess_text(text):
    """
    Clean and normalize complaint text.

    Operations:
    1. Convert to string
    2. Convert to lowercase
    3. Remove URLs
    4. Remove email addresses
    5. Remove HTML tags
    6. Remove special characters
    7. Normalize whitespace
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

    # Keep alphabets, numbers and spaces
    text = re.sub(
        r"[^a-zA-Z0-9\s]",
        " ",
        text
    )

    # Replace multiple spaces with one
    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# ============================================================
# 11. APPLY TEXT PREPROCESSING
# ============================================================

df["clean_text"] = df["complaint"].apply(
    preprocess_text
)

print("\n" + "=" * 60)
print("PREPROCESSING EXAMPLES")
print("=" * 60)

print(
    df[
        ["complaint", "clean_text"]
    ].head(10).to_string(index=False)
)


# ============================================================
# 12. REMOVE EMPTY TEXT RECORDS
# ============================================================

empty_text_count = (
    df["clean_text"].str.len().eq(0).sum()
)

print("\nEmpty text records:", empty_text_count)

df = df[
    df["clean_text"].str.len() > 0
].copy()


# ============================================================
# 13. TEXT LENGTH ANALYSIS
# ============================================================

df["word_count"] = (
    df["clean_text"]
    .str.split()
    .str.len()
)

df["character_count"] = (
    df["clean_text"]
    .str.len()
)

print("\n" + "=" * 60)
print("TEXT LENGTH STATISTICS")
print("=" * 60)

print("\nWord count:")
print(df["word_count"].describe())

print("\nCharacter count:")
print(df["character_count"].describe())


# ============================================================
# 14. WORD COUNT DISTRIBUTION
# ============================================================

plt.figure(figsize=(10, 6))

sns.histplot(
    df["word_count"],
    bins=20,
    kde=True
)

plt.title("Complaint Word Count Distribution")
plt.xlabel("Number of Words")
plt.ylabel("Number of Complaints")

plt.tight_layout()
plt.show()


# ============================================================
# 15. FIND MOST COMMON WORDS
# ============================================================

all_words = []

for text in df["clean_text"]:
    all_words.extend(text.split())


word_frequency = Counter(all_words)

most_common_words = word_frequency.most_common(20)

print("\n" + "=" * 60)
print("TOP 20 MOST COMMON WORDS")
print("=" * 60)

for word, count in most_common_words:
    print(f"{word}: {count}")


# ============================================================
# 16. MOST COMMON WORD VISUALIZATION
# ============================================================

words = [
    item[0]
    for item in most_common_words
]

counts = [
    item[1]
    for item in most_common_words
]

plt.figure(figsize=(12, 6))

sns.barplot(
    x=counts,
    y=words
)

plt.title("Top 20 Most Frequent Words")
plt.xlabel("Frequency")
plt.ylabel("Word")

plt.tight_layout()
plt.show()


# ============================================================
# 17. INFORMAL LANGUAGE NORMALIZATION
# ============================================================
#
# This dictionary can be expanded later according to your
# actual student complaints.
#
# Example:
# "pls"       -> "please"
# "u"         -> "you"
# "ur"        -> "your"
# "bcoz"      -> "because"
# "expensiveee" -> "expensive"
#
# ============================================================

informal_words = {

    "pls": "please",
    "plz": "please",

    "u": "you",
    "ur": "your",

    "bcoz": "because",
    "coz": "because",

    "abt": "about",

    "cant": "cannot",
    "wont": "will not",

    "expensiveee": "expensive",
    "costlyyy": "costly"
}


def normalize_informal_words(text):

    words = text.split()

    normalized_words = []

    for word in words:

        if word in informal_words:
            normalized_words.append(
                informal_words[word]
            )

        else:
            normalized_words.append(word)

    return " ".join(normalized_words)


# Apply informal language normalization
df["clean_text"] = df["clean_text"].apply(
    normalize_informal_words
)


# ============================================================
# 18. OPTIONAL: NORMALIZE REPEATED CHARACTERS
# ============================================================
#
# Example:
#
# "expensiveeeeee" -> "expensive"
# "soooo costly"   -> "soo costly"
#
# This is useful for informal student text.
#
# ============================================================

def normalize_repeated_characters(text):

    return re.sub(
        r"(.)\1{2,}",
        r"\1\1",
        text
    )


df["clean_text"] = df["clean_text"].apply(
    normalize_repeated_characters
)


# ============================================================
# 19. DISPLAY FINAL PREPROCESSED DATA
# ============================================================

print("\n" + "=" * 60)
print("FINAL PREPROCESSED DATA")
print("=" * 60)

print(
    df[
        [
            "complaint",
            "clean_text",
            "category"
        ]
    ].head(15).to_string(index=False)
)


# ============================================================
# 20. CHECK FINAL DATASET BALANCE
# ============================================================

print("\n" + "=" * 60)
print("FINAL CATEGORY DISTRIBUTION")
print("=" * 60)

print(
    df["category"].value_counts()
)


# ============================================================
# 21. CREATE PROCESSED DATASET
# ============================================================

OUTPUT_PATH = "dataset/processed_complaints.csv"

df.to_csv(
    OUTPUT_PATH,
    index=False
)

print("\nProcessed dataset saved to:")
print(OUTPUT_PATH)


# ============================================================
# 22. FINAL DATASET SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("FINAL DATASET SUMMARY")
print("=" * 60)

print("Total records:", len(df))
print("Total columns:", len(df.columns))

print("\nColumns:")
for column in df.columns:
    print("-", column)

print("\nFinal dataset preview:")
print(df.head())


# ============================================================
# END
# ============================================================

print("\n" + "=" * 60)
print("DATA PREPARATION COMPLETED")
print("=" * 60)

print(
    "\nThe processed dataset is now ready for "
    "the next stage: sentiment analysis."
)