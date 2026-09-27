# ============================================================
# NLP-Based Price Complaint Analyzer
# EDA + Text Preprocessing + Dataset Preparation
# ============================================================
#
# Purpose:
# Prepare the complaint dataset for sentiment analysis and
# complaint category classification.
#
# Input:
#     dataset/complaints.csv
#
# Output:
#     dataset/processed_complaints.csv
#     reports/figures/*.png (visualizations)
#
# ============================================================

import os
import re
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# ============================================================
# 1. FILE PATH CONFIGURATION (Dynamic project root)
# ============================================================

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_PATH = os.path.join(PROJECT_ROOT, "dataset", "complaints.csv")
OUTPUT_PATH = os.path.join(PROJECT_ROOT, "dataset", "processed_complaints.csv")
REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports", "figures")

os.makedirs(REPORTS_DIR, exist_ok=True)


# ============================================================
# 2. INFORMAL LANGUAGE & SLANG NORMALIZATION DICTIONARY
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
    "expensiveeee": "expensive",
    "costlyyy": "costly",
    "sooo": "so",
    "too": "too",
    "rs": "rupees",
    "inr": "rupees",
    "chrg": "charge",
    "qty": "quantity",
}


# ============================================================
# 3. TEXT PREPROCESSING FUNCTIONS (Shared & Reusable)
# ============================================================

def preprocess_text(text):
    """
    Clean and normalize raw complaint text.
    1. Convert to string and lowercase
    2. Remove URLs, emails, and HTML tags
    3. Keep alphanumeric characters and spaces
    4. Normalize whitespace
    """
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"\S+@\S+", " ", text)
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def normalize_informal_words(text):
    """Replace informal words, abbreviations, and chat slang."""
    words = text.split()
    normalized_words = [informal_words.get(word, word) for word in words]
    return " ".join(normalized_words)


def normalize_repeated_characters(text):
    """Normalize elongated words (e.g., 'expensiveeee' -> 'expensive')."""
    return re.sub(r"(.)\1{2,}", r"\1\1", text)


def clean_text(text):
    """
    Unified end-to-end preprocessing pipeline for training and inference.
    Guarantees zero train-serve discrepancy across all modules.
    """
    text = preprocess_text(text)
    text = normalize_repeated_characters(text)
    text = normalize_informal_words(text)
    return text.strip()


# ============================================================
# 4. DATASET PREPARATION PIPELINE
# ============================================================

def run_preprocessing(dataset_path=DATASET_PATH, output_path=OUTPUT_PATH, save_figures=True, show_figures=False):
    """
    Executes full EDA, cleaning, feature engineering, and saving of processed dataset.
    """
    print("=" * 60)
    print("NLP-BASED PRICE COMPLAINT ANALYZER - PREPROCESSING")
    print("=" * 60)

    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found at: {dataset_path}")

    df = pd.read_csv(dataset_path)
    print(f"\nDataset loaded from: {dataset_path}")
    print(f"Initial shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")

    # Validate required columns
    required_columns = ["complaint", "category"]
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' is missing from dataset.")

    # Drop missing complaints/categories
    df = df.dropna(subset=required_columns).copy()

    # Clean & standardize sentiment column if present
    if "sentiment" in df.columns:
        df["sentiment"] = df["sentiment"].astype(str).str.strip().str.capitalize()
        df = df[df["sentiment"].isin(["Positive", "Negative"])].copy()

    # Drop exact duplicates in complaints
    df = df.drop_duplicates(subset=["complaint"]).copy()
    print(f"Shape after initial deduplication: {df.shape}")

    # Apply unified text preprocessing
    df["clean_text"] = df["complaint"].apply(clean_text)

    # Remove empty text records
    df = df[df["clean_text"].str.len() > 0].copy()

    # Deduplicate clean_text
    df = df.drop_duplicates(subset=["clean_text"]).copy()
    print(f"Shape after clean_text deduplication: {df.shape}")

    # Feature Engineering: text length statistics
    df["word_count"] = df["clean_text"].str.split().str.len()
    df["character_count"] = df["clean_text"].str.len()

    print("\nWord count statistics:")
    print(df["word_count"].describe())

    # Visualizations
    if save_figures or show_figures:
        # 1. Category Distribution Plot
        plt.figure(figsize=(10, 5))
        order = df["category"].value_counts().index
        sns.countplot(data=df, x="category", order=order, palette="viridis" if hasattr(sns, "color_palette") else None)
        plt.title("Distribution of Complaint Categories", fontsize=14, fontweight="bold")
        plt.xlabel("Complaint Category", fontsize=11)
        plt.ylabel("Count", fontsize=11)
        plt.xticks(rotation=25, ha="right")
        plt.tight_layout()
        if save_figures:
            cat_fig_path = os.path.join(REPORTS_DIR, "category_distribution.png")
            plt.savefig(cat_fig_path, dpi=300)
            print(f"Saved category distribution chart: {cat_fig_path}")
        if show_figures:
            plt.show()
        plt.close()

        # 2. Word Count Distribution Plot
        plt.figure(figsize=(10, 5))
        sns.histplot(df["word_count"], bins=20, kde=True, color="#2b5c8f")
        plt.title("Complaint Word Count Distribution", fontsize=14, fontweight="bold")
        plt.xlabel("Word Count", fontsize=11)
        plt.ylabel("Frequency", fontsize=11)
        plt.tight_layout()
        if save_figures:
            word_fig_path = os.path.join(REPORTS_DIR, "word_count_distribution.png")
            plt.savefig(word_fig_path, dpi=300)
            print(f"Saved word count distribution chart: {word_fig_path}")
        if show_figures:
            plt.show()
        plt.close()

        # 3. Top 20 Most Frequent Words
        all_words = []
        for text in df["clean_text"]:
            all_words.extend(text.split())
        word_freq = Counter(all_words).most_common(20)
        words, counts = zip(*word_freq)

        plt.figure(figsize=(10, 6))
        sns.barplot(x=list(counts), y=list(words), palette="crest" if hasattr(sns, "color_palette") else None)
        plt.title("Top 20 Most Frequent Words in Clean Complaints", fontsize=14, fontweight="bold")
        plt.xlabel("Frequency", fontsize=11)
        plt.ylabel("Word", fontsize=11)
        plt.tight_layout()
        if save_figures:
            top_fig_path = os.path.join(REPORTS_DIR, "top_words.png")
            plt.savefig(top_fig_path, dpi=300)
            print(f"Saved top words chart: {top_fig_path}")
        if show_figures:
            plt.show()
        plt.close()

    # Save processed dataset
    df.to_csv(output_path, index=False)
    print(f"\nProcessed dataset saved to: {output_path}")
    print(f"Total processed records: {len(df)}")
    print(f"Columns: {df.columns.tolist()}")

    print("\nCategory distribution:")
    print(df["category"].value_counts())

    if "sentiment" in df.columns:
        print("\nSentiment distribution:")
        print(df["sentiment"].value_counts())

    print("\n" + "=" * 60)
    print("DATA PREPARATION COMPLETED SUCCESSFULLY")
    print("=" * 60)
    return df


if __name__ == "__main__":
    run_preprocessing(save_figures=True, show_figures=False)