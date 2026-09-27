# ============================================================
# NLP-Based Price Complaint Analyzer
# Unified Complaint & Sentiment Analyzer (Connectivity Layer)
# ============================================================
#
# Purpose:
# Connects the Sentiment Analysis and Complaint Classification
# modules into a cohesive end-to-end inference engine with:
# - Joint sentiment + category predictions
# - Confidence scores & probability distributions
# - Urgency & priority scoring
# - Root-cause key terms extraction
# - Actionable administrative recommendations
#
# ============================================================

import os
import re
import sys
import joblib
import pandas as pd

# Set up project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.preprocessing import clean_text


class PriceComplaintAnalyzer:
    """
    Unified analyzer that integrates sentiment prediction and
    complaint categorization for campus canteen complaints.
    """

    RECOMMENDATIONS = {
        "Price Mismatch": (
            "URGENT AUDIT REQUIRED: Inspect POS billing registers, cash counters, "
            "and verify that physical menu board prices exactly match the billed amounts."
        ),
        "Quantity-Price Concern": (
            "PORTION AUDIT: Re-evaluate standard kitchen ladle sizes, plate portion weights, "
            "and value-to-cost perception for this food item."
        ),
        "High Price": (
            "PRICING REVIEW: Benchmark item cost against student budget guidelines. "
            "Consider offering budget-friendly student combos or subsidized staples."
        ),
        "Price Increase": (
            "TRANSPARENCY NOTICE: If prices were recently revised due to inflation or input costs, "
            "display an informative notice to students explaining the reasons in advance."
        ),
        "General Price Complaint": (
            "FEEDBACK MONITORING: Keep track of customer satisfaction trends and periodically "
            "survey student sentiment regarding overall cafeteria value."
        ),
        "Positive Feedback": (
            "BENCHMARK EXCELLENCE: Item price and quality are well-received by customers. "
            "Maintain current portion and pricing standards as best practices."
        )
    }

    KEYWORD_PATTERNS = {
        "Price Mismatch": [r"menu", r"bill", r"billed", r"charged", r"display", r"printed", r"mismatch", r"inconsistent", r"counter"],
        "Quantity-Price Concern": [r"quantity", r"portion", r"size", r"small", r"meager", r"tiny", r"less", r"little", r"short"],
        "High Price": [r"expensive", r"costly", r"overpriced", r"high", r"pricier", r"too much", r"unhappy"],
        "Price Increase": [r"increase", r"increased", r"jumped", r"hiked", r"before", r"used to", r"now", r"sharp"],
        "General Price Complaint": [r"price", r"rate", r"cost", r"worth", r"review", r"fair", r"reasonable", r"affordable", r"tasty", r"good"]
    }

    def __init__(self, models_dir=None):
        if models_dir is None:
            models_dir = os.path.join(PROJECT_ROOT, "models")
        self.models_dir = models_dir

        self.sentiment_model_path = os.path.join(models_dir, "sentiment_model.pkl")
        self.sentiment_vec_path = os.path.join(models_dir, "sentiment_vectorizer.pkl")
        self.classifier_model_path = os.path.join(models_dir, "complaint_classifier.pkl")
        self.classifier_vec_path = os.path.join(models_dir, "complaint_vectorizer.pkl")

        self.sentiment_model = None
        self.sentiment_vectorizer = None
        self.classifier_model = None
        self.classifier_vectorizer = None

        self._load_models()

    def _load_models(self):
        """Loads and caches both NLP models and vectorizers."""
        missing = []
        for path, name in [
            (self.sentiment_model_path, "Sentiment Model"),
            (self.sentiment_vec_path, "Sentiment Vectorizer"),
            (self.classifier_model_path, "Complaint Classifier"),
            (self.classifier_vec_path, "Complaint Vectorizer"),
        ]:
            if not os.path.exists(path):
                missing.append(f"{name} ({path})")

        if missing:
            raise FileNotFoundError(
                f"Missing model artifacts:\n" + "\n".join(f" - {m}" for m in missing) +
                "\nPlease train the models first using `python main.py --mode train`."
            )

        self.sentiment_model = joblib.load(self.sentiment_model_path)
        self.sentiment_vectorizer = joblib.load(self.sentiment_vec_path)
        self.classifier_model = joblib.load(self.classifier_model_path)
        self.classifier_vectorizer = joblib.load(self.classifier_vec_path)

    def extract_keywords(self, text, category):
        """Identifies key price-related terms and numbers triggering the complaint."""
        found = []
        # Find price numbers
        numbers = re.findall(r"\b\d+\b", text)
        for num in numbers:
            found.append(f"Amount/Number: {num}")

        patterns = self.KEYWORD_PATTERNS.get(category, [])
        lower = text.lower()
        for pat in patterns:
            matches = re.findall(r"\b" + pat + r"\b", lower)
            for m in set(matches):
                if m not in found:
                    found.append(m)

        return found

    def calculate_urgency(self, sentiment, category, category_conf):
        """Computes priority level based on sentiment, category, and model confidence."""
        if sentiment == "Positive":
            return "LOW (Positive Sentiment)"
        if category == "Price Mismatch":
            return "CRITICAL (Immediate Billing Action)"
        if category in ["Quantity-Price Concern", "High Price"]:
            return "HIGH (Review Required)" if category_conf > 0.6 else "MEDIUM"
        if category == "Price Increase":
            return "MEDIUM (Pricing Policy)"
        return "MEDIUM (General Feedback)"

    def analyze(self, raw_complaint: str) -> dict:
        """
        Runs full dual NLP analysis on a single complaint text.
        Returns a rich structured dictionary.
        """
        raw_complaint = str(raw_complaint).strip()
        if not raw_complaint:
            raise ValueError("Input complaint text cannot be empty.")

        cleaned = clean_text(raw_complaint)

        # 1. Sentiment Inference
        s_vec = self.sentiment_vectorizer.transform([cleaned])
        sentiment_pred = self.sentiment_model.predict(s_vec)[0]
        s_probs = self.sentiment_model.predict_proba(s_vec)[0]
        sentiment_conf = float(max(s_probs))
        sentiment_prob_dict = {
            cls: round(float(p), 4)
            for cls, p in zip(self.sentiment_model.classes_, s_probs)
        }

        # 2. Category Inference
        c_vec = self.classifier_vectorizer.transform([cleaned])
        category_pred = self.classifier_model.predict(c_vec)[0]
        c_probs = self.classifier_model.predict_proba(c_vec)[0]
        category_conf = float(max(c_probs))
        category_prob_dict = {
            cls: round(float(p), 4)
            for cls, p in zip(self.classifier_model.classes_, c_probs)
        }

        # 3. Urgency & Keywords
        urgency = self.calculate_urgency(sentiment_pred, category_pred, category_conf)
        keywords = self.extract_keywords(raw_complaint, category_pred)

        # 4. Action Recommendation
        if sentiment_pred == "Positive":
            recommendation = self.RECOMMENDATIONS["Positive Feedback"]
        else:
            recommendation = self.RECOMMENDATIONS.get(
                category_pred,
                self.RECOMMENDATIONS["General Price Complaint"]
            )

        return {
            "original_complaint": raw_complaint,
            "cleaned_text": cleaned,
            "sentiment": sentiment_pred,
            "sentiment_confidence": sentiment_conf,
            "sentiment_probabilities": sentiment_prob_dict,
            "category": category_pred,
            "category_confidence": category_conf,
            "category_probabilities": category_prob_dict,
            "urgency": urgency,
            "keywords_detected": keywords,
            "recommendation": recommendation,
        }

    def analyze_batch(self, complaints: list) -> pd.DataFrame:
        """Processes a list or series of complaints and returns a pandas DataFrame."""
        results = [self.analyze(c) for c in complaints]
        return pd.DataFrame(results)

    def print_analysis(self, result: dict):
        """Displays formatted analysis card in the terminal safely across all OS consoles."""
        print("\n" + "=" * 75)
        print("NLP PRICE COMPLAINT ANALYSIS REPORT")
        print("=" * 75)
        print(f"Complaint:          \"{result['original_complaint']}\"")
        print(f"Cleaned Text:       \"{result['cleaned_text']}\"")
        print("-" * 75)
        s_tag = "[POSITIVE]" if result["sentiment"] == "Positive" else "[NEGATIVE]"
        print(f"Sentiment:          {s_tag} {result['sentiment']} ({result['sentiment_confidence']:.1%} confidence)")
        print(f"Complaint Category: [CATEGORY] {result['category']} ({result['category_confidence']:.1%} confidence)")
        print(f"Urgency Level:      [PRIORITY] {result['urgency']}")
        if result["keywords_detected"]:
            print(f"Key Triggers:       [KEYWORDS] {', '.join(result['keywords_detected'])}")
        print("-" * 75)
        print("Probabilities Breakdown:")
        print("  Sentiment: " + ", ".join(f"{k}: {v:.1%}" for k, v in result["sentiment_probabilities"].items()))
        print("  Category:  " + ", ".join(f"{k}: {v:.1%}" for k, v in result["category_probabilities"].items()))
        print("-" * 75)
        print(f"Management Action:")
        print(f">> {result['recommendation']}")
        print("=" * 75)


if __name__ == "__main__":
    analyzer = PriceComplaintAnalyzer()
    samples = [
        "I was charged 50 rupees at the counter but the menu board said 35.",
        "The dosa price is extremely high for such a small portion.",
        "The tea price increased by 5 rupees yesterday without notice.",
        "The thali is reasonably priced, fresh, and very good value for money!"
    ]
    for s in samples:
        res = analyzer.analyze(s)
        analyzer.print_analysis(res)
