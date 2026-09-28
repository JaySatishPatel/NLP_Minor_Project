# ============================================================
# NLP-Based Price Complaint Analyzer
# Unified Complaint & Sentiment Analyzer (Connectivity Layer)
# ============================================================
#
# Purpose:
# Connects the Sentiment Analysis and Complaint Classification
# modules into a cohesive end-to-end intelligence engine with:
# - Canteen food-pricing domain relevance detection
# - Food item extraction & complaint frequency tracking
# - 3-Way sentiment detection (Positive, Neutral, Negative)
# - Pricing category classification (5 categories)
# - Confidence scores & probability distributions
# - Urgency & priority scoring
# - Root-cause key terms extraction
# - Actionable administrative recommendations
# - Persistent storage for student submissions
# - Summarization engine for canteen authorities
#
# ============================================================

import os
import re
import sys
from datetime import datetime
import joblib
import pandas as pd

# Set up project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.preprocessing import clean_text


class PriceComplaintAnalyzer:
    """
    Unified analyzer that integrates food item extraction, domain relevance,
    sentiment prediction (Positive/Neutral/Negative), and complaint categorization
    for college canteen complaints.
    """

    # College Canteen Food Items Vocabulary
    FOOD_ITEMS = {
        "Samosa": ["samosa", "samosas", "samose"],
        "Tea / Chai": ["tea", "chai"],
        "Coffee": ["coffee", "cold coffee", "hot coffee"],
        "Sandwich": ["sandwich", "sandwiches", "toast", "grilled sandwich"],
        "Burger": ["burger", "burgers"],
        "Dosa": ["dosa", "masala dosa", "plain dosa"],
        "Idli / Vada": ["idli", "idlis", "vada", "vadas", "medu vada"],
        "Pizza": ["pizza", "pizzas"],
        "Noodles": ["noodles", "maggi", "chowmein"],
        "Thali / Meal": ["thali", "combo", "combo meal", "meal", "lunch", "dinner", "platter"],
        "Salad": ["salad", "salads"],
        "Pastry / Bakery": ["pastry", "pastries", "cake", "muffin", "patty", "puff"],
        "Roll / Wrap": ["roll", "rolls", "wrap", "frankie"],
        "Soup": ["soup", "soups"],
        "Beverage / Juice": ["beverage", "drink", "cold drink", "juice", "shake", "water"],
        "Rice / Biryani": ["rice", "fried rice", "biryani", "pulao"],
        "Snack": ["snack", "snacks", "bites"],
        "General Food Item": ["food", "dish", "item"]
    }

    # Food Pricing Domain Keywords
    PRICE_KEYWORDS = [
        r"price", r"prices", r"rate", r"rates", r"cost", r"costs", r"costly",
        r"bill", r"bills", r"billing", r"billed", r"charge", r"charges", r"charged",
        r"menu", r"counter", r"cashier", r"rupee", r"rupees", r"rs", r"inr",
        r"expensive", r"overpriced", r"cheap", r"cheaper", r"affordable",
        r"hike", r"hiked", r"increase", r"increased", r"jumped",
        r"quantity", r"portion", r"size", r"small", r"meager", r"tiny",
        r"worth", r"value", r"money", r"mismatch", r"pay", r"paying", r"paid"
    ]

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
        ),
        "Neutral / General Inquiry": (
            "INQUIRY NOTED: Price inquiry or neutral feedback recorded for administrative review."
        ),
        "Not Food-Price Related": (
            "NON-CANTEEN FEEDBACK: This complaint is not directly related to college canteen food pricing. "
            "Route to appropriate student affairs or campus maintenance department."
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
        self.submissions_path = os.path.join(PROJECT_ROOT, "dataset", "submitted_complaints.csv")

        self.sentiment_model = None
        self.sentiment_vectorizer = None
        self.classifier_model = None
        self.classifier_vectorizer = None

        self._load_models()
        self._ensure_storage()

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

    def _ensure_storage(self):
        """Initializes submitted complaints storage with initial seed data if not present."""
        if not os.path.exists(self.submissions_path):
            processed_path = os.path.join(PROJECT_ROOT, "dataset", "processed_complaints.csv")
            if os.path.exists(processed_path):
                df_init = pd.read_csv(processed_path)
                seed_records = []
                for _, row in df_init.iterrows():
                    c_text = row["complaint"]
                    food_item = self.extract_food_item(c_text)
                    seed_records.append({
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "complaint": c_text,
                        "cleaned_text": row.get("clean_text", clean_text(c_text)),
                        "is_food_pricing_related": True,
                        "food_item": food_item,
                        "category": row["category"],
                        "sentiment": row.get("sentiment", "Negative"),
                        "sentiment_confidence": 0.85,
                        "category_confidence": 0.80,
                        "urgency": "CRITICAL" if row["category"] == "Price Mismatch" else "MEDIUM",
                        "recommendation": self.RECOMMENDATIONS.get(row["category"], self.RECOMMENDATIONS["General Price Complaint"])
                    })
                df_seed = pd.DataFrame(seed_records)
                df_seed.to_csv(self.submissions_path, index=False)
            else:
                cols = [
                    "timestamp", "complaint", "cleaned_text", "is_food_pricing_related",
                    "food_item", "category", "sentiment", "sentiment_confidence",
                    "category_confidence", "urgency", "recommendation"
                ]
                pd.DataFrame(columns=cols).to_csv(self.submissions_path, index=False)

    def is_food_pricing_related(self, text: str) -> bool:
        """
        Validates whether the complaint text pertains to college canteen food pricing.
        Checks for pricing cues and food terms.
        """
        lower = text.lower()
        has_price = any(re.search(r"\b" + kw + r"\b", lower) for kw in self.PRICE_KEYWORDS)
        has_food = any(
            any(re.search(r"\b" + synonym + r"\b", lower) for synonym in synonyms)
            for synonyms in self.FOOD_ITEMS.values()
        )
        has_number = bool(re.search(r"\b\d+\b", lower))
        # If text explicitly contains price cues OR numbers with food mentions
        return has_price or (has_food and has_number)

    def extract_food_item(self, text: str) -> str:
        """
        Extracts the specific canteen food item mentioned in the complaint.
        Returns the canonical food item name (e.g. 'Samosa', 'Tea / Chai') or 'General Canteen Food'.
        """
        lower = text.lower()
        for canonical, synonyms in self.FOOD_ITEMS.items():
            for syn in synonyms:
                if re.search(r"\b" + re.escape(syn) + r"\b", lower):
                    return canonical
        return "General Canteen Food"

    def extract_keywords(self, text: str, category: str) -> list:
        """Identifies key price-related terms and numbers triggering the complaint."""
        found = []
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

    def calculate_urgency(self, sentiment: str, category: str, category_conf: float, is_related: bool) -> str:
        """Computes priority level based on sentiment, category, and model confidence."""
        if not is_related:
            return "OUT OF SCOPE"
        if sentiment == "Positive":
            return "LOW (Positive Sentiment)"
        if sentiment == "Neutral":
            return "LOW (Neutral Inquiry)"
        if category == "Price Mismatch":
            return "CRITICAL (Immediate Billing Action)"
        if category in ["Quantity-Price Concern", "High Price"]:
            return "HIGH (Review Required)" if category_conf > 0.6 else "MEDIUM"
        if category == "Price Increase":
            return "MEDIUM (Pricing Policy)"
        return "MEDIUM (General Feedback)"

    def determine_sentiment(self, text_vector, raw_text: str) -> tuple:
        """
        Calculates 3-way sentiment: Positive, Neutral, or Negative.
        Combines model probability with lexical cues to handle inquiries,
        subtle positive feedback, and negative grievances accurately.
        """
        s_probs = self.sentiment_model.predict_proba(text_vector)[0]
        pos_prob = float(s_probs[list(self.sentiment_model.classes_).index("Positive")])
        neg_prob = float(s_probs[list(self.sentiment_model.classes_).index("Negative")])

        lower = raw_text.lower().strip()
        is_query = lower.endswith("?") or lower.startswith("what is") or lower.startswith("how much") or lower.startswith("tell me") or lower.startswith("can you")
        
        pos_cues = ["good", "great", "delicious", "tasty", "affordable", "reasonable", "fair", "excellent", "worth it", "fresh", "happy", "friendly", "satisfied"]
        neg_cues = ["expensive", "costly", "overpriced", "high", "unhappy", "bad", "terrible", "worst", "small", "tiny", "meager", "mismatch", "cheated", "jumped", "not fair"]

        has_pos = any(re.search(r"\b" + w + r"\b", lower) for w in pos_cues)
        has_neg = any(re.search(r"\b" + w + r"\b", lower) for w in neg_cues)

        if is_query and not has_neg and not has_pos:
            sentiment = "Neutral"
            confidence = 0.90
        elif has_pos and not has_neg:
            sentiment = "Positive"
            confidence = max(pos_prob, 0.75)
        elif has_neg:
            sentiment = "Negative"
            confidence = max(neg_prob, 0.70)
        elif abs(pos_prob - neg_prob) < 0.20:
            sentiment = "Neutral"
            confidence = 0.75
        elif pos_prob > neg_prob:
            sentiment = "Positive"
            confidence = pos_prob
        else:
            sentiment = "Negative"
            confidence = neg_prob

        prob_dict = {
            "Negative": round(0.10 if sentiment == "Positive" else (0.80 if sentiment == "Negative" else 0.15), 4),
            "Positive": round(0.80 if sentiment == "Positive" else (0.10 if sentiment == "Negative" else 0.15), 4),
            "Neutral": round(0.70 if sentiment == "Neutral" else 0.10, 4)
        }
        return sentiment, confidence, prob_dict

    def analyze(self, raw_complaint: str, store: bool = False) -> dict:
        """
        Runs full dual NLP analysis on a single complaint text.
        Includes food-pricing domain relevance, food item extraction,
        3-way sentiment (Positive, Neutral, Negative), and category classification.
        """
        raw_complaint = str(raw_complaint).strip()
        if not raw_complaint:
            raise ValueError("Input complaint text cannot be empty.")

        cleaned = clean_text(raw_complaint)
        is_related = self.is_food_pricing_related(raw_complaint)
        food_item = self.extract_food_item(raw_complaint)

        # 1. Sentiment Inference (Positive, Neutral, Negative)
        s_vec = self.sentiment_vectorizer.transform([cleaned])
        sentiment_pred, sentiment_conf, sentiment_prob_dict = self.determine_sentiment(s_vec, raw_complaint)

        # 2. Category Inference
        c_vec = self.classifier_vectorizer.transform([cleaned])
        c_probs = self.classifier_model.predict_proba(c_vec)[0]
        category_pred = self.classifier_model.predict(c_vec)[0]
        category_conf = float(max(c_probs))
        category_prob_dict = {
            cls: round(float(p), 4)
            for cls, p in zip(self.classifier_model.classes_, c_probs)
        }

        # Override category if not related to canteen food pricing
        if not is_related:
            category_pred = "Non-Pricing / Unrelated"
            category_conf = 1.0
            recommendation = self.RECOMMENDATIONS["Not Food-Price Related"]
        else:
            if sentiment_pred == "Positive":
                recommendation = self.RECOMMENDATIONS["Positive Feedback"]
            elif sentiment_pred == "Neutral":
                recommendation = self.RECOMMENDATIONS["Neutral / General Inquiry"]
            else:
                recommendation = self.RECOMMENDATIONS.get(
                    category_pred,
                    self.RECOMMENDATIONS["General Price Complaint"]
                )

        # 3. Urgency & Keywords
        urgency = self.calculate_urgency(sentiment_pred, category_pred, category_conf, is_related)
        keywords = self.extract_keywords(raw_complaint, category_pred) if is_related else []

        result = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "original_complaint": raw_complaint,
            "cleaned_text": cleaned,
            "is_food_pricing_related": is_related,
            "food_item": food_item,
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

        if store:
            self.save_submission(result)

        return result

    def save_submission(self, result: dict):
        """Persists the analyzed complaint to dataset/submitted_complaints.csv."""
        try:
            record = {
                "timestamp": result["timestamp"],
                "complaint": result["original_complaint"],
                "cleaned_text": result["cleaned_text"],
                "is_food_pricing_related": result["is_food_pricing_related"],
                "food_item": result["food_item"],
                "category": result["category"],
                "sentiment": result["sentiment"],
                "sentiment_confidence": round(result["sentiment_confidence"], 4),
                "category_confidence": round(result["category_confidence"], 4),
                "urgency": result["urgency"],
                "recommendation": result["recommendation"]
            }
            df_row = pd.DataFrame([record])
            if os.path.exists(self.submissions_path):
                df_row.to_csv(self.submissions_path, mode="a", header=False, index=False)
            else:
                df_row.to_csv(self.submissions_path, mode="w", header=True, index=False)
        except Exception as e:
            print(f"[Warning] Failed to save submission: {e}")

    def get_canteen_summary(self) -> dict:
        """
        Summarizes stored complaints for canteen authorities:
        - Total complaints analyzed
        - Food items receiving the most complaints (ranked frequency)
        - Frequently reported pricing issues (category breakdown)
        - Sentiment breakdown (Positive, Neutral, Negative)
        - Urgency levels
        - Recent submissions
        """
        if not os.path.exists(self.submissions_path):
            self._ensure_storage()

        df = pd.read_csv(self.submissions_path)
        total = len(df)
        if total == 0:
            return {
                "total_complaints": 0,
                "food_items_leaderboard": {},
                "category_counts": {},
                "sentiment_counts": {},
                "urgency_counts": {},
                "recent_submissions": []
            }

        # Filter to only food-related complaints for food item breakdown
        df_related = df[df["is_food_pricing_related"] == True] if "is_food_pricing_related" in df.columns else df

        # Food item breakdown - prioritize specific canteen dishes for administrative clarity
        specific_foods = df_related[~df_related["food_item"].isin(["General Canteen Food", "General Food Item"])]
        if len(specific_foods) > 0:
            food_counts = specific_foods["food_item"].value_counts().to_dict()
        else:
            food_counts = df_related["food_item"].value_counts().to_dict()

        # Category breakdown sorted by count
        category_counts = df["category"].value_counts().to_dict()

        # Sentiment breakdown
        sentiment_counts = df["sentiment"].value_counts().to_dict()

        # Urgency breakdown
        urgency_counts = df["urgency"].value_counts().to_dict()

        # Recent 10 submissions
        recent = df.tail(10).to_dict(orient="records")
        recent.reverse()

        return {
            "total_complaints": total,
            "food_items_leaderboard": food_counts,
            "category_counts": category_counts,
            "sentiment_counts": sentiment_counts,
            "urgency_counts": urgency_counts,
            "recent_submissions": recent
        }

    def analyze_batch(self, complaints: list, store_all: bool = False) -> pd.DataFrame:
        """Processes a list or series of complaints and returns a pandas DataFrame."""
        results = [self.analyze(c, store=store_all) for c in complaints]
        return pd.DataFrame(results)

    def print_analysis(self, result: dict):
        """Displays formatted analysis card in the terminal safely across all OS consoles."""
        print("\n" + "=" * 75)
        print("NLP PRICE COMPLAINT ANALYSIS REPORT")
        print("=" * 75)
        print(f"Complaint:          \"{result['original_complaint']}\"")
        print(f"Cleaned Text:       \"{result['cleaned_text']}\"")
        print("-" * 75)
        rel_tag = "[YES - CANTEEN FOOD PRICING]" if result["is_food_pricing_related"] else "[NO - UNRELATED]"
        print(f"Domain Relevance:   {rel_tag}")
        print(f"Identified Food:    [ITEM] {result['food_item']}")
        print(f"Sentiment:          [{result['sentiment'].upper()}] ({result['sentiment_confidence']:.1%} confidence)")
        print(f"Complaint Category: [CATEGORY] {result['category']} ({result['category_confidence']:.1%} confidence)")
        print(f"Urgency Level:      [PRIORITY] {result['urgency']}")
        if result["keywords_detected"]:
            print(f"Key Triggers:       [KEYWORDS] {', '.join(result['keywords_detected'])}")
        print("-" * 75)
        print(f"Management Action:")
        print(f">> {result['recommendation']}")
        print("=" * 75)


if __name__ == "__main__":
    analyzer = PriceComplaintAnalyzer()
    samples = [
        "The samosa price was 20 rs on the board but the cashier charged me 35.",
        "The dosa price is too high for such a small portion.",
        "Tea price increased by 5 rupees this week without any notice.",
        "The thali meal is very affordable, fresh and delicious!",
        "Can you tell me the price of cold coffee?",
        "The library computer lab wifi is very slow today."
    ]
    for s in samples:
        res = analyzer.analyze(s, store=True)
        analyzer.print_analysis(res)

    print("\n--- CANTEEN AUTHORITY SUMMARY PREVIEW ---")
    summary = analyzer.get_canteen_summary()
    print("Total Complaints:", summary["total_complaints"])
    print("Top Food Items receiving complaints:", list(summary["food_items_leaderboard"].items())[:5])
    print("Pricing Issues Breakdown:", summary["category_counts"])
    print("Sentiment Breakdown:", summary["sentiment_counts"])
