# ============================================================
# NLP-Based Price Complaint Analyzer
# Flask Web Application & REST API
# ============================================================

import os
import sys
import pandas as pd
from flask import Flask, render_template, request, jsonify

# Set project root
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.analyzer import PriceComplaintAnalyzer

app = Flask(__name__)

# Initialize analyzer
analyzer = None


def get_analyzer():
    global analyzer
    if analyzer is None:
        analyzer = PriceComplaintAnalyzer()
    return analyzer


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/analyze", methods=["POST"])
def api_analyze():
    data = request.get_json(silent=True) or {}
    text = data.get("complaint", "").strip()

    if not text:
        return jsonify({"error": "Complaint text cannot be empty."}), 400

    try:
        active_analyzer = get_analyzer()
        result = active_analyzer.analyze(text)
        return jsonify({"status": "success", "data": result})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/stats", methods=["GET"])
def api_stats():
    csv_path = os.path.join(PROJECT_ROOT, "dataset", "processed_complaints.csv")
    if not os.path.exists(csv_path):
        return jsonify({"error": "Dataset not found"}), 404

    df = pd.read_csv(csv_path)

    total_records = len(df)
    category_counts = df["category"].value_counts().to_dict()
    sentiment_counts = df["sentiment"].value_counts().to_dict() if "sentiment" in df.columns else {}

    # Sample complaints for demonstration
    samples = df.sample(min(8, len(df)), random_state=42)[["complaint", "category", "sentiment"]].to_dict(orient="records")

    return jsonify({
        "total_records": total_records,
        "categories": category_counts,
        "sentiments": sentiment_counts,
        "samples": samples
    })


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
