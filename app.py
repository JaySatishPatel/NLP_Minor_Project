# ============================================================
# NLP-Based Price Complaint Analyzer
# Flask Web Application & REST API
# ============================================================

import os
import sys
import pandas as pd
from flask import Flask, render_template, request, jsonify, send_file

# Set project root
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.analyzer import PriceComplaintAnalyzer

app = Flask(__name__)

# Initialize analyzer singleton
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
    """
    Submits a student complaint for NLP analysis.
    Stores the analyzed complaint to help canteen authorities track issues.
    """
    data = request.get_json(silent=True) or {}
    text = data.get("complaint", "").strip()
    store = data.get("store", True)  # Persist submissions by default

    if not text:
        return jsonify({"error": "Complaint text cannot be empty."}), 400

    try:
        active_analyzer = get_analyzer()
        result = active_analyzer.analyze(text, store=store)
        return jsonify({"status": "success", "data": result})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/canteen-summary", methods=["GET"])
def api_canteen_summary():
    """
    Returns aggregated analytics for college canteen authorities:
    - Top food items receiving complaints
    - Frequently reported pricing issues (category distribution)
    - Sentiment distribution (Positive, Neutral, Negative)
    - Recent submissions log
    """
    try:
        active_analyzer = get_analyzer()
        summary = active_analyzer.get_canteen_summary()
        return jsonify({"status": "success", "data": summary})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/export", methods=["GET"])
def api_export():
    """Exports all stored student complaints as a CSV file for administration."""
    csv_path = os.path.join(PROJECT_ROOT, "dataset", "submitted_complaints.csv")
    if not os.path.exists(csv_path):
        return jsonify({"error": "No complaints recorded yet."}), 404
    return send_file(
        csv_path,
        mimetype="text/csv",
        as_attachment=True,
        download_name="canteen_student_complaints_report.csv"
    )


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
