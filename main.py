# ============================================================
# NLP-Based Price Complaint Analyzer
# Unified Command-Line Interface & Orchestrator
# ============================================================
#
# Modes:
# 1. analyze     : Analyze a single complaint text directly
# 2. interactive : Real-time interactive terminal shell
# 3. train       : Execute full training pipeline (preprocess -> train)
# 4. batch       : Batch classify a CSV of complaints
# 5. web         : Launch the interactive web dashboard
#
# ============================================================

import argparse
import os
import sys

# Ensure root directory is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.preprocessing import run_preprocessing
from src.sentiment_analysis import train_sentiment_model
from src.complaint_classifier import train_complaint_classifier
from src.analyzer import PriceComplaintAnalyzer


def run_full_pipeline():
    """Runs data preprocessing, sentiment training, and classifier training sequentially."""
    print("\n" + "=" * 75)
    print("STARTING FULL END-TO-END TRAINING PIPELINE")
    print("=" * 75)

    print("\n[STEP 1/3] Running Data Cleaning, Slang Normalization & EDA...")
    run_preprocessing(save_figures=True, show_figures=False)

    print("\n[STEP 2/3] Training Sentiment Analysis Model...")
    train_sentiment_model(save_figures=True, show_figures=False, save_model=True)

    print("\n[STEP 3/3] Training Complaint Classification Model...")
    train_complaint_classifier(save_figures=True, show_figures=False, save_model=True)

    print("\n" + "=" * 75)
    print("ALL MODELS AND VISUALIZATIONS GENERATED SUCCESSFULLY!")
    print("Figures saved in: reports/figures/")
    print("Models saved in:  models/")
    print("=" * 75)


def run_interactive_mode(analyzer: PriceComplaintAnalyzer):
    """Interactive CLI prompt for evaluating complaints continuously."""
    print("\n" + "=" * 75)
    print("NLP PRICE COMPLAINT ANALYZER - INTERACTIVE MODE")
    print("Enter any complaint to analyze sentiment, category, urgency & actions.")
    print("Type 'exit' or 'quit' to return.")
    print("=" * 75)

    while True:
        try:
            user_input = input("\nEnter complaint > ").strip()
            if not user_input:
                continue
            if user_input.lower() in ["exit", "quit", "q"]:
                print("Exiting interactive mode. Goodbye!")
                break

            result = analyzer.analyze(user_input)
            analyzer.print_analysis(result)

        except (KeyboardInterrupt, EOFError):
            print("\nExiting interactive mode. Goodbye!")
            break
        except Exception as e:
            print(f"[Error] Failed to process complaint: {e}")


def run_batch_mode(analyzer: PriceComplaintAnalyzer, input_file: str, output_file: str):
    """Processes a CSV containing complaints and outputs analysis results."""
    import pandas as pd

    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")

    print(f"\nLoading batch data from: {input_file}")
    df = pd.read_csv(input_file)

    text_col = "complaint" if "complaint" in df.columns else ("clean_text" if "clean_text" in df.columns else None)
    if not text_col:
        raise ValueError("Could not find 'complaint' or 'clean_text' column in CSV.")

    print(f"Analyzing {len(df)} records using '{text_col}' column...")
    results = analyzer.analyze_batch(df[text_col].tolist())

    out_df = pd.concat([df.reset_index(drop=True), results.reset_index(drop=True)], axis=1)

    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
    out_df.to_csv(output_file, index=False)
    print(f"Batch analysis successfully saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="NLP-Based Price Complaint Analyzer - Unified Orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --mode analyze --text "The samosa was 20 on the menu but billed 30"
  python main.py --mode interactive
  python main.py --mode train
  python main.py --mode batch --input dataset/processed_complaints.csv --output reports/batch_results.csv
  python main.py --mode web --port 5000
        """
    )

    parser.add_argument(
        "--mode",
        choices=["analyze", "interactive", "train", "batch", "web"],
        default="interactive",
        help="Execution mode (default: interactive)"
    )
    parser.add_argument(
        "--text", "-t",
        type=str,
        default="",
        help="Complaint text for 'analyze' mode"
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        default=os.path.join(PROJECT_ROOT, "dataset", "processed_complaints.csv"),
        help="Input CSV path for 'batch' mode"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=os.path.join(PROJECT_ROOT, "reports", "batch_predictions.csv"),
        help="Output CSV path for 'batch' mode"
    )
    parser.add_argument(
        "--port", "-p",
        type=int,
        default=5000,
        help="Port for 'web' mode (default: 5000)"
    )

    args, unknown = parser.parse_known_args()

    # If user provided a raw positional string, treat as analyze text
    if unknown and not args.text:
        args.text = " ".join(unknown)
        args.mode = "analyze"

    if args.mode == "train":
        run_full_pipeline()
        return

    if args.mode == "web":
        from app import app
        print(f"\nStarting Web Application on http://127.0.0.1:{args.port} ...")
        app.run(host="0.0.0.0", port=args.port, debug=False)
        return

    # For inference modes, initialize analyzer
    analyzer = PriceComplaintAnalyzer()

    if args.mode == "analyze":
        if not args.text:
            print("[Error] Please specify text using --text \"your complaint\"")
            sys.exit(1)
        res = analyzer.analyze(args.text)
        analyzer.print_analysis(res)

    elif args.mode == "batch":
        run_batch_mode(analyzer, args.input, args.output)

    elif args.mode == "interactive":
        run_interactive_mode(analyzer)


if __name__ == "__main__":
    main()
