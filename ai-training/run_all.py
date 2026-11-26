r"""
SIGNBRIDGE AI — Master Training Pipeline Orchestrator
=======================================================
Orchestrates the complete Indian Sign Language (ISL) model training lifecycle:
  1. Dataset Scanning & Validation
  2. MediaPipe Landmark Extraction (Resumable)
  3. Feature Construction & Label Mapping
  4. Stratified Dataset Splitting
  5. Baseline Model Training
  6. Comprehensive Metric Evaluation
"""

import argparse
import sys
import time
from pathlib import Path

# Ensure real-time line buffering in console output
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

import config
from dataset_scanner import scan_dataset, print_scan_report
from landmark_extractor import extract_all
from dataset_builder import build_dataset
from dataset_splitter import split_processed_dataset
from train import train_model
from evaluate import evaluate_model


def run_pipeline(
    do_scan: bool = False,
    do_extract: bool = False,
    do_build: bool = False,
    do_split: bool = False,
    do_train: bool = False,
    do_eval: bool = False,
    limit_total: int = None,
    limit_per_class: int = None,
    overwrite: bool = False,
    model_type: str = config.DEFAULT_MODEL_TYPE,
):
    start_time = time.time()
    print("\n" + "#" * 72)
    print("           SIGNBRIDGE AI — COMPLETE MODEL TRAINING PIPELINE")
    print("#" * 72 + "\n")

    # STAGE 1: SCAN
    if do_scan:
        print("\n>>> STAGE 1: Scanning & Validating Dataset Videos...")
        manifest = scan_dataset()
        print_scan_report(manifest)

    # STAGE 2: EXTRACT
    if do_extract:
        print("\n>>> STAGE 2: Extracting MediaPipe Hand Landmarks...")
        extract_all(
            limit_total=limit_total,
            limit_per_class=limit_per_class,
            overwrite=overwrite,
        )

    # STAGE 3: BUILD
    if do_build:
        print("\n>>> STAGE 3: Building Structured Feature Matrices...")
        build_dataset()

    # STAGE 4: SPLIT
    if do_split:
        print("\n>>> STAGE 4: Partitioning Stratified Train/Val/Test Splits...")
        split_processed_dataset()

    # STAGE 5: TRAIN
    if do_train:
        print("\n>>> STAGE 5: Training Baseline Classification Model...")
        train_model(model_type=model_type)

    # STAGE 6: EVALUATE
    if do_eval:
        print("\n>>> STAGE 6: Evaluating Model on Held-out Test Split...")
        evaluate_model()

    elapsed = time.time() - start_time
    print("\n" + "#" * 72)
    print(f"Pipeline Execution Complete in {elapsed:.2f} seconds.")
    print("#" * 72 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="SIGNBRIDGE AI Training Pipeline Master Orchestrator."
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all pipeline stages in sequence (scan -> extract -> build -> split -> train -> eval).",
    )
    parser.add_argument("--scan", action="store_true", help="Run Stage 1: Dataset scanning only.")
    parser.add_argument("--extract", action="store_true", help="Run Stage 2: Batch landmark extraction only.")
    parser.add_argument("--build", action="store_true", help="Run Stage 3: Dataset feature building only.")
    parser.add_argument("--split", action="store_true", help="Run Stage 4: Dataset splitting only.")
    parser.add_argument("--train", action="store_true", help="Run Stage 5: Model training only.")
    parser.add_argument("--eval", action="store_true", help="Run Stage 6: Model evaluation only.")

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit total number of videos to process during extraction (useful for quick smoke tests).",
    )
    parser.add_argument(
        "--limit-per-class",
        type=int,
        default=None,
        help="Limit videos per class during extraction.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite previously extracted landmark files (disables resumable caching).",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default=config.DEFAULT_MODEL_TYPE,
        choices=["softmax_regression", "nearest_centroid"],
        help="Classifier architecture to train (default: softmax_regression).",
    )

    args = parser.parse_args()

    # If no flags passed, display usage and default to running help
    if not (args.all or args.scan or args.extract or args.build or args.split or args.train or args.eval):
        print("SIGNBRIDGE AI Pipeline Orchestrator")
        print("Usage:")
        print("  python run_all.py --all           # Run entire pipeline")
        print("  python run_all.py --scan          # Scan dataset only")
        print("  python run_all.py --extract       # Extract landmarks (resumable)")
        print("  python run_all.py --build         # Build feature matrices")
        print("  python run_all.py --split         # Split train/val/test")
        print("  python run_all.py --train         # Train baseline model")
        print("  python run_all.py --eval          # Evaluate test metrics")
        print("  python run_all.py --extract --limit 10 # Extract 10 sample videos")
        sys.exit(0)

    run_pipeline(
        do_scan=args.all or args.scan,
        do_extract=args.all or args.extract,
        do_build=args.all or args.build,
        do_split=args.all or args.split,
        do_train=args.all or args.train,
        do_eval=args.all or args.eval,
        limit_total=args.limit,
        limit_per_class=args.limit_per_class,
        overwrite=args.overwrite,
        model_type=args.model_type,
    )


if __name__ == "__main__":
    main()

