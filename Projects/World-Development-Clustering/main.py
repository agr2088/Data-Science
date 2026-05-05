"""
main.py
World Development Clustering — Pipeline Entry Point

Usage:
    python main.py               # full pipeline, auto k selection
    python main.py --k 4         # force k=4 for all models
    python main.py --stage ingest
    python main.py --stage preprocess
    python main.py --stage features
    python main.py --stage train
"""

import sys
import argparse

from src.utils.logger import get_logger

logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="World Development Clustering — ML Pipeline (5 Models)"
    )
    parser.add_argument(
        "--stage",
        choices=["ingest", "preprocess", "features", "train", "all"],
        default="all",
        help="Which pipeline stage to run (default: all)",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=None,
        help="Force a specific number of clusters (default: auto-select via Silhouette + BIC)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.stage == "ingest":
        from src.data.data_ingestion import run_ingestion
        run_ingestion()

    elif args.stage == "preprocess":
        from src.data.data_preprocessing import run_preprocessing
        run_preprocessing()

    elif args.stage == "features":
        from src.features.feature_engineering import run_feature_engineering
        from src.features.feature_selection   import run_feature_selection
        df_features, _ = run_feature_engineering()
        run_feature_selection(df_features)

    elif args.stage in ("train", "all"):
        from src.pipeline.training_pipeline import run_training_pipeline
        results = run_training_pipeline(optimal_k=args.k)
        logger.info(
            f"\nPipeline finished. "
            f"Selected: {results['best_model']} | k={results['optimal_k']}"
        )

    logger.info("Done.")


if __name__ == "__main__":
    main()