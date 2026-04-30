"""
============================================================
Main Entry Point — Online Course Recommendation System
============================================================

Purpose
-------
Single entry point for all pipeline commands.

Usage
-----
    python main.py train
        Run the full training pipeline

    python main.py recommend --user_id 123 --n 10
        Get hybrid recommendations for a user

    python main.py similar --course_id 42 --n 5
        Get content-similar courses

    python main.py popular --n 10 --difficulty Beginner

    python main.py serve
        Launch the Streamlit dashboard

"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

from src.utils.logger import get_logger

logger = get_logger("main")


# ==========================================================
# Model Existence Guard
# ==========================================================

def _check_models_exist(*model_paths: str) -> bool:
    """
    Verify that all required model pickle files exist on disk.
    Returns True if all exist, False otherwise.
    Prints a helpful message and exits if any are missing.
    """
    missing = [p for p in model_paths if not os.path.exists(p)]
    if missing:
        print(
            "\n[ERROR] The following trained model files were not found:\n"
            + "\n".join(f"  - {p}" for p in missing)
            + "\n\nThe pipeline has not been trained yet. "
            "Run the following command first:\n"
            "\n    python main.py train\n"
        )
        return False
    return True


# ==========================================================
# Command Handlers
# ==========================================================

def cmd_train(_args):
    """Stage 1-6: Run the full end-to-end training pipeline."""
    from src.pipeline.training_pipeline import TrainingPipeline
    TrainingPipeline().run()


def cmd_serve(_args):
    """Launch the Streamlit dashboard."""
    import subprocess
    from config.config import HYBRID_MODEL_PATH

    if not os.path.exists(HYBRID_MODEL_PATH):
        logger.warning(
            "Hybrid model not found. Dashboard will launch but recommendations "
            "won't work until you run 'python main.py train'."
        )

    app_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "app", "app.py"
    )
    logger.info(f"Starting Streamlit dashboard: {app_path}")
    subprocess.run([sys.executable, "-m", "streamlit", "run", app_path], check=True)


def cmd_recommend(args):
    """
    Load the pre-trained hybrid model and generate recommendations.
    Uses HybridModel.serve() which calls load() then recommend() — it does
    NOT call fit(), so no sub-models are re-trained on this request.
    """
    from config.config import HYBRID_MODEL_PATH
    from src.models.hybrid_model import HybridModel

    if not _check_models_exist(HYBRID_MODEL_PATH):
        sys.exit(1)

    recs = HybridModel.serve(user_id=int(args.user_id), n=int(args.n))

    print(f"\nTop-{args.n} recommendations for user_id={args.user_id}:")
    print(recs.to_string(index=False))


def cmd_similar(args):
    """Load content model and find courses similar to a given course."""
    from config.config import CONTENT_MODEL_PATH
    from src.models.content_based_model import ContentBasedModel

    if not _check_models_exist(CONTENT_MODEL_PATH):
        sys.exit(1)

    model = ContentBasedModel.load()
    recs  = model.recommend_similar(int(args.course_id), n=int(args.n))

    print(f"\nTop-{args.n} courses similar to course_id={args.course_id}:")
    print(recs.to_string(index=False))


def cmd_popular(args):
    """Load popularity model and return trending courses."""
    from config.config import POPULARITY_MODEL_PATH
    from src.models.popularity_model import PopularityModel

    if not _check_models_exist(POPULARITY_MODEL_PATH):
        sys.exit(1)

    model = PopularityModel.load()
    recs  = model.recommend(n=int(args.n), difficulty=args.difficulty)

    print(f"\nTop-{args.n} popular courses:")
    print(recs.to_string(index=False))


# ==========================================================
# Argument Parser
# ==========================================================

def main():

    parser = argparse.ArgumentParser(
        description="Online Course Recommendation System"
    )
    sub = parser.add_subparsers(dest="command")

    # train
    sub.add_parser("train", help="Run full training pipeline")

    # serve
    sub.add_parser("serve", help="Launch Streamlit dashboard")

    # recommend
    p_rec = sub.add_parser("recommend", help="Hybrid recommendations for a user")
    p_rec.add_argument("--user_id", required=True, help="Target user ID")
    p_rec.add_argument("--n",       default=10,    help="Number of results")

    # similar
    p_sim = sub.add_parser("similar", help="Content-similar courses")
    p_sim.add_argument("--course_id", required=True, help="Seed course ID")
    p_sim.add_argument("--n",         default=10,    help="Number of results")

    # popular
    p_pop = sub.add_parser("popular", help="Popularity-based courses")
    p_pop.add_argument("--n", default=10, help="Number of results")
    p_pop.add_argument(
        "--difficulty",
        default=None,
        choices=["Beginner", "Intermediate", "Advanced"],
        help="Filter by difficulty level",
    )

    args = parser.parse_args()

    dispatch = {
        "train":     cmd_train,
        "serve":     cmd_serve,
        "recommend": cmd_recommend,
        "similar":   cmd_similar,
        "popular":   cmd_popular,
    }

    if args.command not in dispatch:
        parser.print_help()
        sys.exit(1)

    dispatch[args.command](args)


# ==========================================================
# Script Entry Point
# ==========================================================

if __name__ == "__main__":
    main()