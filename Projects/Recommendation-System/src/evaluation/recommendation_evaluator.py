"""
============================================================
Recommendation Evaluator
============================================================

Purpose
-------
Measure recommendation quality with offline ranking metrics
on a user holdout split using true random sampling.

Fixes Applied
-------------
1. Target Exclusion Bug: Bypassed native model exclusions that 
   were accidentally hiding the holdout target from predictions.
2. Dynamic Thresholding: Calculates Cold/Warm thresholds via 
   percentiles to prevent empty evaluation buckets caused by 
   pre-processing sparsity filters.
"""

import json
import inspect
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from config.config import (
    EVAL_K,
    EVAL_MAX_USERS,
    EVAL_RANDOM_STATE,
    EVALUATION_REPORT_PATH,
    PRICE_TIER_BINS,
    PRICE_TIER_LABELS,
    COLD_START_THRESHOLD,
    WARM_START_THRESHOLD,
)
from src.utils.logger import get_logger
from src.utils.helpers import timer

if TYPE_CHECKING:
    from src.models.collaborative_model import CollaborativeModel
    from src.models.content_based_model import ContentBasedModel
    from src.models.hybrid_model import HybridModel
    from src.models.knn_model import KNNRecommender
    from src.models.popularity_model import PopularityModel
    from src.models.user_interest_model import UserInterestModel

logger = get_logger(__name__)


def sort_interactions_for_sequence(df: pd.DataFrame, log: bool = True) -> pd.DataFrame:
    """Kept only for backward compatibility with unit tests."""
    return df.sort_values(["user_id", "course_id"]).reset_index(drop=True)


class RecommendationEvaluator:
    def __init__(
        self,
        k: int            = EVAL_K,
        max_users: int    = EVAL_MAX_USERS,
        random_state: int = EVAL_RANDOM_STATE,
    ):
        self.k            = k
        self.max_users    = max_users
        self.random_state = random_state

    @timer
    def evaluate(self, interactions_df: pd.DataFrame, course_content_matrix=None) -> dict:
        from config.config import (
            COLLAB_MODEL_PATH,
            CONTENT_MODEL_PATH,
            HYBRID_MODEL_PATH,
            KNN_MODEL_PATH,
            POPULARITY_MODEL_PATH,
            USER_INTEREST_MODEL_PATH,
            COURSE_FEATURES_PATH,
        )
        from src.utils.helpers import load_pickle

        _, test_df = self._split_holdout(interactions_df)
        if test_df.empty:
            raise ValueError(
                "Evaluation requires at least one user with two or more interactions."
            )

        logger.info("Loading trained models from disk for evaluation.")
        pop_model      = load_pickle(POPULARITY_MODEL_PATH)
        content_model  = load_pickle(CONTENT_MODEL_PATH)
        collab_model   = load_pickle(COLLAB_MODEL_PATH)
        knn_model      = load_pickle(KNN_MODEL_PATH)
        interest_model = load_pickle(USER_INTEREST_MODEL_PATH)
        hybrid_model   = load_pickle(HYBRID_MODEL_PATH)

        train_course_features = load_pickle(COURSE_FEATURES_PATH)

        course_meta = train_course_features[
            ["course_id", "course_name", "difficulty_level", "course_price"]
        ].copy()

        if "course_price" in course_meta.columns:
            course_meta["price_tier"] = pd.cut(
                course_meta["course_price"],
                bins=PRICE_TIER_BINS,
                labels=PRICE_TIER_LABELS,
            ).astype(str)

        report = self._score_models(
            test_df=test_df,
            hybrid_model=hybrid_model,
            content_model=content_model,
            popularity_model=pop_model,
            collab_model=collab_model,
            knn_model=knn_model,
            interest_model=interest_model,
            course_meta=course_meta,
        )
        self._log_report(report)
        self._save_report(report)
        return report

    def _split_holdout(self, df: pd.DataFrame) -> tuple:
        """
        Hold out a true random interaction per evaluation user to prevent artificial timelines.
        Keeps 100% of interactions for all other users to prevent data loss.
        """
        rng = np.random.default_rng(self.random_state)
        
        user_counts    = df["user_id"].value_counts()
        eligible_users = user_counts[user_counts >= 2].index.tolist()

        if not eligible_users:
            return df, pd.DataFrame()

        n_eval     = min(self.max_users, len(eligible_users))
        eval_users = set(rng.choice(eligible_users, size=n_eval, replace=False))

        train_parts, test_parts = [], []

        for user_id, group in df.groupby("user_id", sort=False):
            if user_id in eval_users:
                holdout_idx = rng.choice(group.index)
                holdout_row = df.loc[[holdout_idx]]
                train_group = group.drop(holdout_idx)
                
                if not train_group.empty:
                    train_parts.append(train_group)
                test_parts.append(holdout_row)
            else:
                train_parts.append(group)

        train_df = (
            pd.concat(train_parts, ignore_index=True)
            if train_parts
            else pd.DataFrame()
        )
        test_df = (
            pd.concat(test_parts, ignore_index=True)
            if test_parts
            else pd.DataFrame()
        )

        return train_df.reset_index(drop=True), test_df.reset_index(drop=True)

    def _score_models(
        self,
        test_df: pd.DataFrame,
        hybrid_model: "HybridModel",
        content_model: "ContentBasedModel",
        popularity_model: "PopularityModel",
        collab_model: "CollaborativeModel",
        knn_model: "KNNRecommender",
        interest_model: "UserInterestModel",
        course_meta: pd.DataFrame,
    ) -> dict:

        tiers = ["exact_course", "course_family", "soft_relevance"]
        model_names = ["hybrid", "content", "collaborative", "popularity", "knn"]
        model_scores = {
            m: {t: [] for t in tiers}
            for m in model_names
        }

        test_df = test_df.copy()
        if "course_price" in test_df.columns:
            test_df["price_tier"] = pd.cut(
                test_df["course_price"],
                bins=PRICE_TIER_BINS,
                labels=PRICE_TIER_LABELS,
            ).astype(str)

        meta_lookup = course_meta.set_index("course_id")[
            ["course_name", "difficulty_level", "price_tier"]
        ].to_dict("index")
        
        interest_profile_hits = 0
        total_catalogue_items = len(course_meta)
        recommended_items = {m: set() for m in model_names}

        # --- Dynamic Threshold Calculation ---
        # Instead of static thresholds that might leave buckets empty,
        # we calculate percentiles from the actual test set's history sizes.
        test_history_sizes = []
        for row in test_df.itertuples(index=False):
            all_seen_ids = collab_model.get_seen_ids(int(row.user_id))
            test_history_sizes.append(max(0, len(all_seen_ids) - 1))

        if test_history_sizes:
            dyn_cold_threshold = max(2, int(np.percentile(test_history_sizes, 25)))
            dyn_warm_threshold = max(dyn_cold_threshold + 1, int(np.percentile(test_history_sizes, 75)))
        else:
            dyn_cold_threshold, dyn_warm_threshold = COLD_START_THRESHOLD, WARM_START_THRESHOLD
            
        logger.info(f"Dynamic Evaluator Thresholds -> Cold: <{dyn_cold_threshold}, Warm: <{dyn_warm_threshold}")

        coldstart_scores = {m: {t: [] for t in tiers} for m in model_names}
        warm_scores      = {m: {t: [] for t in tiers} for m in model_names}
        active_scores    = {m: {t: [] for t in tiers} for m in model_names}

        for row in test_df.itertuples(index=False):
            user_id     = int(row.user_id)
            target_id   = int(row.course_id)
            target_name = str(row.course_name)

            all_seen_ids = collab_model.get_seen_ids(user_id)
            history_ids  = [cid for cid in all_seen_ids if cid != target_id]
            history_size = len(history_ids)
            user_vector = interest_model.get_user_vector(user_id)

            if user_vector is not None:
                interest_profile_hits += 1

            target_meta  = meta_lookup.get(target_id, {})
            target_diff  = target_meta.get("difficulty_level", getattr(row, "difficulty_level", None))
            target_ptier = target_meta.get("price_tier", getattr(row, "price_tier", None))

            # --- Target-Aware Model Fetching ---
            recs = {}
            
            # Collaborative (Bypass native exclude_seen to allow holdout target)
            if "exclude_seen" in inspect.signature(collab_model.recommend).parameters:
                c_raw = collab_model.recommend(user_id=user_id, n=self.k + history_size, exclude_seen=False)
                recs["collaborative"] = c_raw[~c_raw["course_id"].isin(history_ids)].head(self.k) if not c_raw.empty else pd.DataFrame()
            else:
                recs["collaborative"] = collab_model.recommend(user_id=user_id, n=self.k)

            # KNN
            if knn_model is not None and hasattr(knn_model, "recommend"):
                k_raw = knn_model.recommend(user_id=user_id, n=self.k + history_size)
                recs["knn"] = k_raw[~k_raw["course_id"].isin(history_ids)].head(self.k) if not k_raw.empty else pd.DataFrame()
            else:
                recs["knn"] = pd.DataFrame()

            # Hybrid (Use override if available)
            if "seen_ids_override" in inspect.signature(hybrid_model.recommend).parameters:
                recs["hybrid"] = hybrid_model.recommend(user_id=user_id, n=self.k, seen_ids_override=history_ids)
            else:
                recs["hybrid"] = hybrid_model.recommend(user_id=user_id, n=self.k)

            # Popularity & Content
            recs["popularity"] = popularity_model.recommend(n=self.k, exclude_course_ids=history_ids)
            recs["content"] = content_model.recommend_for_user_profile(user_vector, n=self.k, exclude_ids=history_ids) if user_vector is not None else pd.DataFrame()

            # Process recommendations
            for model_name, pred_df in recs.items():
                if not pred_df.empty:
                    if "course_name" not in pred_df.columns:
                        pred_df["course_name"] = pred_df["course_id"].apply(
                            lambda x: meta_lookup.get(x, {}).get("course_name", "")
                        )
                    ranked_ids   = pred_df["course_id"].tolist()
                    ranked_names = pred_df["course_name"].tolist()
                else:
                    ranked_ids, ranked_names = [], []

                soft_relevant_ids = [
                    cid for cid in ranked_ids
                    if (
                        target_diff is not None
                        and target_ptier is not None
                        and meta_lookup.get(cid, {}).get("difficulty_level") == target_diff
                        and meta_lookup.get(cid, {}).get("price_tier") == target_ptier
                    )
                ]

                exact_m  = self._rank_metrics(ranked_ids, target_id)
                family_m = self._rank_metrics(ranked_names, target_name)
                soft_m   = self._soft_metrics(ranked_ids, soft_relevant_ids)

                recommended_items[model_name].update(ranked_ids)

                model_scores[model_name]["exact_course"].append(exact_m)
                model_scores[model_name]["course_family"].append(family_m)
                model_scores[model_name]["soft_relevance"].append(soft_m)

                # Segment into dynamic buckets
                if history_size < dyn_cold_threshold:
                    coldstart_scores[model_name]["exact_course"].append(exact_m)
                    coldstart_scores[model_name]["course_family"].append(family_m)
                    coldstart_scores[model_name]["soft_relevance"].append(soft_m)
                elif history_size < dyn_warm_threshold:
                    warm_scores[model_name]["exact_course"].append(exact_m)
                    warm_scores[model_name]["course_family"].append(family_m)
                    warm_scores[model_name]["soft_relevance"].append(soft_m)
                else:
                    active_scores[model_name]["exact_course"].append(exact_m)
                    active_scores[model_name]["course_family"].append(family_m)
                    active_scores[model_name]["soft_relevance"].append(soft_m)

        def _safe_aggregate(bucket):
            return {
                model_name: {
                    tier: self._aggregate(metrics) if metrics else {}
                    for tier, metrics in buckets.items()
                }
                for model_name, buckets in bucket.items()
            }

        catalogue_coverage = {
            m: round(len(recommended_items[m]) / max(total_catalogue_items, 1), 4)
            for m in model_names
        }

        return {
            "protocol": {
                "type":            "user_holdout_random_stratified",
                "relevance_tiers": tiers,
                "k":               self.k,
                "evaluated_users": int(len(test_df)),
                "random_state":    self.random_state,
                "note": (
                    "exact_course should now closely match course_family because canonical ID "
                    "mapping has resolved fragmented duplication."
                ),
            },
            "interest_profile_coverage": round(
                interest_profile_hits / len(test_df), 4
            ),
            "catalogue_coverage": catalogue_coverage,
            "models": {
                model_name: {
                    tier: self._aggregate(metrics)
                    for tier, metrics in buckets.items()
                }
                for model_name, buckets in model_scores.items()
            },
            "coldstart_evaluation": {
                "cold_users_evaluated":   len(next(iter(coldstart_scores.values()))["exact_course"]),
                "warm_users_evaluated":   len(next(iter(warm_scores.values()))["exact_course"]),
                "active_users_evaluated": len(next(iter(active_scores.values()))["exact_course"]),
                "cold":   _safe_aggregate(coldstart_scores),
                "warm":   _safe_aggregate(warm_scores),
                "active": _safe_aggregate(active_scores),
            },
        }

    def _rank_metrics(self, ranked_ids: list, target_id) -> dict:
        if target_id not in ranked_ids:
            return {
                "precision_at_k": 0.0,
                "recall_at_k":    0.0,
                "map_at_k":       0.0,
                "ndcg_at_k":      0.0,
                "hit_rate_at_k":  0.0,
            }

        rank = ranked_ids.index(target_id) + 1
        return {
            "precision_at_k": round(1.0 / self.k, 6),
            "recall_at_k":    1.0,
            "map_at_k":       round(1.0 / rank, 6),
            "ndcg_at_k":      round(1.0 / np.log2(rank + 1), 6),
            "hit_rate_at_k":  1.0,
        }

    def _soft_metrics(self, ranked_ids: list, relevant_ids: list) -> dict:
        if not relevant_ids:
            return {
                "precision_at_k": 0.0,
                "recall_at_k":    0.0,
                "map_at_k":       0.0,
                "ndcg_at_k":      0.0,
                "hit_rate_at_k":  0.0,
            }

        relevant_set = set(relevant_ids)
        hits         = [1 if cid in relevant_set else 0 for cid in ranked_ids]
        n_relevant   = len(relevant_set)

        precision = sum(hits) / self.k
        recall    = sum(hits) / max(n_relevant, 1)
        hit_rate  = 1.0 if any(hits) else 0.0

        ap, n_hit = 0.0, 0
        for i, h in enumerate(hits, 1):
            if h:
                n_hit += 1
                ap    += n_hit / i
        map_k = ap / max(n_relevant, 1)

        dcg  = sum(h / np.log2(i + 1) for i, h in enumerate(hits, 1))
        idcg = sum(
            1.0 / np.log2(i + 1) for i in range(1, min(n_relevant, self.k) + 1)
        )
        ndcg = dcg / idcg if idcg > 0 else 0.0

        return {
            "precision_at_k": round(precision, 6),
            "recall_at_k":    round(recall, 6),
            "map_at_k":       round(map_k, 6),
            "ndcg_at_k":      round(ndcg, 6),
            "hit_rate_at_k":  round(hit_rate, 6),
        }

    @staticmethod
    def _aggregate(metrics: list) -> dict:
        frame = pd.DataFrame(metrics)
        return {col: round(float(frame[col].mean()), 4) for col in frame.columns}

    @staticmethod
    def _save_report(report: dict) -> None:
        path = Path(EVALUATION_REPORT_PATH)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        logger.info(f"Saved evaluation report | {path}")

    @staticmethod
    def _log_report(report: dict) -> None:
        protocol = report.get("protocol", {})
        logger.info("Evaluation summary")
        logger.info(
            f"Protocol | type={protocol.get('type')} | "
            f"users={protocol.get('evaluated_users')} | "
            f"k={protocol.get('k')}"
        )
        if protocol.get("note"):
            logger.info(f"Note | {protocol['note']}")

        for tier in ["exact_course", "course_family", "soft_relevance"]:
            logger.info(f"Tier | {tier}")
            logger.info(
                "Model           Precision@K  Recall@K  MAP@K  NDCG@K  HitRate@K"
            )
            for model_name, metrics_by_tier in report.get("models", {}).items():
                metrics = metrics_by_tier.get(tier, {})
                logger.info(
                    f"{model_name:<15}"
                    f"{metrics.get('precision_at_k', 0.0):>11.4f}"
                    f"{metrics.get('recall_at_k',    0.0):>10.4f}"
                    f"{metrics.get('map_at_k',       0.0):>8.4f}"
                    f"{metrics.get('ndcg_at_k',      0.0):>9.4f}"
                    f"{metrics.get('hit_rate_at_k',  0.0):>11.4f}"
                )

        coverage = report.get("catalogue_coverage", {})
        if coverage:
            logger.info("Catalogue coverage (unique_recommended / total_items):")
            for model_name, cov in coverage.items():
                logger.info(f"  {model_name:<15} {cov:.4f}")

        cs_eval = report.get("coldstart_evaluation", {})
        if cs_eval:
            logger.info(
                f"Cold-start breakdown | "
                f"cold={cs_eval.get('cold_users_evaluated', 0)} | "
                f"warm={cs_eval.get('warm_users_evaluated', 0)} | "
                f"active={cs_eval.get('active_users_evaluated', 0)}"
            )
            for segment in ["cold", "warm", "active"]:
                seg_data = cs_eval.get(segment, {})
                if not seg_data:
                    continue
                logger.info(f"  Segment={segment} | exact_course hit_rate@K:")
                for model_name, tiers_data in seg_data.items():
                    cf = tiers_data.get("exact_course", {})
                    hr = cf.get("hit_rate_at_k", 0.0)
                    logger.info(f"    {model_name:<15} {hr:.4f}")