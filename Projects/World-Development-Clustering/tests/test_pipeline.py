"""
tests/test_pipeline.py
Unit tests for core pipeline components.

Run:
    pytest tests/test_pipeline.py -v
"""

import pytest
import numpy as np
import pandas as pd
import sys, os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.utils.helpers import strip_currency, strip_percent, missing_summary
from src.data.data_ingestion import fix_column_types, drop_unusable_columns, aggregate_by_country


# ── Helper function tests ─────────────────────────────────────────────────────

class TestHelpers:

    def test_strip_currency_normal(self):
        assert strip_currency("$1,234,567") == pytest.approx(1234567.0)

    def test_strip_currency_small(self):
        assert strip_currency("$60") == pytest.approx(60.0)

    def test_strip_currency_null(self):
        assert np.isnan(strip_currency(None))
        assert np.isnan(strip_currency(float("nan")))

    def test_strip_percent_normal(self):
        assert strip_percent("76.9%") == pytest.approx(76.9)

    def test_strip_percent_null(self):
        assert np.isnan(strip_percent(None))


# ── Ingestion tests ───────────────────────────────────────────────────────────

class TestIngestion:

    @pytest.fixture
    def sample_df(self):
        return pd.DataFrame({
            "Country"          : ["USA", "USA", "UK", "UK"],
            "GDP"              : ["$1,000,000", "$2,000,000", "$500,000", "$600,000"],
            "Business Tax Rate": ["25%", "26%", "20%", "21%"],
            "Ease of Business" : [50.0, 51.0, 60.0, 61.0],
            "Number of Records": [1, 1, 1, 1],
            "Birth Rate"       : [0.012, 0.012, 0.010, 0.010],
        })

    def test_fix_column_types(self, sample_df):
        df = fix_column_types(sample_df.copy())
        assert df["GDP"].dtype in [np.float64, float]
        assert df["Business Tax Rate"].dtype in [np.float64, float]
        assert df["GDP"].iloc[0] == pytest.approx(1_000_000.0)

    def test_drop_unusable_columns(self, sample_df):
        df = drop_unusable_columns(sample_df.copy())
        assert "Ease of Business" not in df.columns
        assert "Number of Records" not in df.columns

    def test_aggregate_by_country(self, sample_df):
        df = fix_column_types(sample_df)
        df = drop_unusable_columns(df)
        df_agg = aggregate_by_country(df)
        assert len(df_agg) == 2
        assert "Country" in df_agg.columns


# ── Preprocessing tests ───────────────────────────────────────────────────────

class TestPreprocessing:

    def test_knn_imputer_fills_nulls(self):
        from sklearn.impute import KNNImputer
        X = pd.DataFrame({
            "a": [1.0, 2.0, np.nan, 4.0, 5.0],
            "b": [10.0, 20.0, 30.0, 40.0, 50.0],
        })
        imputer = KNNImputer(n_neighbors=2)
        X_imp = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
        assert X_imp.isnull().sum().sum() == 0
        assert X_imp["a"].iloc[2] > 0

    def test_power_transformer_reduces_skewness(self):
        from sklearn.preprocessing import PowerTransformer
        np.random.seed(42)
        X = pd.DataFrame({"gdp": np.random.exponential(scale=1e6, size=100)})
        pt = PowerTransformer(method="yeo-johnson", standardize=False)
        X_t = pd.DataFrame(pt.fit_transform(X), columns=X.columns)
        assert abs(X_t["gdp"].skew()) < abs(X["gdp"].skew())

    def test_robust_scaler_resistant_to_outliers(self):
        from sklearn.preprocessing import RobustScaler
        X_clean   = np.array([[1], [2], [3], [4], [5]], dtype=float)
        X_outlier = np.array([[1], [2], [3], [4], [1000]], dtype=float)
        rs_clean   = RobustScaler().fit_transform(X_clean)
        rs_outlier = RobustScaler().fit_transform(X_outlier)
        assert abs(rs_clean[1, 0] - rs_outlier[1, 0]) < 0.5


# ── Clustering tests ──────────────────────────────────────────────────────────

class TestClustering:

    @pytest.fixture
    def X_sample(self):
        np.random.seed(42)
        c1 = np.random.randn(30, 5) + [0, 0, 0, 0, 0]
        c2 = np.random.randn(30, 5) + [5, 5, 5, 5, 5]
        c3 = np.random.randn(30, 5) + [10, 10, 10, 10, 10]
        return np.vstack([c1, c2, c3])

    def test_kmeans_labels_shape(self, X_sample):
        from sklearn.cluster import KMeans
        km = KMeans(n_clusters=3, random_state=42, n_init=10)
        labels = km.fit_predict(X_sample)
        assert len(labels) == 90
        assert len(set(labels)) == 3

    def test_gmm_probabilities_sum_to_one(self, X_sample):
        from sklearn.mixture import GaussianMixture
        gmm = GaussianMixture(n_components=3, random_state=42)
        gmm.fit(X_sample)
        probs = gmm.predict_proba(X_sample)
        assert probs.shape == (90, 3)
        assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-6)

    def test_silhouette_well_separated(self, X_sample):
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score
        km = KMeans(n_clusters=3, random_state=42, n_init=10)
        labels = km.fit_predict(X_sample)
        score = silhouette_score(X_sample, labels)
        assert score > 0.6