"""Tests for RepresentativeSampler."""

import numpy as np
import pandas as pd

from dfcontext.sampler import RepresentativeSampler


class TestRepresentativeSampler:
    def test_basic_sample(self) -> None:
        df = pd.DataFrame({"x": range(100)})
        sampler = RepresentativeSampler()
        result = sampler.sample(df, max_rows=5)

        assert len(result) == 5

    def test_small_dataframe(self) -> None:
        df = pd.DataFrame({"x": [1, 2, 3]})
        sampler = RepresentativeSampler()
        result = sampler.sample(df, max_rows=10)

        assert len(result) == 3

    def test_empty_dataframe(self) -> None:
        df = pd.DataFrame()
        sampler = RepresentativeSampler()
        result = sampler.sample(df, max_rows=5)

        assert len(result) == 0

    def test_stratified_with_categories(self) -> None:
        np.random.seed(42)
        df = pd.DataFrame({
            "cat": ["A"] * 50 + ["B"] * 30 + ["C"] * 20,
            "val": range(100),
        })
        sampler = RepresentativeSampler()
        result = sampler.sample(df, max_rows=6)

        # Should have samples from multiple categories
        assert len(result) <= 6
        assert result["cat"].nunique() >= 2

    def test_numeric_diverse(self) -> None:
        df = pd.DataFrame({"x": range(100)})
        sampler = RepresentativeSampler()
        result = sampler.sample(df, max_rows=5)

        # Should include values near min and max
        assert result["x"].min() <= 5
        assert result["x"].max() >= 95

    def test_deterministic(self) -> None:
        df = pd.DataFrame({"x": range(100)})
        sampler = RepresentativeSampler()
        r1 = sampler.sample(df, max_rows=5)
        r2 = sampler.sample(df, max_rows=5)

        pd.testing.assert_frame_equal(r1, r2)

    def test_zero_max_rows(self) -> None:
        df = pd.DataFrame({"x": [1, 2, 3]})
        sampler = RepresentativeSampler()
        result = sampler.sample(df, max_rows=0)

        assert len(result) == 0
