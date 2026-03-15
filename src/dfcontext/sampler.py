"""Representative row sampling."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd_runtime

if TYPE_CHECKING:
    import pandas as pd


class RepresentativeSampler:
    """Sample diverse, representative rows from a DataFrame.

    Uses stratified sampling when categorical columns exist,
    otherwise selects rows near min/max/median of numeric columns.
    """

    def sample(
        self,
        df: pd.DataFrame,
        max_rows: int,
        seed: int = 42,
    ) -> pd.DataFrame:
        """Sample representative rows.

        Parameters
        ----------
        df : pd.DataFrame
            The source DataFrame.
        max_rows : int
            Maximum number of rows to return.
        seed : int
            Random seed for reproducibility.

        Returns
        -------
        pd.DataFrame
            A subset of representative rows.
        """
        if df.empty or max_rows <= 0:
            return pd_runtime.DataFrame(df.head(0))

        if len(df) <= max_rows:
            return df

        # Try stratified sampling on categorical columns
        cat_cols = [
            c
            for c in df.columns
            if pd_runtime.api.types.is_object_dtype(df[c])
            or isinstance(
                df[c].dtype, pd_runtime.CategoricalDtype
            )
        ]

        if cat_cols:
            return _stratified_sample(
                df, cat_cols[0], max_rows, seed
            )

        # Numeric-only: pick diverse rows
        num_cols = [
            c
            for c in df.columns
            if pd_runtime.api.types.is_numeric_dtype(df[c])
        ]

        if num_cols:
            return _numeric_diverse_sample(
                df, num_cols[0], max_rows, seed
            )

        # Fallback: evenly spaced
        return _evenly_spaced(df, max_rows)


def _stratified_sample(
    df: pd.DataFrame,
    col: str,
    max_rows: int,
    seed: int,
) -> pd.DataFrame:
    """Sample rows stratified by a categorical column."""
    groups = df.groupby(col, observed=True)
    n_groups = groups.ngroups

    if n_groups == 0:
        return pd_runtime.DataFrame(df.head(max_rows))

    per_group = max(1, max_rows // n_groups)
    samples = []
    for _, group in groups:
        n = min(per_group, len(group))
        samples.append(group.sample(n=n, random_state=seed))

    result = pd_runtime.concat(samples)
    return pd_runtime.DataFrame(result.head(max_rows))


def _numeric_diverse_sample(
    df: pd.DataFrame,
    col: str,
    max_rows: int,
    seed: int,
) -> pd.DataFrame:
    """Sample rows near min, max, median, plus random."""
    sorted_df = df.sort_values(col)
    indices: set[int] = set()

    # Min, max, median
    indices.add(int(sorted_df.index[0]))
    indices.add(int(sorted_df.index[-1]))
    mid = len(sorted_df) // 2
    indices.add(int(sorted_df.index[mid]))

    # Fill remaining with random
    remaining = max_rows - len(indices)
    if remaining > 0:
        pool = sorted_df.index.difference(list(indices))
        if len(pool) > 0:
            n = min(remaining, len(pool))
            random_idx = sorted_df.loc[pool].sample(
                n=n, random_state=seed
            ).index
            indices.update(int(i) for i in random_idx)

    return pd_runtime.DataFrame(
        df.loc[list(indices)].head(max_rows)
    )


def _evenly_spaced(
    df: pd.DataFrame, max_rows: int
) -> pd.DataFrame:
    """Select evenly spaced rows."""
    import numpy as np

    idx = np.linspace(0, len(df) - 1, num=max_rows, dtype=int)
    return pd_runtime.DataFrame(df.iloc[idx])
