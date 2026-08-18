"""
src/clusterer.py

Vectorizes error_msg text with TF-IDF and clusters similar errors using
DBSCAN (cosine distance). Returns logs grouped by cluster ID, excluding
DBSCAN's noise label (-1).
"""

import logging
from typing import Dict, List, Union

import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.feature_extraction.text import TfidfVectorizer

logger = logging.getLogger(__name__)

LOG_FIELDS = ["job_id", "service", "status", "error_msg", "trace_id", "timestamp"]


def _to_dataframe(logs: Union[pd.DataFrame, List[dict]]) -> pd.DataFrame:
    if isinstance(logs, pd.DataFrame):
        return logs.copy()
    return pd.DataFrame(logs)


def cluster_error_logs(
    logs: Union[pd.DataFrame, List[dict]],
    eps: float = 0.5,
    min_samples: int = 2,
    max_features: int = 2000,
) -> Dict[int, List[dict]]:
    """
    Vectorize error_msg text with TF-IDF and cluster similar errors with
    DBSCAN (cosine distance, since TF-IDF vectors are direction-sensitive
    and vary in magnitude with message length).

    Only rows with a non-null, non-empty error_msg are considered. DBSCAN's
    noise label (-1) is excluded from the returned result, since those are
    one-off/dissimilar errors rather than a meaningful cluster.

    Parameters
    ----------
    logs : DataFrame or list of dicts with at least
        [job_id, service, error_msg, ...]
    eps : DBSCAN neighborhood radius (cosine distance, 0 = identical, 1 = orthogonal)
    min_samples : minimum cluster size for DBSCAN
    max_features : cap on TF-IDF vocabulary size

    Returns
    -------
    Dict[int, List[dict]] mapping cluster_id -> list of log records
    (as dicts) belonging to that cluster. Noise (-1) is excluded.
    """
    df = _to_dataframe(logs)
    error_df = df[df["error_msg"].notna() & (df["error_msg"].str.strip() != "")].copy()

    if error_df.empty:
        return {}

    error_df = error_df.reset_index(drop=True)

    vectorizer = TfidfVectorizer(
        max_features=max_features,
        stop_words="english",
        ngram_range=(1, 2),
    )
    tfidf_matrix = vectorizer.fit_transform(error_df["error_msg"])

    # Cosine distance requires the 'brute' algorithm in scikit-learn's DBSCAN.
    dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric="cosine", algorithm="brute")
    labels = dbscan.fit_predict(tfidf_matrix)
    error_df["cluster_id"] = labels

    clustered = error_df[error_df["cluster_id"] != -1]

    result: Dict[int, List[dict]] = {}
    for cluster_id, group in clustered.groupby("cluster_id"):
        cols = [c for c in LOG_FIELDS if c in group.columns]
        result[int(cluster_id)] = group[cols].to_dict(orient="records")

    logger.info(
        "Clustered %d error logs into %d clusters (%d noise points excluded).",
        len(error_df),
        len(result),
        int((labels == -1).sum()),
    )
    return result
