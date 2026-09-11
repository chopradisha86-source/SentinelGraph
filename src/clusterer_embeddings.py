"""
src/clusterer_embeddings.py

Semantic-embedding variant of the TF-IDF + DBSCAN clusterer in
src/clusterer.py. Same interface (cluster_error_logs), same DBSCAN
clustering step, but text is embedded with a sentence-transformer model
instead of TF-IDF -- so similarity is based on meaning, not word overlap.

This is a standalone addition (does not modify src/clusterer.py) so the
two approaches can be compared directly on the same evaluation harness.

Install:
    pip install sentence-transformers
"""

import logging
from typing import Dict, List, Union

import pandas as pd
from sklearn.cluster import DBSCAN

logger = logging.getLogger(__name__)

LOG_FIELDS = ["job_id", "service", "status", "error_msg", "trace_id", "timestamp"]

_model = None  # lazy-loaded, so importing this module doesn't force a model download


def _get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


def _to_dataframe(logs: Union[pd.DataFrame, List[dict]]) -> pd.DataFrame:
    if isinstance(logs, pd.DataFrame):
        return logs.copy()
    return pd.DataFrame(logs)


def cluster_error_logs_semantic(
    logs: Union[pd.DataFrame, List[dict]],
    eps: float = 0.3,
    min_samples: int = 2,
) -> Dict[int, List[dict]]:
    """
    Same contract as clusterer.cluster_error_logs(), but embeds error_msg
    text with a sentence-transformer model (semantic similarity) instead
    of TF-IDF (lexical/word-overlap similarity), then clusters with DBSCAN
    (cosine distance).

    Note: eps has a different effective scale here than in the TF-IDF
    version -- sentence embeddings are much more tightly clustered by
    meaning, so a smaller default eps (0.3 vs 0.5) is typically needed.
    """
    df = _to_dataframe(logs)
    error_df = df[df["error_msg"].notna() & (df["error_msg"].str.strip() != "")].copy()

    if error_df.empty:
        return {}

    error_df = error_df.reset_index(drop=True)

    model = _get_model()
    embeddings = model.encode(error_df["error_msg"].tolist(), show_progress_bar=False)

    dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric="cosine", algorithm="brute")
    labels = dbscan.fit_predict(embeddings)
    error_df["cluster_id"] = labels

    clustered = error_df[error_df["cluster_id"] != -1]

    result: Dict[int, List[dict]] = {}
    for cluster_id, group in clustered.groupby("cluster_id"):
        cols = [c for c in LOG_FIELDS if c in group.columns]
        result[int(cluster_id)] = group[cols].to_dict(orient="records")

    logger.info(
        "Semantic-clustered %d error logs into %d clusters (%d noise points excluded).",
        len(error_df), len(result), int((labels == -1).sum()),
    )
    return result