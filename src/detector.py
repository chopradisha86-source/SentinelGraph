"""
src/detector.py

Rolling time-window failure-burst detection. Raises an alert whenever
failures for a given service exceed a threshold count within a trailing
time window (default: more than 5 failures in a 2-minute window).
"""

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Union

import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_FAILURE_STATUSES = ("FAILED", "TIMEOUT")


@dataclass
class FailureAlert:
    service: str
    triggered_at: pd.Timestamp
    window_failure_count: int
    window_seconds: int
    threshold: int
    trace_ids: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "service": self.service,
            "triggered_at": self.triggered_at.isoformat(),
            "window_failure_count": self.window_failure_count,
            "window_seconds": self.window_seconds,
            "threshold": self.threshold,
            "trace_ids": self.trace_ids,
        }


def _to_dataframe(logs: Union[pd.DataFrame, List[dict]]) -> pd.DataFrame:
    if isinstance(logs, pd.DataFrame):
        return logs.copy()
    return pd.DataFrame(logs)


def detect_failure_bursts(
    logs: Union[pd.DataFrame, List[dict]],
    window_seconds: int = 120,
    threshold: int = 5,
    failure_statuses: Sequence[str] = DEFAULT_FAILURE_STATUSES,
    services: Optional[Sequence[str]] = None,
) -> List[FailureAlert]:
    """
    Slide a trailing time window (default 2 minutes) over each service's
    failure events and raise an alert the moment the count of failures
    inside that window exceeds `threshold` (default: more than 5).

    An alert is generated at the *first* event that pushes a service's
    trailing-window failure count over the threshold, not for every
    subsequent event that stays over threshold -- this avoids alert spam
    for a single sustained outage.

    Parameters
    ----------
    logs : DataFrame or list of dicts with at least
        [service, status, timestamp, trace_id, job_id]
    window_seconds : trailing window size in seconds
    threshold : alert fires when window_failure_count > threshold
    failure_statuses : which `status` values count as a failure
    services : optional subset of services to check; defaults to all present

    Returns
    -------
    List[FailureAlert], ordered by triggered_at.
    """
    df = _to_dataframe(logs)
    if df.empty:
        return []

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["is_failure"] = df["status"].isin(failure_statuses)

    target_services = services if services is not None else df["service"].unique()

    window = f"{window_seconds}s"
    alerts: List[FailureAlert] = []

    for service in target_services:
        svc_df = df[df["service"] == service].sort_values("timestamp")
        failures = svc_df[svc_df["is_failure"]].set_index("timestamp")
        if failures.empty:
            continue

        # Trailing rolling count of failures within the time window,
        # evaluated at each failure event's own timestamp.
        rolling_count = failures["job_id"].rolling(window).count()

        was_over = False
        for ts, count in rolling_count.items():
            count = int(count)
            is_over = count > threshold
            if is_over and not was_over:
                trace_ids = (
                    failures.loc[
                        (failures.index > ts - pd.Timedelta(seconds=window_seconds))
                        & (failures.index <= ts),
                        "trace_id",
                    ]
                    .dropna()
                    .unique()
                    .tolist()
                )
                alerts.append(
                    FailureAlert(
                        service=service,
                        triggered_at=ts,
                        window_failure_count=count,
                        window_seconds=window_seconds,
                        threshold=threshold,
                        trace_ids=trace_ids,
                    )
                )
            was_over = is_over

    alerts.sort(key=lambda a: a.triggered_at)
    return alerts
