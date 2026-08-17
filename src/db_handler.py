"""
src/db_handler.py

Memgraph ingestion layer: loads log events as :Job nodes and derives
:FAILED_TOGETHER relationships between failed jobs that share a trace_id
and occurred within a configurable time window.

Uses gqlalchemy's Memgraph client over Bolt (default port 7687). Swapping
in the raw `neo4j` driver is a near 1:1 change, since Memgraph speaks Bolt.

Requires a running Memgraph instance, e.g.:
    docker run -p 7687:7687 memgraph/memgraph-platform
"""

import logging
from typing import List

from gqlalchemy import Memgraph

logger = logging.getLogger(__name__)


class MemgraphConnectionError(RuntimeError):
    """Raised when the handler cannot reach or operate on Memgraph."""


class MemgraphHandler:
    """Thin wrapper around a Memgraph connection for the observability
    pipeline's ingestion and correlation queries."""

    MERGE_JOB_QUERY = """
    MERGE (j:Job {job_id: $job_id})
    SET j.service    = $service,
        j.status      = $status,
        j.error_msg   = $error_msg,
        j.trace_id    = $trace_id,
        j.timestamp   = $timestamp,
        j.ts_unix     = localDateTime($timestamp).timestamp
    """

    # ts_unix is milliseconds (Memgraph's localDateTime().timestamp), so the
    # window bound below is converted from seconds to milliseconds by the
    # caller before being passed in as $window_ms.
    FAILED_TOGETHER_QUERY = """
    MATCH (a:Job {status: "FAILED"}), (b:Job {status: "FAILED"})
    WHERE a.trace_id = b.trace_id
      AND id(a) < id(b)
      AND abs(a.ts_unix - b.ts_unix) <= $window_ms
    MERGE (a)-[r:FAILED_TOGETHER]-(b)
    SET r.trace_id = a.trace_id,
        r.delta_seconds = abs(a.ts_unix - b.ts_unix) / 1000.0
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 7687):
        self.host = host
        self.port = port
        try:
            self.db = Memgraph(host=host, port=port)
        except Exception as exc:  # pragma: no cover - depends on live DB
            raise MemgraphConnectionError(
                f"Could not connect to Memgraph at {host}:{port}: {exc}"
            ) from exc

    def ensure_indexes(self) -> None:
        """Create indexes to speed up job lookups and the trace_id/status
        matching used by the correlation query."""
        try:
            self.db.execute("CREATE INDEX ON :Job(job_id);")
            self.db.execute("CREATE INDEX ON :Job(trace_id);")
            self.db.execute("CREATE INDEX ON :Job(status);")
        except Exception as exc:
            raise MemgraphConnectionError(f"Failed to create indexes: {exc}") from exc

    def load_logs(self, logs: List[dict], batch_size: int = 500) -> int:
        """MERGE a list of log event dicts into Memgraph as :Job nodes.
        MERGE (not CREATE) on job_id makes this idempotent -- re-running
        against the same data won't create duplicates."""
        count = 0
        batch: List[dict] = []

        try:
            for event in logs:
                batch.append(event)
                if len(batch) >= batch_size:
                    self._run_batch(batch)
                    count += len(batch)
                    batch = []

            if batch:
                self._run_batch(batch)
                count += len(batch)
        except Exception as exc:
            raise MemgraphConnectionError(f"Failed to load logs into Memgraph: {exc}") from exc

        logger.info("Loaded/merged %d :Job nodes.", count)
        return count

    def _run_batch(self, batch: List[dict]) -> None:
        for event in batch:
            self.db.execute(self.MERGE_JOB_QUERY, event)

    def create_failed_together_relationships(self, window_seconds: int = 10) -> int:
        """Create :FAILED_TOGETHER relationships between failed jobs that
        share a trace_id and occurred within `window_seconds` of each
        other. Returns the number of relationships created (undirected
        pairs, not double-counted)."""
        try:
            self.db.execute(self.FAILED_TOGETHER_QUERY, {"window_ms": window_seconds * 1000})
            return self.count_failed_together()
        except Exception as exc:
            raise MemgraphConnectionError(
                f"Failed to create FAILED_TOGETHER relationships: {exc}"
            ) from exc

    def count_failed_together(self) -> int:
        """Count :FAILED_TOGETHER relationships currently in the graph."""
        result = list(
            self.db.execute_and_fetch("MATCH ()-[r:FAILED_TOGETHER]-() RETURN count(r) AS rel_count;")
        )
        rel_count = result[0]["rel_count"] if result else 0
        # The undirected MATCH pattern counts each pair twice.
        return rel_count // 2
