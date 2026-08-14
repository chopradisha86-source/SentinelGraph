"""
src/generator.py

Synthetic log event generation for the observability pipeline.

Provides two modes:
- Batch generation (generate_logs): simulates distributed traces hopping
  across services, for offline/CLI use (main.py).
- Streaming generation (generate_single_event): one event at a time with
  a rolling trace-id pool, for live use (app.py).

All events share a single canonical schema:
    job_id, service, status, error_msg, trace_id, timestamp
"""

import json
import logging
import random
import uuid
from collections import deque
from datetime import datetime, timedelta, timezone
from typing import Deque, List, Optional

from faker import Faker

logger = logging.getLogger(__name__)

fake = Faker()

SERVICES = ["auth", "payment", "inventory", "shipping"]

STATUSES = ["SUCCESS", "FAILED", "TIMEOUT", "RETRY"]
# Weighted toward success, but with enough failures for the detector and
# clusterer downstream to have meaningful data to work with.
STATUS_WEIGHTS = [0.70, 0.18, 0.07, 0.05]

FAILURE_STATUSES = ("FAILED", "TIMEOUT")

ERROR_MESSAGES_BY_SERVICE = {
    "auth": [
        "Invalid credentials",
        "Token expired",
        "JWT signature verification failed",
        "User session not found",
        "Rate limit exceeded on login endpoint",
    ],
    "payment": [
        "Card declined by issuer",
        "Payment gateway timeout",
        "Insufficient funds",
        "Currency conversion service unavailable",
        "Duplicate transaction detected",
    ],
    "inventory": [
        "SKU not found in warehouse",
        "Stock reservation conflict",
        "Inventory database connection pool exhausted",
        "Negative stock count detected",
        "Warehouse sync lag exceeded threshold",
    ],
    "shipping": [
        "Carrier API unreachable",
        "Invalid shipping address",
        "Label generation failed",
        "Rate calculation service timeout",
        "Package weight exceeds carrier limit",
    ],
}


def _random_status() -> str:
    return random.choices(STATUSES, weights=STATUS_WEIGHTS, k=1)[0]


def _error_msg_for(service: str, status: str) -> Optional[str]:
    if status in FAILURE_STATUSES:
        return random.choice(ERROR_MESSAGES_BY_SERVICE[service])
    return None


# --------------------------------------------------------------------------
# Batch generation (used by main.py)
# --------------------------------------------------------------------------

def generate_event(base_time: datetime, trace_id: str, service: str) -> dict:
    """Generate a single synthetic log event belonging to a known trace."""
    status = _random_status()
    error_msg = _error_msg_for(service, status)

    # Small random jitter so events in the same trace are close in time
    # but not identical -- this is what the correlation/window logic
    # downstream (db_handler, detector) relies on.
    jitter = timedelta(seconds=random.uniform(0, 15))
    timestamp = base_time + jitter

    return {
        "job_id": str(uuid.uuid4()),
        "service": service,
        "status": status,
        "error_msg": error_msg,
        "trace_id": trace_id,
        "timestamp": timestamp.isoformat(),
    }


def generate_trace_events(base_time: datetime) -> List[dict]:
    """Simulate one distributed request hopping through a random subset
    of services (e.g. auth -> payment -> inventory -> shipping)."""
    trace_id = str(uuid.uuid4())
    hop_count = random.randint(1, len(SERVICES))
    services_in_trace = random.sample(SERVICES, hop_count)

    events = []
    current_time = base_time
    for service in services_in_trace:
        events.append(generate_event(current_time, trace_id, service))
        current_time += timedelta(seconds=random.uniform(0, 5))
    return events


def generate_logs(count: int, start_time: Optional[datetime] = None) -> List[dict]:
    """Generate `count` log events across randomly generated traces,
    sorted chronologically."""
    if start_time is None:
        start_time = datetime.now(timezone.utc) - timedelta(hours=1)

    logs: List[dict] = []
    while len(logs) < count:
        offset = timedelta(seconds=random.uniform(0, 3600))
        trace_base_time = start_time + offset
        logs.extend(generate_trace_events(trace_base_time))

    logs = logs[:count]
    logs.sort(key=lambda e: e["timestamp"])
    return logs


def write_ndjson(logs: List[dict], path: str) -> None:
    """Write a list of log dicts to a newline-delimited JSON file."""
    with open(path, "w") as f:
        for event in logs:
            f.write(json.dumps(event) + "\n")
    logger.info("Wrote %d events to %s", len(logs), path)


def load_ndjson(path: str) -> List[dict]:
    """Read a newline-delimited JSON log file back into a list of dicts."""
    logs = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                logs.append(json.loads(line))
    return logs


# --------------------------------------------------------------------------
# Streaming generation (used by app.py)
# --------------------------------------------------------------------------

# Rolling pool of recent trace_ids so consecutive streamed events
# occasionally share a trace, simulating hops within one live request.
_recent_trace_ids: Deque[str] = deque(maxlen=8)


def generate_single_event() -> dict:
    """Generate one mock log event timestamped at call time, for use in a
    live streaming context (e.g. the WebSocket background task)."""
    service = random.choice(SERVICES)
    status = _random_status()
    error_msg = _error_msg_for(service, status)

    if _recent_trace_ids and random.random() < 0.3:
        trace_id = random.choice(_recent_trace_ids)
    else:
        trace_id = str(uuid.uuid4())
        _recent_trace_ids.append(trace_id)

    return {
        "job_id": str(uuid.uuid4()),
        "service": service,
        "status": status,
        "error_msg": error_msg,
        "trace_id": trace_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
