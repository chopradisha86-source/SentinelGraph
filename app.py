"""
app.py

FastAPI application that streams mock microservice log events over a
WebSocket, evaluating each new event against the pipeline's failure-burst
detector, periodically re-clustering recent error logs, and pushing JSON
payloads to every connected client. Clients can also request an on-demand
AI-generated incident report for a specific cluster.

Run:
    uvicorn app:app --reload

Connect to ws://localhost:8000/ws/logs. Message types pushed by the server:

    {"type": "log_event", "log": {...}, "alerts": [...]}
        -- sent every tick (STREAM_INTERVAL_SECONDS). "alerts" is usually
           empty; it's populated only when a new failure-burst alert fires.

    {"type": "cluster_update", "clusters": {"<cluster_id>": <count>, ...}}
        -- sent every CLUSTER_INTERVAL_TICKS ticks, summarizing current
           error clusters (noise excluded).

Clients can send a JSON command to request an AI incident report:

    {"action": "generate_report", "cluster_id": 0}

The server responds (to that client only) with:

    {"type": "incident_report", "cluster_id": 0, "report": "..."}
    or
    {"type": "error", "message": "..."}
"""

import asyncio
import json
import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from typing import Deque, Dict, List, Set
from collections import deque

from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from src import ai_synthesis, clusterer, detector, generator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("app")

# -------------------------------------------------------------------------
# Connection management
# --------------------------------------------------------------------------

class ConnectionManager:
    """Tracks active WebSocket clients and broadcasts messages to all of them."""

    def __init__(self) -> None:
        self.active_connections: Set[WebSocket] = set()
        self._lock = asyncio.Lock()

    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        async with self._lock:
            self.active_connections.add(websocket)

    async def disconnect(self, websocket: WebSocket) -> None:
        async with self._lock:
            self.active_connections.discard(websocket)

    async def broadcast(self, message: str) -> None:
        """Send `message` to every connected client, dropping any that
        have gone away without breaking the loop for the rest."""
        async with self._lock:
            connections = list(self.active_connections)

        dead: List[WebSocket] = []
        for connection in connections:
            try:
                await connection.send_text(message)
            except Exception:
                dead.append(connection)

        if dead:
            async with self._lock:
                for connection in dead:
                    self.active_connections.discard(connection)


manager = ConnectionManager()

# --------------------------------------------------------------------------
# Pipeline evaluation state
# --------------------------------------------------------------------------

BURST_WINDOW_SECONDS = int(os.environ.get("BURST_WINDOW_SECONDS", 120))
BURST_THRESHOLD = int(os.environ.get("BURST_THRESHOLD", 5))
BUFFER_RETENTION_SECONDS = BURST_WINDOW_SECONDS * 2
STREAM_INTERVAL_SECONDS = float(os.environ.get("STREAM_INTERVAL_SECONDS", 1.0))
CLUSTER_INTERVAL_TICKS = int(os.environ.get("CLUSTER_INTERVAL_TICKS", 30))
CLUSTER_EPS = float(os.environ.get("CLUSTER_EPS", 0.5))
CLUSTER_MIN_SAMPLES = int(os.environ.get("CLUSTER_MIN_SAMPLES", 3))

log_buffer: Deque[dict] = deque()
# Tracks (service, triggered_at_iso) pairs already broadcast so the same
# alert isn't re-sent every tick while it remains within the window.
sent_alert_keys: Set[tuple] = set()
# Most recently computed clusters, kept so on-demand report requests don't
# need to recompute clustering synchronously inside the request handler.
latest_clusters: Dict[int, List[dict]] = {}


def _trim_buffer(now: datetime) -> None:
    cutoff = now - timedelta(seconds=BUFFER_RETENTION_SECONDS)
    while log_buffer and datetime.fromisoformat(log_buffer[0]["timestamp"]) < cutoff:
        log_buffer.popleft()


def _evaluate_bursts() -> List[dict]:
    """Run failure-burst detection over the current buffer and return only
    alerts that haven't already been broadcast."""
    if not log_buffer:
        return []

    alerts = detector.detect_failure_bursts(
        list(log_buffer),
        window_seconds=BURST_WINDOW_SECONDS,
        threshold=BURST_THRESHOLD,
    )

    new_alerts = []
    for alert in alerts:
        key = (alert.service, alert.triggered_at.isoformat())
        if key not in sent_alert_keys:
            sent_alert_keys.add(key)
            new_alerts.append(alert.to_dict())

    return new_alerts


def _refresh_clusters() -> Dict[int, List[dict]]:
    """Re-cluster the current buffer's error logs and cache the result."""
    global latest_clusters
    if not log_buffer:
        latest_clusters = {}
        return latest_clusters

    latest_clusters = clusterer.cluster_error_logs(
        list(log_buffer), eps=CLUSTER_EPS, min_samples=CLUSTER_MIN_SAMPLES,
    )
    return latest_clusters


async def log_stream_task() -> None:
    """Background loop: generate a log event every tick, evaluate the
    pipeline, and broadcast raw log + any new alerts to all clients.
    Periodically re-clusters and broadcasts a cluster summary too."""
    tick = 0
    while True:
        event = generator.generate_single_event()
        log_buffer.append(event)
        _trim_buffer(datetime.fromisoformat(event["timestamp"]))

        new_alerts = _evaluate_bursts()

        await manager.broadcast(json.dumps({
            "type": "log_event",
            "log": event,
            "alerts": new_alerts,
        }))

        tick += 1
        if tick % CLUSTER_INTERVAL_TICKS == 0:
            clusters = _refresh_clusters()
            summary = {str(cid): len(entries) for cid, entries in clusters.items()}
            await manager.broadcast(json.dumps({
                "type": "cluster_update",
                "clusters": summary,
            }))

        await asyncio.sleep(STREAM_INTERVAL_SECONDS)


async def _handle_client_command(websocket: WebSocket, raw_message: str) -> None:
    """Handle an inbound JSON command from a client, e.g. an on-demand
    AI incident report request for a specific cluster."""
    try:
        command = json.loads(raw_message)
    except json.JSONDecodeError:
        await websocket.send_text(json.dumps({"type": "error", "message": "Invalid JSON."}))
        return

    action = command.get("action")
    if action != "generate_report":
        await websocket.send_text(json.dumps({"type": "error", "message": f"Unknown action: {action}"}))
        return

    cluster_id = command.get("cluster_id")
    cluster_logs = latest_clusters.get(cluster_id)
    if cluster_logs is None:
        # Cluster IDs may arrive as strings over JSON.
        try:
            cluster_logs = latest_clusters.get(int(cluster_id))
        except (TypeError, ValueError):
            cluster_logs = None

    if not cluster_logs:
        await websocket.send_text(json.dumps({
            "type": "error",
            "message": f"No cluster found with id {cluster_id}. Available: {list(latest_clusters.keys())}",
        }))
        return

    try:
        # generate_incident_report performs blocking network I/O; run it in
        # a worker thread so it doesn't stall the event loop / other clients.
        report = await asyncio.to_thread(ai_synthesis.generate_incident_report, cluster_logs)
        await websocket.send_text(json.dumps({
            "type": "incident_report",
            "cluster_id": cluster_id,
            "report": report,
        }))
    except Exception as exc:
        logger.warning("Incident report generation failed: %s", exc)
        await websocket.send_text(json.dumps({
            "type": "error",
            "message": f"Failed to generate incident report: {exc}",
        }))


# --------------------------------------------------------------------------
# FastAPI app
# --------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    task = asyncio.create_task(log_stream_task())
    logger.info("Log stream background task started.")
    try:
        yield
    finally:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        logger.info("Log stream background task stopped.")


app = FastAPI(title="Microservice Observability Stream", lifespan=lifespan)


@app.get("/")
async def root():
    return {
        "status": "ok",
        "websocket_endpoint": "/ws/logs",
        "active_connections": len(manager.active_connections),
        "buffered_logs": len(log_buffer),
        "latest_cluster_count": len(latest_clusters),
    }


@app.websocket("/ws/logs")
async def websocket_logs(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            raw_message = await websocket.receive_text()
            await _handle_client_command(websocket, raw_message)
    except WebSocketDisconnect:
        pass
    finally:
        await manager.disconnect(websocket)
