"""
main.py

End-to-end CLI batch runner for the observability pipeline:

    generator -> db_handler (Memgraph) -> detector -> clusterer -> ai_synthesis

Generates synthetic logs, optionally ingests them into Memgraph and derives
FAILED_TOGETHER relationships, runs rolling-window failure-burst detection,
clusters error messages, and (optionally) generates an AI incident report
per cluster. All outputs are written to an output directory.

Usage:
    python main.py --count 3000 --out-dir output/

    # Skip Memgraph ingestion (e.g. no local instance running):
    python main.py --count 3000 --skip-db

    # Skip AI report generation (e.g. no Gemini API key configured):
    python main.py --count 3000 --skip-ai

Memgraph connection defaults to $MEMGRAPH_HOST / $MEMGRAPH_PORT if set
(falling back to 127.0.0.1:7687), so no flags are needed when running
inside Docker Compose alongside a 'memgraph' service.
"""

import argparse
import json
import logging
import os
import sys

from dotenv import load_dotenv
load_dotenv()
from src import ai_synthesis, clusterer, detector, generator
from src.db_handler import MemgraphConnectionError, MemgraphHandler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("main")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the observability pipeline end to end.")
    parser.add_argument("--count", type=int, default=2000, help="Number of log events to generate.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility.")
    parser.add_argument("--out-dir", type=str, default="output", help="Directory for all output files.")

    parser.add_argument("--skip-db", action="store_true", help="Skip Memgraph ingestion.")
    parser.add_argument(
        "--db-host", type=str, default=os.environ.get("MEMGRAPH_HOST", "127.0.0.1"),
        help="Memgraph host. Defaults to $MEMGRAPH_HOST, then 127.0.0.1 "
             "(use the 'memgraph' service name when running via Docker Compose).",
    )
    parser.add_argument(
        "--db-port", type=int, default=int(os.environ.get("MEMGRAPH_PORT", 7687)),
        help="Memgraph Bolt port. Defaults to $MEMGRAPH_PORT, then 7687.",
    )
    parser.add_argument("--correlation-window", type=int, default=10,
                         help="FAILED_TOGETHER correlation window, in seconds.")

    parser.add_argument("--burst-window", type=int, default=120,
                         help="Failure-burst rolling window, in seconds.")
    parser.add_argument("--burst-threshold", type=int, default=5,
                         help="Failure count threshold (alert fires when exceeded).")

    parser.add_argument("--eps", type=float, default=0.5, help="DBSCAN cosine-distance eps.")
    parser.add_argument("--min-samples", type=int, default=3, help="DBSCAN min_samples.")

    parser.add_argument("--skip-ai", action="store_true", help="Skip AI incident report generation.")
    parser.add_argument("--gemini-model", type=str, default=ai_synthesis.DEFAULT_MODEL, help="Gemini model name.")
    parser.add_argument("--max-reports", type=int, default=5,
                         help="Max number of clusters to generate AI reports for (largest first).")

    return parser.parse_args()


def step_generate(args: argparse.Namespace, out_dir: str) -> list:
    logger.info("Step 1/5: Generating %d synthetic log events...", args.count)
    logs = generator.generate_logs(args.count)
    log_path = os.path.join(out_dir, "logs.jsonl")
    generator.write_ndjson(logs, log_path)
    failed = sum(1 for e in logs if e["status"] in generator.FAILURE_STATUSES)
    logger.info("Generated %d events (%d failed/timeout).", len(logs), failed)
    return logs


def step_ingest(args: argparse.Namespace, logs: list) -> None:
    if args.skip_db:
        logger.info("Step 2/5: Skipping Memgraph ingestion (--skip-db).")
        return

    logger.info("Step 2/5: Ingesting logs into Memgraph at %s:%d...", args.db_host, args.db_port)
    try:
        handler = MemgraphHandler(host=args.db_host, port=args.db_port)
        handler.ensure_indexes()
        handler.load_logs(logs)
        rel_count = handler.create_failed_together_relationships(window_seconds=args.correlation_window)
        logger.info("Created %d FAILED_TOGETHER relationships.", rel_count)
    except MemgraphConnectionError as exc:
        logger.warning("Memgraph ingestion skipped due to connection error: %s", exc)
        logger.warning("Continuing pipeline without graph ingestion.")


def step_detect(args: argparse.Namespace, logs: list, out_dir: str) -> list:
    logger.info(
        "Step 3/5: Detecting failure bursts (>%d failures / %ds window)...",
        args.burst_threshold, args.burst_window,
    )
    alerts = detector.detect_failure_bursts(
        logs, window_seconds=args.burst_window, threshold=args.burst_threshold,
    )
    alerts_path = os.path.join(out_dir, "alerts.json")
    with open(alerts_path, "w") as f:
        json.dump([a.to_dict() for a in alerts], f, indent=2)
    logger.info("Found %d failure-burst alerts. Written to %s", len(alerts), alerts_path)
    return alerts


def step_cluster(args: argparse.Namespace, logs: list, out_dir: str) -> dict:
    logger.info("Step 4/5: Clustering error logs (eps=%.2f, min_samples=%d)...", args.eps, args.min_samples)
    clusters = clusterer.cluster_error_logs(logs, eps=args.eps, min_samples=args.min_samples)
    clusters_path = os.path.join(out_dir, "clusters.json")
    with open(clusters_path, "w") as f:
        json.dump(clusters, f, indent=2)
    logger.info("Found %d clusters. Written to %s", len(clusters), clusters_path)
    return clusters


def step_synthesize(args: argparse.Namespace, clusters: dict, out_dir: str) -> None:
    if args.skip_ai:
        logger.info("Step 5/5: Skipping AI incident report generation (--skip-ai).")
        return

    if not clusters:
        logger.info("Step 5/5: No clusters to report on.")
        return

    logger.info("Step 5/5: Generating AI incident reports...")
    reports_dir = os.path.join(out_dir, "incident_reports")
    os.makedirs(reports_dir, exist_ok=True)

    # Largest clusters first -- most likely to represent a real incident
    # worth an SRE's attention.
    ranked_clusters = sorted(clusters.items(), key=lambda kv: len(kv[1]), reverse=True)
    ranked_clusters = ranked_clusters[: args.max_reports]

    for cluster_id, cluster_logs in ranked_clusters:
        try:
            report = ai_synthesis.generate_incident_report(
                cluster_logs, model=args.gemini_model,
            )
        except ValueError as exc:
            logger.warning("Skipping AI report for cluster %s: %s", cluster_id, exc)
            continue
        except Exception as exc:  # network/API errors shouldn't kill the run
            logger.warning("AI report generation failed for cluster %s: %s", cluster_id, exc)
            continue

        report_path = os.path.join(reports_dir, f"cluster_{cluster_id}.md")
        with open(report_path, "w") as f:
            f.write(report)
        logger.info("Wrote incident report for cluster %s to %s", cluster_id, report_path)


def main() -> None:
    args = parse_args()

    if args.seed is not None:
        import random
        from faker import Faker
        random.seed(args.seed)
        Faker.seed(args.seed)

    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    logs = step_generate(args, out_dir)
    step_ingest(args, logs)
    step_detect(args, logs, out_dir)
    clusters = step_cluster(args, logs, out_dir)
    step_synthesize(args, clusters, out_dir)

    logger.info("Pipeline complete. All outputs written to %s/", out_dir)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(1)
