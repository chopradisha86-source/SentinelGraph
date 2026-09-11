"""
eval_clustering_hard.py

The basic evaluation (eval_clustering.py) hits 20/20 perfect separation at
every eps from 0.3-0.8 -- but that's because the 20 known error messages
share almost no vocabulary, making them trivially easy for TF-IDF to tell
apart. This script tests the harder, more realistic case: multiple
DIFFERENT PHRASINGS of the SAME underlying failure, which is what real
production logs actually look like (different services/versions/authors
wording the same error differently).

Ground truth here is a semantic group ID (not the exact string), e.g. all
phrasings of "payment gateway timed out" share one group ID even though
the wording differs. This directly tests whether lexical (TF-IDF) clustering
can still find the right groups once wording diverges.
"""

import argparse
import random
import uuid
from datetime import datetime, timedelta, timezone
from collections import defaultdict

from faker import Faker
from src.clusterer import cluster_error_logs

# Each group = paraphrased variants of the SAME underlying failure.
# This is the harder ground truth: group ID, not exact string.
SEMANTIC_GROUPS = {
    "payment_gateway_timeout": [
        "Payment gateway timeout",
        "Timed out while contacting the payment gateway",
        "Gateway did not respond within the timeout window",
        "Payment provider request exceeded time limit",
    ],
    "db_pool_exhausted": [
        "Inventory database connection pool exhausted",
        "No available connections in the database pool",
        "Connection pool limit reached for inventory DB",
        "Database pool saturated, unable to acquire connection",
    ],
    "auth_token_expired": [
        "Token expired",
        "Authentication token has expired",
        "JWT token is no longer valid (expired)",
        "Session token expired, re-authentication required",
    ],
    "carrier_unreachable": [
        "Carrier API unreachable",
        "Unable to reach shipping carrier API",
        "Carrier service connection failed",
        "Shipping provider API is not responding",
    ],
    "card_declined": [
        "Card declined by issuer",
        "Payment card was declined by the issuing bank",
        "Card issuer rejected the transaction",
        "Transaction declined: card issuer refused payment",
    ],
}

SERVICES = ["auth", "payment", "inventory", "shipping"]


def generate_hard_logs(count: int) -> list:
    logs = []
    group_names = list(SEMANTIC_GROUPS.keys())
    base_time = datetime.now(timezone.utc) - timedelta(hours=1)

    for _ in range(count):
        group = random.choice(group_names)
        msg = random.choice(SEMANTIC_GROUPS[group])
        service = random.choice(SERVICES)
        ts = base_time + timedelta(seconds=random.uniform(0, 3600))
        logs.append({
            "job_id": str(uuid.uuid4()),
            "service": service,
            "status": "FAILED",
            "error_msg": msg,
            "trace_id": str(uuid.uuid4()),
            "timestamp": ts.isoformat(),
            "_semantic_group": group,  # ground truth, stripped before clustering
        })
    return logs


def run_hard_eval(count: int, eps: float, min_samples: int, seed: int = 42):
    random.seed(seed)
    Faker.seed(seed)

    logs = generate_hard_logs(count)
    ground_truth = {log["job_id"]: log["_semantic_group"] for log in logs}
    # cluster_error_logs doesn't need/use the extra field, but strip it to
    # mirror the real pipeline's actual log schema.
    clean_logs = [{k: v for k, v in log.items() if k != "_semantic_group"} for log in logs]

    clusters = cluster_error_logs(clean_logs, eps=eps, min_samples=min_samples)

    group_to_clusters = defaultdict(set)
    cluster_to_groups = defaultdict(set)
    total_clustered = 0

    for cluster_id, entries in clusters.items():
        for entry in entries:
            group = ground_truth[entry["job_id"]]
            group_to_clusters[group].add(cluster_id)
            cluster_to_groups[cluster_id].add(group)
            total_clustered += 1

    correctly_unified = 0   # all variants of one semantic group in ONE cluster, that cluster pure
    fragmented = []          # one semantic group split across multiple clusters (paraphrase not recognized)
    merged = []               # one cluster contains multiple DIFFERENT semantic groups

    for group, cids in group_to_clusters.items():
        if len(cids) == 1:
            cid = next(iter(cids))
            if len(cluster_to_groups[cid]) == 1:
                correctly_unified += 1
            else:
                merged.append((group, cid, cluster_to_groups[cid]))
        else:
            fragmented.append((group, cids))

    print(f"\n=== HARD eval (paraphrased variants): eps={eps}, min_samples={min_samples}, count={count} ===")
    print(f"Error logs clustered: {total_clustered} (of {count} generated)")
    print(f"Clusters found: {len(clusters)}  (ideal: {len(SEMANTIC_GROUPS)}, since there are "
          f"{len(SEMANTIC_GROUPS)} underlying failure types)")
    print(f"Semantic groups correctly unified into one clean cluster: {correctly_unified}/{len(group_to_clusters)}")
    if fragmented:
        print(f"Fragmented (paraphrases NOT recognized as same failure) ({len(fragmented)}):")
        for group, cids in fragmented:
            print(f"  - {group!r} split across clusters {sorted(cids)}")
    if merged:
        print(f"False merges ({len(merged)}):")
        for group, cid, others in merged:
            print(f"  - cluster {cid} mixes groups {sorted(others)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=2000)
    parser.add_argument("--eps", type=float, default=0.5)
    parser.add_argument("--min-samples", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run_hard_eval(args.count, args.eps, args.min_samples, seed=args.seed)


if __name__ == "__main__":
    main()