"""
eval_clustering.py

Measures clustering quality for src/clusterer.py against ground truth:
generator.py's ERROR_MESSAGES_BY_SERVICE defines exactly 20 distinct known
error types. This script generates a batch of synthetic logs, runs
cluster_error_logs(), and checks -- per known error type -- whether all its
occurrences landed in one clean cluster (correct), got split across
multiple clusters (over-segmentation), or got merged with a DIFFERENT
error type in the same cluster (false merge -- the worse failure mode).

Usage:
    python eval_clustering.py --count 4000 --eps 0.5 --min-samples 3
    python eval_clustering.py --sweep   # eps sensitivity sweep, 0.3-0.8
"""

import argparse
from collections import defaultdict

from src import generator
from src.clusterer import cluster_error_logs


def run_eval(count: int, eps: float, min_samples: int, seed: int = None, verbose: bool = True) -> dict:
    if seed is not None:
        import random
        from faker import Faker
        random.seed(seed)
        Faker.seed(seed)

    logs = generator.generate_logs(count)
    clusters = cluster_error_logs(logs, eps=eps, min_samples=min_samples)

    # Ground truth: exact error_msg string -> which cluster_id(s) it appears in.
    truth_to_clusters = defaultdict(set)
    cluster_to_truths = defaultdict(set)
    total_error_logs = 0

    for cluster_id, entries in clusters.items():
        for entry in entries:
            msg = entry["error_msg"]
            truth_to_clusters[msg].add(cluster_id)
            cluster_to_truths[cluster_id].add(msg)
            total_error_logs += 1

    all_known_types = {msg for msgs in generator.ERROR_MESSAGES_BY_SERVICE.values() for msg in msgs}
    seen_types = set(truth_to_clusters.keys())
    missing_types = all_known_types - seen_types  # didn't appear at all, or all noise

    correctly_separated = 0
    split_types = []       # one error type spread across >1 cluster
    merged_clusters = []   # one cluster containing >1 error type

    for msg in seen_types:
        cluster_ids = truth_to_clusters[msg]
        if len(cluster_ids) == 1:
            cid = next(iter(cluster_ids))
            if len(cluster_to_truths[cid]) == 1:
                correctly_separated += 1
            else:
                merged_clusters.append((msg, cid, cluster_to_truths[cid]))
        else:
            split_types.append((msg, cluster_ids))

    result = {
        "count_generated": count,
        "eps": eps,
        "min_samples": min_samples,
        "total_error_logs_clustered": total_error_logs,
        "num_clusters_found": len(clusters),
        "known_types_total": len(all_known_types),
        "known_types_seen": len(seen_types),
        "known_types_missing": sorted(missing_types),
        "correctly_separated": correctly_separated,
        "split_types": split_types,
        "merged_clusters": merged_clusters,
    }

    if verbose:
        print(f"\n=== eps={eps}, min_samples={min_samples}, count={count} ===")
        print(f"Error logs clustered: {total_error_logs}")
        print(f"Clusters found: {len(clusters)}")
        print(f"Known error types seen in this batch: {len(seen_types)}/{len(all_known_types)}")
        print(f"Correctly separated (1 type <-> 1 clean cluster): {correctly_separated}/{len(seen_types)}")
        if split_types:
            print(f"Split across multiple clusters ({len(split_types)}):")
            for msg, cids in split_types:
                print(f"  - {msg!r} -> clusters {sorted(cids)}")
        if merged_clusters:
            print(f"FALSE MERGES ({len(merged_clusters)}) -- different error types in same cluster:")
            for msg, cid, others in merged_clusters:
                print(f"  - cluster {cid} contains {sorted(others)}")
        if missing_types:
            print(f"Known types not clustered at all (too rare / all noise) ({len(missing_types)}):")
            for msg in sorted(missing_types):
                print(f"  - {msg!r}")

    return result


def main():
    parser = argparse.ArgumentParser(description="Evaluate clustering quality against known error types.")
    parser.add_argument("--count", type=int, default=4000)
    parser.add_argument("--eps", type=float, default=0.5)
    parser.add_argument("--min-samples", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sweep", action="store_true", help="Run an eps sensitivity sweep instead of a single run.")
    args = parser.parse_args()

    if args.sweep:
        print(f"{'eps':>5} | {'clusters':>8} | {'seen':>6} | {'correct':>7} | {'split':>5} | {'merged':>6}")
        print("-" * 55)
        for eps in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
            r = run_eval(args.count, eps, args.min_samples, seed=args.seed, verbose=False)
            print(f"{eps:>5} | {r['num_clusters_found']:>8} | {r['known_types_seen']:>6} | "
                  f"{r['correctly_separated']:>7} | {len(r['split_types']):>5} | {len(r['merged_clusters']):>6}")
    else:
        run_eval(args.count, args.eps, args.min_samples, seed=args.seed, verbose=True)


if __name__ == "__main__":
    main()