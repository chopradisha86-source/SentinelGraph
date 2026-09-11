"""
eval_clustering_embeddings.py

Runs the same hard paraphrase test as eval_clustering_hard.py, but using
semantic embeddings (src/clusterer_embeddings.py) instead of TF-IDF
(src/clusterer.py), to directly measure whether embeddings actually fix
the limitation found earlier: TF-IDF couldn't unify paraphrased variants
of the same underlying failure without also false-merging distinct ones.

Usage:
    python eval_clustering_embeddings.py --count 2000 --eps 0.3 --min-samples 3
    python eval_clustering_embeddings.py --sweep   # eps sweep for the embedding model
"""

import argparse
import random
from collections import defaultdict

from faker import Faker

from eval_clustering_hard import SEMANTIC_GROUPS, generate_hard_logs
from src.clusterer_embeddings import cluster_error_logs_semantic


def run_semantic_eval(count: int, eps: float, min_samples: int, seed: int = 42, verbose: bool = True) -> dict:
    random.seed(seed)
    Faker.seed(seed)

    logs = generate_hard_logs(count)
    ground_truth = {log["job_id"]: log["_semantic_group"] for log in logs}
    clean_logs = [{k: v for k, v in log.items() if k != "_semantic_group"} for log in logs]

    clusters = cluster_error_logs_semantic(clean_logs, eps=eps, min_samples=min_samples)

    group_to_clusters = defaultdict(set)
    cluster_to_groups = defaultdict(set)
    total_clustered = 0

    for cluster_id, entries in clusters.items():
        for entry in entries:
            group = ground_truth[entry["job_id"]]
            group_to_clusters[group].add(cluster_id)
            cluster_to_groups[cluster_id].add(group)
            total_clustered += 1

    correctly_unified = 0
    fragmented = []
    merged = []

    for group, cids in group_to_clusters.items():
        if len(cids) == 1:
            cid = next(iter(cids))
            if len(cluster_to_groups[cid]) == 1:
                correctly_unified += 1
            else:
                merged.append((group, cid, cluster_to_groups[cid]))
        else:
            fragmented.append((group, cids))

    if verbose:
        print(f"\n=== SEMANTIC eval (embeddings): eps={eps}, min_samples={min_samples}, count={count} ===")
        print(f"Error logs clustered: {total_clustered} (of {count} generated)")
        print(f"Clusters found: {len(clusters)}  (ideal: {len(SEMANTIC_GROUPS)})")
        print(f"Semantic groups correctly unified into one clean cluster: {correctly_unified}/{len(group_to_clusters)}")
        if fragmented:
            print(f"Fragmented ({len(fragmented)}):")
            for group, cids in fragmented:
                print(f"  - {group!r} split across clusters {sorted(cids)}")
        if merged:
            print(f"False merges ({len(merged)}):")
            for group, cid, others in merged:
                print(f"  - cluster {cid} mixes groups {sorted(others)}")

    return {
        "eps": eps, "clusters_found": len(clusters),
        "correctly_unified": correctly_unified, "fragmented": len(fragmented), "merged": len(merged),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=2000)
    parser.add_argument("--eps", type=float, default=0.3)
    parser.add_argument("--min-samples", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sweep", action="store_true")
    args = parser.parse_args()

    if args.sweep:
        print(f"{'eps':>5} | {'clusters':>8} | {'unified':>7} | {'fragmented':>10} | {'merged':>6}")
        print("-" * 50)
        for eps in [0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]:
            r = run_semantic_eval(args.count, eps, args.min_samples, seed=args.seed, verbose=False)
            print(f"{eps:>5} | {r['clusters_found']:>8} | {r['correctly_unified']:>7} | "
                  f"{r['fragmented']:>10} | {r['merged']:>6}")
    else:
        run_semantic_eval(args.count, args.eps, args.min_samples, seed=args.seed, verbose=True)


if __name__ == "__main__":
    main()