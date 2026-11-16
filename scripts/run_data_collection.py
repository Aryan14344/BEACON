"""
Pipeline collection runner for PhishBP.
Fetches active phishing feeds (OpenPhish) and benign lists (Tranco),
runs asynchronous DNS lookups, and builds the HeteroData graph object.

Usage:
    python scripts/run_data_collection.py --n_phish 500 --n_benign 500
"""

import os
import sys
import argparse
import asyncio
import logging
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from src.pipeline.feed_collector import collect_openphish, collect_tranco, prune_shared_hosting, load_public_suffix_list
from src.pipeline.dns_resolver import resolve_domains_bulk, ips_to_asns_bulk
from src.pipeline.graph_builder import GraphBuilder, make_temporal_splits
from src.features.lexical import extract_features_batch

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("DataPipeline")


async def main():
    parser = argparse.ArgumentParser(description="Run PhishBP Data Collection and Graph Construction")
    parser.add_argument("--n_phish", type=int, default=500, help="Max phishing samples")
    parser.add_argument("--n_benign", type=int, default=500, help="Max benign samples")
    parser.add_argument("--concurrency", type=int, default=30, help="Async DNS concurrency")
    parser.add_argument("--output_dir", type=str, default="data/processed", help="Directory to save graphs")
    args = parser.parse_args()

    import aiohttp
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Step 1: Collecting phishing and benign feeds...")
    async with aiohttp.ClientSession() as session:
        phish_entries = await collect_openphish(session)
        phish_entries = phish_entries[:args.n_phish]
        
        n_top = args.n_benign // 2
        n_mid = args.n_benign - n_top
        benign_entries = await collect_tranco(session, n_top=n_top, n_mid=n_mid)

    shared_suffixes = load_public_suffix_list()
    benign_entries = prune_shared_hosting(benign_entries, shared_suffixes)
    all_entries = phish_entries + benign_entries

    logger.info(f"Collected {len(phish_entries)} phishing and {len(benign_entries)} benign entries.")

    domains_to_resolve = list({e.domain for e in all_entries if e.domain})
    logger.info(f"Step 2: Resolving DNS for {len(domains_to_resolve)} unique domains (concurrency={args.concurrency})...")
    dns_records = await resolve_domains_bulk(domains_to_resolve, concurrency=args.concurrency, progress=True)

    # Collect unique IPs for ASN resolution
    unique_ips = set()
    for rec in dns_records.values():
        unique_ips.update(rec.a_records)
    logger.info(f"Resolving ASNs for {len(unique_ips)} unique IPs...")
    asn_records = await ips_to_asns_bulk(list(unique_ips), concurrency=20)

    logger.info("Step 3: Extracting feature vectors...")
    X_features = extract_features_batch(all_entries, dns_records=dns_records, verbose=True)

    logger.info("Step 4: Building PyG HeteroData graph...")
    builder = GraphBuilder()
    hetero_data = builder.build(all_entries, dns_records=dns_records, asn_records=asn_records)

    # Attach features to domain/url nodes
    # Map features to registry indices
    if hetero_data['url'].num_nodes > 0:
        url_features = torch.zeros((hetero_data['url'].num_nodes, X_features.shape[1]))
        for i, entry in enumerate(all_entries):
            idx = builder.registry.get("url", entry.url)
            if idx is not None and idx < hetero_data['url'].num_nodes:
                url_features[idx] = torch.from_numpy(X_features[i])
        hetero_data['url'].x = url_features

    save_path = out_dir / "hetero_graph.pt"
    torch.save(hetero_data, save_path)
    logger.info(f"Saved completed HeteroData graph to {save_path}!")

    # Generate temporal windows
    splits = make_temporal_splits(all_entries, n_windows=3)
    torch.save(splits, out_dir / "temporal_splits.pt")
    logger.info("Saved temporal splits!")


if __name__ == "__main__":
    asyncio.run(main())
