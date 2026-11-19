"""
PhishBP Multi-Source Data Collection Pipeline
==============================================
Collects phishing + benign URLs from 5 sources, resolves DNS,
builds a HeteroData graph, and saves temporal splits.

Sources:
  PHISHING:  OpenPhish | PhishTank | PhishStats
  BENIGN:    Tranco Top-1M | Cisco Umbrella Top-1M

Usage (from repo root, with venv active):
    python scripts/run_data_collection.py
    python scripts/run_data_collection.py --n_tranco_top 5000 --n_umbrella 5000
    python scripts/run_data_collection.py --phishtank_api_key <KEY>
"""

import os
import sys
import argparse
import asyncio
import logging
from pathlib import Path

# ── Path setup (repo root) ────────────────────────────────────────────────────
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from src.pipeline.feed_collector import collect_all
from src.pipeline.dns_resolver import resolve_domains_bulk, ips_to_asns_bulk
from src.pipeline.graph_builder import GraphBuilder, make_temporal_splits
from src.features.lexical import extract_features_batch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("DataPipeline")


async def main(args: argparse.Namespace) -> None:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Collect all feeds ──────────────────────────────────────────────────
    logger.info("Step 1: Collecting from all 5 feeds (OpenPhish, PhishTank, PhishStats, Tranco, Umbrella)…")
    all_entries = await collect_all(
        save_dir=str(out_dir.parent / "raw"),
        phishtank_api_key=args.phishtank_api_key or None,
        n_tranco_top=args.n_tranco_top,
        n_tranco_mid=args.n_tranco_mid,
        n_umbrella=args.n_umbrella,
        max_phishtank=args.max_phishtank,
        max_phishstats=args.max_phishstats,
    )

    phish_count  = sum(1 for e in all_entries if e.label == 1)
    benign_count = sum(1 for e in all_entries if e.label == 0)
    logger.info(f"After preprocessing: {phish_count} phishing | {benign_count} benign | "
                f"{len(all_entries)} total")

    if len(all_entries) < 100:
        logger.error("Too few samples — check network access and feed URLs.")
        return

    # ── 2. DNS resolution ─────────────────────────────────────────────────────
    domains_to_resolve = list({e.domain for e in all_entries if e.domain})
    logger.info(f"Step 2: Resolving DNS for {len(domains_to_resolve)} unique domains "
                f"(concurrency={args.concurrency})…")
    dns_records = await resolve_domains_bulk(
        domains_to_resolve, concurrency=args.concurrency, progress=True
    )

    unique_ips: set[str] = set()
    for rec in dns_records.values():
        unique_ips.update(rec.a_records)
    logger.info(f"Step 2b: Resolving ASNs for {len(unique_ips)} unique IPs…")
    asn_records = await ips_to_asns_bulk(list(unique_ips), concurrency=20)

    # ── 3. Feature extraction ─────────────────────────────────────────────────
    logger.info("Step 3: Extracting 21-dim lexical + DNS features…")
    X_features = extract_features_batch(all_entries, dns_records=dns_records, verbose=True)

    # ── 4. Graph construction ─────────────────────────────────────────────────
    logger.info("Step 4: Building PyG HeteroData graph…")
    builder     = GraphBuilder()
    hetero_data = builder.build(all_entries, dns_records=dns_records, asn_records=asn_records)

    # Attach URL-node features
    if hetero_data["url"].num_nodes > 0:
        url_features = torch.zeros((hetero_data["url"].num_nodes, X_features.shape[1]))
        for i, entry in enumerate(all_entries):
            idx = builder.registry.get("url", entry.url)
            if idx is not None and idx < hetero_data["url"].num_nodes:
                url_features[idx] = torch.from_numpy(X_features[i])
        hetero_data["url"].x = url_features

    # ── 5. Save ───────────────────────────────────────────────────────────────
    graph_path = out_dir / "hetero_graph.pt"
    torch.save(hetero_data, graph_path)
    logger.info(f"Saved HeteroData → {graph_path}")

    for nt in ["url", "domain", "ip", "nameserver", "asn"]:
        if nt in hetero_data.node_types:
            logger.info(f"  {nt}: {hetero_data[nt].num_nodes} nodes")

    splits = make_temporal_splits(all_entries, n_windows=3)
    splits_path = out_dir / "temporal_splits.pt"
    torch.save(splits, splits_path)
    logger.info(f"Saved temporal splits → {splits_path}")

    # ── 6. Homophily report ───────────────────────────────────────────────────
    logger.info("Step 6: Edge homophily report:")
    rel_map = {
        ("url", "resolves_to", "domain"): "url→domain",
        ("domain", "resolves_to", "ip"):   "domain→ip",
        ("domain", "uses_ns", "nameserver"): "domain→ns",
    }
    for etype, name in rel_map.items():
        if etype in hetero_data.edge_types:
            ei   = hetero_data[etype].edge_index
            src_t, _, dst_t = etype
            y_src = hetero_data[src_t].y[ei[0]]
            y_dst = hetero_data[dst_t].y[ei[1]]
            # Only count edges where both endpoints are labelled (≥0)
            mask  = (y_src >= 0) & (y_dst >= 0)
            if mask.sum() > 0:
                h = (y_src[mask] == y_dst[mask]).float().mean().item()
                logger.info(f"  {name:20s} h = {h:.4f}  ({mask.sum().item()} labelled edges)")

    logger.info("✓ Data collection and graph construction complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PhishBP multi-source data collection")
    parser.add_argument("--phishtank_api_key", type=str, default=None,
                        help="Optional PhishTank API key (unauthenticated still works, slower)")
    parser.add_argument("--n_tranco_top",   type=int, default=3_000)
    parser.add_argument("--n_tranco_mid",   type=int, default=4_000)
    parser.add_argument("--n_umbrella",     type=int, default=3_000)
    parser.add_argument("--max_phishtank",  type=int, default=10_000)
    parser.add_argument("--max_phishstats", type=int, default=15_000)
    parser.add_argument("--concurrency",    type=int, default=30)
    parser.add_argument("--output_dir",     type=str, default="data/processed")
    args = parser.parse_args()
    asyncio.run(main(args))
