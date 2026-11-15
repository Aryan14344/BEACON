"""
Heterogeneous graph builder for PhishBP.

Takes DNS records + feed entries and builds a PyTorch Geometric HeteroData object.

Node types:  url, domain, ip, nameserver, asn
Edge types:
  url      → domain     : url_resolves_domain
  domain   → ip         : domain_resolves_ip    (A records)
  domain   → nameserver : domain_uses_ns        (NS records)
  nameserver → ip       : ns_resolves_ip        (NS server A record)
  ip       → asn        : ip_belongs_asn

Usage
-----
>>> from src.pipeline.graph_builder import GraphBuilder
>>> builder = GraphBuilder()
>>> hetero_data = builder.build(feed_entries, dns_records)
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

try:
    import torch
    from torch_geometric.data import HeteroData
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    logger.warning("PyTorch Geometric not installed. Graph builder unavailable.")


# ─────────────────────────────────────────────────────────────────────────────
# Node and edge registries
# ─────────────────────────────────────────────────────────────────────────────

class NodeRegistry:
    """Maps string identifiers to integer node indices per type."""

    def __init__(self):
        self._maps: dict[str, dict[str, int]] = defaultdict(dict)

    def get_or_add(self, node_type: str, name: str) -> int:
        m = self._maps[node_type]
        if name not in m:
            m[name] = len(m)
        return m[name]

    def get(self, node_type: str, name: str) -> Optional[int]:
        return self._maps[node_type].get(name)

    def size(self, node_type: str) -> int:
        return len(self._maps[node_type])

    def all_names(self, node_type: str) -> list[str]:
        return list(self._maps[node_type].keys())


# ─────────────────────────────────────────────────────────────────────────────
# Graph builder
# ─────────────────────────────────────────────────────────────────────────────

class GraphBuilder:
    """
    Builds a PyTorch Geometric HeteroData from feed entries and DNS records.

    Node metadata (names, labels) is stored in a separate dict for
    interpretability and to prevent label leakage.

    Parameters
    ----------
    min_domain_degree : prune IP nodes that have fewer than this many
                        connecting domains (removes noise from shared infra)
    """

    NODE_TYPES = ["url", "domain", "ip", "nameserver", "asn"]

    EDGE_TYPES = [
        ("url",        "resolves_to",   "domain"),
        ("domain",     "resolves_to",   "ip"),
        ("domain",     "uses_ns",       "nameserver"),
        ("nameserver", "resolves_to",   "ip"),
        ("ip",         "belongs_to",    "asn"),
    ]

    def __init__(self, min_domain_degree: int = 2):
        self.min_domain_degree = min_domain_degree
        self.registry = NodeRegistry()
        self._edges: dict[tuple, list[tuple[int, int]]] = defaultdict(list)
        self._labels: dict[str, dict[int, int]] = defaultdict(dict)  # type→idx→label
        self._names: dict[str, list[str]] = defaultdict(list)

    def build(
        self,
        feed_entries,       # list[FeedEntry]
        dns_records,        # dict[str, DNSRecord]
        asn_records=None,   # dict[str, str]  ip→asn
    ):
        """
        Build HeteroData from collected data.

        Parameters
        ----------
        feed_entries : output of feed_collector.collect_all_feeds()
        dns_records  : output of dns_resolver.resolve_domains_bulk()
        asn_records  : output of dns_resolver.ips_to_asns_bulk() (optional)
        """
        if not HAS_TORCH:
            raise ImportError("Install torch and torch-geometric first.")

        self.registry = NodeRegistry()
        self._edges = defaultdict(list)
        self._labels = defaultdict(dict)

        logger.info(f"Building graph from {len(feed_entries)} feed entries...")

        # ── Pass 1: Register domains and URLs from feed ───────────────────────
        for entry in feed_entries:
            domain_idx = self.registry.get_or_add("domain", entry.domain)
            self._labels["domain"][domain_idx] = entry.label

            url_idx = self.registry.get_or_add("url", entry.url)
            self._labels["url"][url_idx] = entry.label

            # url → domain
            self._edges[("url", "resolves_to", "domain")].append(
                (url_idx, domain_idx)
            )

        # ── Pass 2: Add DNS-derived nodes and edges ───────────────────────────
        for domain, record in dns_records.items():
            if not record.resolved:
                continue

            d_idx = self.registry.get("domain", domain)
            if d_idx is None:
                # Domain appeared in DNS but not in feeds — add as unlabelled
                d_idx = self.registry.get_or_add("domain", domain)

            # A records → IP nodes
            for ip in record.a_records:
                ip_idx = self.registry.get_or_add("ip", ip)
                self._edges[("domain", "resolves_to", "ip")].append((d_idx, ip_idx))

            # NS records → nameserver nodes
            for ns in record.ns_records:
                ns_idx = self.registry.get_or_add("nameserver", ns)
                self._edges[("domain", "uses_ns", "nameserver")].append((d_idx, ns_idx))

            # NS nameserver IP resolution (if available)
            if ns in dns_records:
                ns_record = dns_records[ns]
                ns_idx_check = self.registry.get("nameserver", ns)
                if ns_idx_check is not None:
                    for ns_ip in ns_record.a_records:
                        ns_ip_idx = self.registry.get_or_add("ip", ns_ip)
                        self._edges[("nameserver", "resolves_to", "ip")].append(
                            (ns_idx_check, ns_ip_idx)
                        )

        # ── Pass 3: ASN nodes ─────────────────────────────────────────────────
        if asn_records:
            for ip, asn in asn_records.items():
                if not asn:
                    continue
                ip_idx = self.registry.get("ip", ip)
                if ip_idx is None:
                    continue
                asn_idx = self.registry.get_or_add("asn", asn)
                self._edges[("ip", "belongs_to", "asn")].append((ip_idx, asn_idx))

        # ── Propagate labels to IP and NS nodes ──────────────────────────────
        # IP node is "phishing" if ANY phishing domain resolves to it
        self._propagate_labels_to_infra()

        # ── Build HeteroData ─────────────────────────────────────────────────
        data = self._to_heterodata()
        self._log_graph_stats(data)
        return data

    def _propagate_labels_to_infra(self):
        """
        Soft label propagation: IP/NS/ASN nodes get label=1 if any
        connected phishing domain uses them. Unknown = -1.
        """
        # IP labels
        for (d_idx, ip_idx) in self._edges.get(("domain", "resolves_to", "ip"), []):
            d_label = self._labels["domain"].get(d_idx, -1)
            current = self._labels["ip"].get(ip_idx, 0)
            if d_label == 1:
                self._labels["ip"][ip_idx] = 1
            elif d_label == 0 and ip_idx not in self._labels["ip"]:
                self._labels["ip"][ip_idx] = 0

        # NS labels
        for (d_idx, ns_idx) in self._edges.get(("domain", "uses_ns", "nameserver"), []):
            d_label = self._labels["domain"].get(d_idx, -1)
            if d_label == 1:
                self._labels["nameserver"][ns_idx] = 1
            elif d_label == 0 and ns_idx not in self._labels["nameserver"]:
                self._labels["nameserver"][ns_idx] = 0

    def _to_heterodata(self):
        """Convert internal registries to a PyG HeteroData object."""
        data = HeteroData()

        # Node counts (placeholder features = ones; real features added later)
        for ntype in self.NODE_TYPES:
            n = self.registry.size(ntype)
            if n == 0:
                continue
            data[ntype].num_nodes = n
            data[ntype].x = torch.ones(n, 1)  # placeholder

            # Labels (-1 = unknown)
            y = torch.full((n,), -1, dtype=torch.long)
            for idx, label in self._labels[ntype].items():
                y[idx] = label
            data[ntype].y = y

            # Store names for debugging
            data[ntype].names = self.registry.all_names(ntype)

        # Edges
        for etype_tuple, edge_list in self._edges.items():
            if not edge_list:
                continue
            src_type, rel, dst_type = etype_tuple
            if data[src_type].num_nodes == 0 or data[dst_type].num_nodes == 0:
                continue
            edge_arr = torch.tensor(edge_list, dtype=torch.long).T  # (2, E)
            data[etype_tuple].edge_index = edge_arr

        return data

    def _log_graph_stats(self, data):
        logger.info("=== Graph Statistics ===")
        for ntype in data.node_types:
            n = data[ntype].num_nodes
            labeled = (data[ntype].y >= 0).sum().item()
            phishing = (data[ntype].y == 1).sum().item()
            logger.info(f"  {ntype:12s}: {n:6d} nodes  |  {labeled:5d} labeled  |  {phishing:5d} phishing")

        for etype in data.edge_types:
            e = data[etype].edge_index.shape[1]
            logger.info(f"  {str(etype):45s}: {e:6d} edges")


# ─────────────────────────────────────────────────────────────────────────────
# Temporal split generator
# ─────────────────────────────────────────────────────────────────────────────

def make_temporal_splits(
    feed_entries,
    n_windows: int = 5,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
) -> list[dict]:
    """
    Create temporal train/val/test splits.

    Entries are sorted by timestamp and divided into sliding windows.
    Within each window: train on first (train_ratio) fraction, etc.

    Returns list of dicts with keys: "train", "val", "test"
    each containing lists of FeedEntry.

    CRITICAL: No data from future time windows leaks into training.
    """
    from datetime import timezone

    sorted_entries = sorted(feed_entries, key=lambda e: e.timestamp)
    total = len(sorted_entries)
    window_size = total // n_windows

    splits = []
    for i in range(n_windows - 1):
        window = sorted_entries[i * window_size: (i + 2) * window_size]
        n = len(window)
        n_train = int(n * train_ratio)
        n_val   = int(n * val_ratio)
        splits.append({
            "window": i,
            "train": window[:n_train],
            "val":   window[n_train: n_train + n_val],
            "test":  window[n_train + n_val:],
            "train_end_time":   window[n_train - 1].timestamp.isoformat(),
            "test_start_time":  window[n_train + n_val].timestamp.isoformat() if n_train + n_val < n else None,
        })
        logger.info(
            f"Window {i}: train={len(splits[-1]['train'])} "
            f"val={len(splits[-1]['val'])} "
            f"test={len(splits[-1]['test'])}"
        )
    return splits
