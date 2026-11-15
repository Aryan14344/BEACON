"""
Feed collectors for PhishBP data pipeline.

Sources:
  - PhishTank  : phishing URLs (requires free API key)
  - OpenPhish  : phishing URLs (free, no key)
  - Tranco     : top-1M benign domains (ranked list)

Each collector returns a list of (url_or_domain, label, timestamp) tuples.
label: 1 = phishing, 0 = benign

Usage
-----
>>> from src.pipeline.feed_collector import collect_all_feeds
>>> import asyncio
>>> data = asyncio.run(collect_all_feeds(save_dir="data/raw"))
"""

from __future__ import annotations

import asyncio
import csv
import io
import json
import logging
import os
import time
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

try:
    import aiohttp
    HAS_AIOHTTP = True
except ImportError:
    HAS_AIOHTTP = False
    logger.warning("aiohttp not installed. Feed collection unavailable.")

try:
    import tldextract
    HAS_TLDEXTRACT = True
except ImportError:
    HAS_TLDEXTRACT = False


# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class FeedEntry:
    url:       str
    domain:    str
    label:     int      # 0 = benign, 1 = phishing
    source:    str      # "phishtank" | "openphish" | "tranco"
    timestamp: datetime
    extra:     dict     = None  # source-specific metadata


def _extract_domain(url: str) -> str:
    """Extract registered domain from a URL using tldextract."""
    if HAS_TLDEXTRACT:
        ext = tldextract.extract(url)
        if ext.domain and ext.suffix:
            return f"{ext.domain}.{ext.suffix}"
    # Fallback: naive split
    url = url.replace("https://", "").replace("http://", "").split("/")[0]
    return url.split("@")[-1].split(":")[0]


# ─────────────────────────────────────────────────────────────────────────────
# OpenPhish (no API key, completely free)
# ─────────────────────────────────────────────────────────────────────────────

async def collect_openphish(session: "aiohttp.ClientSession") -> list[FeedEntry]:
    """
    Fetch the OpenPhish free feed.
    URL: https://openphish.com/feed.txt
    Returns one phishing URL per line.
    Terms: https://openphish.com/terms.html — free for research.
    """
    url = "https://openphish.com/feed.txt"
    entries = []
    try:
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=30)) as resp:
            if resp.status != 200:
                logger.warning(f"OpenPhish returned HTTP {resp.status}")
                return []
            text = await resp.text()
        now = datetime.now(timezone.utc)
        for line in text.strip().splitlines():
            line = line.strip()
            if not line or not line.startswith("http"):
                continue
            entries.append(FeedEntry(
                url=line,
                domain=_extract_domain(line),
                label=1,
                source="openphish",
                timestamp=now,
                extra={},
            ))
        logger.info(f"OpenPhish: collected {len(entries)} phishing URLs")
    except Exception as e:
        logger.error(f"OpenPhish collection failed: {e}")
    return entries


# ─────────────────────────────────────────────────────────────────────────────
# PhishTank (requires free API key)
# ─────────────────────────────────────────────────────────────────────────────

async def collect_phishtank(
    session: "aiohttp.ClientSession",
    api_key: Optional[str] = None,
    max_entries: int = 10000,
) -> list[FeedEntry]:
    """
    Fetch PhishTank verified phishing database (JSON format).
    API key: register free at https://www.phishtank.com/api_register.php
    Terms: https://www.phishtank.com/terms.php — free for non-commercial research.

    If api_key is None, uses the anonymous download (rate-limited).
    """
    if api_key:
        url = f"http://data.phishtank.com/data/{api_key}/online-valid.json.bz2"
    else:
        url = "http://data.phishtank.com/data/online-valid.json.bz2"

    entries = []
    try:
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=120)) as resp:
            if resp.status != 200:
                logger.warning(f"PhishTank returned HTTP {resp.status}")
                return []
            raw = await resp.read()

        import bz2
        data = json.loads(bz2.decompress(raw))

        for item in data[:max_entries]:
            if not item.get("verified") or item.get("valid") != "yes":
                continue
            phish_url = item.get("url", "")
            ts_str = item.get("submission_time", "")
            try:
                ts = datetime.fromisoformat(ts_str.replace("T", " ").replace("+00:00", ""))
                ts = ts.replace(tzinfo=timezone.utc)
            except Exception:
                ts = datetime.now(timezone.utc)

            entries.append(FeedEntry(
                url=phish_url,
                domain=_extract_domain(phish_url),
                label=1,
                source="phishtank",
                timestamp=ts,
                extra={"phish_id": item.get("phish_id"), "country": item.get("country")},
            ))

        logger.info(f"PhishTank: collected {len(entries)} verified phishing URLs")
    except Exception as e:
        logger.error(f"PhishTank collection failed: {e}")
    return entries


# ─────────────────────────────────────────────────────────────────────────────
# Tranco (benign domains)
# ─────────────────────────────────────────────────────────────────────────────

async def collect_tranco(
    session: "aiohttp.ClientSession",
    n_top: int = 5000,
    n_mid: int = 5000,
    mid_start: int = 10_000,
    mid_end: int = 500_000,
) -> list[FeedEntry]:
    """
    Download the Tranco top-1M list and sample both top (popular) and
    mid-range domains to avoid benign-label bias.

    n_top  : take this many from the very top (rank 1 to n_top)
    n_mid  : take this many from mid-range (rank mid_start to mid_end)

    Terms: https://tranco-list.eu — free for research.
    """
    # Latest Tranco list (always up to date)
    list_url = "https://tranco-list.eu/top-1m.csv.zip"
    entries = []
    try:
        async with session.get(list_url, timeout=aiohttp.ClientTimeout(total=120)) as resp:
            if resp.status != 200:
                logger.warning(f"Tranco returned HTTP {resp.status}")
                return []
            raw = await resp.read()

        with zipfile.ZipFile(io.BytesIO(raw)) as z:
            csv_name = [n for n in z.namelist() if n.endswith(".csv")][0]
            with z.open(csv_name) as f:
                reader = csv.reader(io.TextIOWrapper(f, encoding="utf-8"))
                rows = [(int(row[0]), row[1]) for row in reader if row]

        now = datetime.now(timezone.utc)
        import random
        rng = random.Random(42)

        # Top domains
        top_domains = [domain for _, domain in rows[:n_top]]
        # Mid-range domains (less popular — reduces bias)
        mid_rows = [domain for rank, domain in rows if mid_start <= rank <= mid_end]
        mid_domains = rng.sample(mid_rows, min(n_mid, len(mid_rows)))

        for domain in top_domains + mid_domains:
            entries.append(FeedEntry(
                url=f"http://{domain}",
                domain=domain,
                label=0,
                source="tranco",
                timestamp=now,
                extra={},
            ))

        logger.info(
            f"Tranco: collected {len(top_domains)} top + {len(mid_domains)} mid-range benign domains"
        )
    except Exception as e:
        logger.error(f"Tranco collection failed: {e}")
    return entries


# ─────────────────────────────────────────────────────────────────────────────
# Public Suffix List pruning
# ─────────────────────────────────────────────────────────────────────────────

def load_public_suffix_list(psl_path: Optional[str] = None) -> set[str]:
    """
    Load the Mozilla Public Suffix List to identify shared hosting domains
    (e.g. github.io, blogspot.com) that should be pruned.

    Returns a set of public suffixes to exclude from benign node creation.
    """
    # Common shared-hosting suffixes to always exclude
    SHARED_HOSTING = {
        "github.io", "gitlab.io", "netlify.app", "vercel.app",
        "blogspot.com", "wordpress.com", "weebly.com", "wix.com",
        "squarespace.com", "tumblr.com", "medium.com",
        "s3.amazonaws.com", "storage.googleapis.com",
        "azurewebsites.net", "cloudfront.net",
    }

    if psl_path and Path(psl_path).exists():
        try:
            with open(psl_path) as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("//"):
                        SHARED_HOSTING.add(line.lstrip("*."))
        except Exception as e:
            logger.warning(f"Could not load PSL from {psl_path}: {e}")

    return SHARED_HOSTING


def prune_shared_hosting(
    entries: list[FeedEntry],
    shared_suffixes: set[str],
) -> list[FeedEntry]:
    """Remove entries whose domain is or ends with a shared-hosting suffix."""
    pruned = []
    removed = 0
    for entry in entries:
        d = entry.domain
        is_shared = any(d == s or d.endswith("." + s) for s in shared_suffixes)
        if is_shared:
            removed += 1
        else:
            pruned.append(entry)
    logger.info(f"PSL pruning: removed {removed} shared-hosting entries, kept {len(pruned)}")
    return pruned


# ─────────────────────────────────────────────────────────────────────────────
# Orchestrator
# ─────────────────────────────────────────────────────────────────────────────

async def collect_all_feeds(
    save_dir: str = "data/raw",
    phishtank_api_key: Optional[str] = None,
    n_benign_top: int = 5000,
    n_benign_mid: int = 5000,
    max_phishtank: int = 10000,
    psl_path: Optional[str] = None,
) -> list[FeedEntry]:
    """
    Collect phishing + benign feeds, prune shared hosting, and save to disk.

    Returns combined list of FeedEntry objects.
    """
    if not HAS_AIOHTTP:
        raise ImportError("aiohttp required: pip install aiohttp")

    Path(save_dir).mkdir(parents=True, exist_ok=True)
    all_entries: list[FeedEntry] = []

    headers = {
        "User-Agent": "PhishBP-Research/0.1 (academic research, not for attack)"
    }

    async with aiohttp.ClientSession(headers=headers) as session:
        phish_entries, tranco_entries = await asyncio.gather(
            _collect_phish(session, phishtank_api_key, max_phishtank),
            collect_tranco(session, n_benign_top, n_benign_mid),
        )

    all_entries = phish_entries + tranco_entries

    # Prune shared hosting
    shared = load_public_suffix_list(psl_path)
    # Only prune benign entries (phishing via shared hosting is a valid signal)
    benign = [e for e in all_entries if e.label == 0]
    phish  = [e for e in all_entries if e.label == 1]
    benign = prune_shared_hosting(benign, shared)
    all_entries = phish + benign

    # Save to disk
    _save_entries(all_entries, save_dir)

    logger.info(
        f"Total: {sum(1 for e in all_entries if e.label==1)} phishing, "
        f"{sum(1 for e in all_entries if e.label==0)} benign"
    )
    return all_entries


async def _collect_phish(session, api_key, max_entries) -> list[FeedEntry]:
    openphish, phishtank = await asyncio.gather(
        collect_openphish(session),
        collect_phishtank(session, api_key, max_entries),
    )
    return openphish + phishtank


def _save_entries(entries: list[FeedEntry], save_dir: str) -> None:
    """Save entries to a timestamped JSONL file."""
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    path = Path(save_dir) / f"feeds_{ts}.jsonl"
    with open(path, "w") as f:
        for e in entries:
            record = {
                "url": e.url,
                "domain": e.domain,
                "label": e.label,
                "source": e.source,
                "timestamp": e.timestamp.isoformat(),
                "extra": e.extra or {},
            }
            f.write(json.dumps(record) + "\n")
    logger.info(f"Saved {len(entries)} entries to {path}")
