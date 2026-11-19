"""
PhishBP Multi-Dataset Collector and Preprocessor
=================================================
Integrates THREE phishing + TWO benign data sources with full preprocessing:

PHISHING SOURCES:
  1. OpenPhish   – live feed, no key needed         (~300–500 URLs / day)
  2. PhishTank   – verified, free API key            (~10k URLs)
  3. PhishStats  – CSV export, historical archive    (~50k+ URLs)

BENIGN SOURCES:
  1. Tranco Top-1M – ranked list, top + mid-range sampling
  2. Cisco Umbrella Top-1M – alternative ranking for diversity

PREPROCESSING PIPELINE:
  1. URL normalisation (lowercase scheme+host, strip utm params, strip auth)
  2. Deduplication by (registered_domain, label) pair
  3. Public Suffix List (PSL) pruning of shared hosting domains
  4. Temporal tagging for train/val/test temporal splits
  5. Class balance audit and warning
  6. Graph node feature population (lexical + DNS)
  7. Edge homophily computation and report
"""

from __future__ import annotations

import asyncio
import csv
import hashlib
import io
import json
import logging
import re
import zipfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse, urlencode, parse_qs, urlunparse

logger = logging.getLogger(__name__)

try:
    import aiohttp
    HAS_AIOHTTP = True
except ImportError:
    HAS_AIOHTTP = False

try:
    import tldextract
    HAS_TLD = True
except ImportError:
    HAS_TLD = False

# ─────────────────────────────────────────────────────────────────────────────
# Data model
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Entry:
    url:       str
    domain:    str           # registered domain (eTLD+1)
    label:     int           # 0=benign, 1=phishing
    source:    str
    timestamp: datetime
    extra:     dict = field(default_factory=dict)

    @property
    def dedup_key(self) -> str:
        return hashlib.md5(f"{self.domain}:{self.label}".encode()).hexdigest()


# ─────────────────────────────────────────────────────────────────────────────
# URL normalisation
# ─────────────────────────────────────────────────────────────────────────────

# UTM / tracking params to strip
_TRACKING = re.compile(
    r"^(utm_source|utm_medium|utm_campaign|utm_term|utm_content|fbclid|gclid|ref|referrer)$",
    re.IGNORECASE,
)

def normalise_url(url: str) -> str:
    """
    Normalise a URL for deduplication and consistent feature extraction.
    Steps:
      1. Force lowercase scheme and host.
      2. Remove authentication (user:pass@).
      3. Strip tracking query parameters (utm_*, fbclid, etc.).
      4. Collapse repeated slashes in path.
    """
    try:
        p = urlparse(url.strip())
        scheme   = (p.scheme or "http").lower()
        netloc   = p.hostname or ""   # drops user:pass
        if p.port and p.port not in (80, 443):
            netloc += f":{p.port}"
        path     = re.sub(r"/{2,}", "/", p.path)
        # Strip tracking params
        qs       = {k: v for k, v in parse_qs(p.query).items() if not _TRACKING.match(k)}
        query    = urlencode({k: v[0] for k, v in qs.items()})
        return urlunparse((scheme, netloc, path, "", query, ""))
    except Exception:
        return url


def extract_domain(url: str) -> str:
    """Extract registered domain (eTLD+1) from a URL."""
    if HAS_TLD:
        ext = tldextract.extract(url)
        if ext.domain and ext.suffix:
            return f"{ext.domain}.{ext.suffix}".lower()
    # Fallback
    h = url.replace("https://", "").replace("http://", "").split("/")[0].split(":")[0]
    return h.lower()


# ─────────────────────────────────────────────────────────────────────────────
# Source 1: OpenPhish (live, free)
# ─────────────────────────────────────────────────────────────────────────────

async def fetch_openphish(session: "aiohttp.ClientSession") -> list[Entry]:
    """
    OpenPhish free feed: one phishing URL per line.
    URL: https://openphish.com/feed.txt
    Terms: free for research use.
    """
    entries = []
    try:
        async with session.get("https://openphish.com/feed.txt",
                               timeout=aiohttp.ClientTimeout(total=30)) as r:
            if r.status != 200:
                logger.warning(f"OpenPhish HTTP {r.status}")
                return []
            text = await r.text()
        now = datetime.now(timezone.utc)
        for line in text.strip().splitlines():
            u = normalise_url(line.strip())
            if u.startswith("http"):
                entries.append(Entry(
                    url=u, domain=extract_domain(u), label=1,
                    source="openphish", timestamp=now,
                ))
        logger.info(f"OpenPhish: {len(entries)} entries")
    except Exception as e:
        logger.error(f"OpenPhish failed: {e}")
    return entries


# ─────────────────────────────────────────────────────────────────────────────
# Source 2: PhishTank (verified, free API key)
# ─────────────────────────────────────────────────────────────────────────────

async def fetch_phishtank(
    session: "aiohttp.ClientSession",
    api_key: Optional[str] = None,
    max_entries: int = 10_000,
) -> list[Entry]:
    """
    PhishTank verified phishing database (bz2-compressed JSON).
    Register free at https://www.phishtank.com/api_register.php
    Terms: https://www.phishtank.com/terms.php — free for non-commercial research.
    """
    import bz2
    url = (
        f"http://data.phishtank.com/data/{api_key}/online-valid.json.bz2"
        if api_key else
        "http://data.phishtank.com/data/online-valid.json.bz2"
    )
    entries = []
    try:
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=120)) as r:
            if r.status != 200:
                logger.warning(f"PhishTank HTTP {r.status}")
                return []
            raw = await r.read()
        data = json.loads(bz2.decompress(raw))
        for item in data[:max_entries]:
            if not item.get("verified") or item.get("valid") != "yes":
                continue
            phish_url = normalise_url(item.get("url", ""))
            try:
                ts = datetime.fromisoformat(
                    item.get("submission_time", "").replace("T", " ").rstrip("+00:00")
                ).replace(tzinfo=timezone.utc)
            except Exception:
                ts = datetime.now(timezone.utc)
            entries.append(Entry(
                url=phish_url, domain=extract_domain(phish_url), label=1,
                source="phishtank", timestamp=ts,
                extra={"phish_id": item.get("phish_id")},
            ))
        logger.info(f"PhishTank: {len(entries)} verified entries")
    except Exception as e:
        logger.error(f"PhishTank failed: {e}")
    return entries


# ─────────────────────────────────────────────────────────────────────────────
# Source 3: PhishStats (CSV, historical)
# ─────────────────────────────────────────────────────────────────────────────

async def fetch_phishstats(
    session: "aiohttp.ClientSession",
    min_score: float = 5.0,
    max_entries: int = 15_000,
) -> list[Entry]:
    """
    PhishStats CSV feed — historical phishing data with confidence scores.
    URL: https://phishstats.info/phish_score.csv
    Terms: https://phishstats.info — free for research.
    Format: #id,date,url,ip,score
    """
    entries = []
    try:
        async with session.get(
            "https://phishstats.info/phish_score.csv",
            timeout=aiohttp.ClientTimeout(total=60),
        ) as r:
            if r.status != 200:
                logger.warning(f"PhishStats HTTP {r.status}")
                return []
            text = await r.text(errors="replace")
        reader = csv.reader(io.StringIO(text))
        count = 0
        for row in reader:
            if count >= max_entries:
                break
            if len(row) < 5 or row[0].startswith("#"):
                continue
            try:
                score = float(row[4].strip())
                if score < min_score:
                    continue
                u = normalise_url(row[2].strip())
                ts_str = row[1].strip()
                try:
                    ts = datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
                except Exception:
                    ts = datetime.now(timezone.utc)
                entries.append(Entry(
                    url=u, domain=extract_domain(u), label=1,
                    source="phishstats", timestamp=ts,
                    extra={"score": score, "ip": row[3].strip()},
                ))
                count += 1
            except Exception:
                continue
        logger.info(f"PhishStats: {len(entries)} entries (score≥{min_score})")
    except Exception as e:
        logger.error(f"PhishStats failed: {e}")
    return entries


# ─────────────────────────────────────────────────────────────────────────────
# Source 4: Tranco (benign, balanced sampling)
# ─────────────────────────────────────────────────────────────────────────────

async def fetch_tranco(
    session: "aiohttp.ClientSession",
    n_top: int = 3_000,
    n_mid: int = 4_000,
    mid_lo: int = 10_000,
    mid_hi: int = 500_000,
) -> list[Entry]:
    """
    Tranco Top-1M balanced sampling.
    n_top: from rank 1–n_top (high popularity)
    n_mid: sampled from rank mid_lo–mid_hi (less popular, reduces popularity bias)
    Terms: https://tranco-list.eu — free for research.
    """
    import random
    entries = []
    try:
        async with session.get(
            "https://tranco-list.eu/top-1m.csv.zip",
            timeout=aiohttp.ClientTimeout(total=120),
        ) as r:
            if r.status != 200:
                logger.warning(f"Tranco HTTP {r.status}")
                return []
            raw = await r.read()
        with zipfile.ZipFile(io.BytesIO(raw)) as z:
            csv_name = [n for n in z.namelist() if n.endswith(".csv")][0]
            with z.open(csv_name) as f:
                rows = [(int(row[0]), row[1]) for row in csv.reader(io.TextIOWrapper(f))]

        now = datetime.now(timezone.utc)
        rng = random.Random(42)

        top_domains  = [d for _, d in rows[:n_top]]
        mid_pool     = [d for rank, d in rows if mid_lo <= rank <= mid_hi]
        mid_domains  = rng.sample(mid_pool, min(n_mid, len(mid_pool)))

        for domain in top_domains + mid_domains:
            entries.append(Entry(
                url=f"https://{domain}/", domain=domain, label=0,
                source="tranco", timestamp=now,
            ))
        logger.info(f"Tranco: {len(entries)} benign ({n_top} top + {len(mid_domains)} mid)")
    except Exception as e:
        logger.error(f"Tranco failed: {e}")
    return entries


# ─────────────────────────────────────────────────────────────────────────────
# Source 5: Cisco Umbrella Top-1M (second benign source)
# ─────────────────────────────────────────────────────────────────────────────

async def fetch_umbrella(
    session: "aiohttp.ClientSession",
    n_sample: int = 3_000,
    start_rank: int = 5_000,
    end_rank: int = 300_000,
) -> list[Entry]:
    """
    Cisco Umbrella Top-1M — alternative popularity ranking.
    Provides diversity vs. Tranco (different aggregation methodology).
    URL: https://umbrella-static.s3-us-west-1.amazonaws.com/top-1m.csv.zip
    Terms: publicly available, intended for DNS research.
    """
    import random
    entries = []
    try:
        async with session.get(
            "https://umbrella-static.s3-us-west-1.amazonaws.com/top-1m.csv.zip",
            timeout=aiohttp.ClientTimeout(total=90),
        ) as r:
            if r.status != 200:
                logger.warning(f"Umbrella HTTP {r.status}")
                return []
            raw = await r.read()
        with zipfile.ZipFile(io.BytesIO(raw)) as z:
            csv_name = z.namelist()[0]
            with z.open(csv_name) as f:
                rows = [(int(row[0]), row[1]) for row in csv.reader(io.TextIOWrapper(f))]

        pool = [d for rank, d in rows if start_rank <= rank <= end_rank]
        rng  = random.Random(99)
        sampled = rng.sample(pool, min(n_sample, len(pool)))
        now = datetime.now(timezone.utc)
        for domain in sampled:
            entries.append(Entry(
                url=f"https://{domain}/", domain=domain, label=0,
                source="umbrella", timestamp=now,
            ))
        logger.info(f"Umbrella: {len(entries)} benign samples")
    except Exception as e:
        logger.error(f"Umbrella failed: {e}")
    return entries


# ─────────────────────────────────────────────────────────────────────────────
# Preprocessing: deduplication, PSL pruning, balance audit
# ─────────────────────────────────────────────────────────────────────────────

SHARED_HOSTING_PATTERNS = {
    "github.io", "gitlab.io", "netlify.app", "vercel.app",
    "blogspot.com", "wordpress.com", "weebly.com", "wix.com",
    "squarespace.com", "tumblr.com", "medium.com", "substack.com",
    "s3.amazonaws.com", "storage.googleapis.com", "blob.core.windows.net",
    "azurewebsites.net", "cloudfront.net", "pages.dev",
}


def preprocess(entries: list[Entry]) -> list[Entry]:
    """
    Full preprocessing pipeline:
      1. URL normalisation (already done at fetch time)
      2. Deduplication by (domain, label) hash
      3. PSL pruning of shared hosting for BENIGN entries
      4. Remove entries with empty domains
      5. Report class balance
    """
    # Step 1–2: Dedup
    seen: set[str] = set()
    deduped: list[Entry] = []
    for e in entries:
        k = e.dedup_key
        if k not in seen:
            seen.add(k)
            deduped.append(e)
    logger.info(f"After dedup: {len(deduped)} (removed {len(entries)-len(deduped)})")

    # Step 3: PSL pruning (benign only — phishing via shared hosting IS a signal)
    pruned = []
    removed_psl = 0
    for e in deduped:
        if not e.domain:
            continue
        is_shared = any(
            e.domain == s or e.domain.endswith("." + s)
            for s in SHARED_HOSTING_PATTERNS
        )
        if e.label == 0 and is_shared:
            removed_psl += 1
        else:
            pruned.append(e)
    logger.info(f"After PSL pruning: {len(pruned)} (removed {removed_psl} shared-hosting benign)")

    # Step 5: Balance audit
    n_phish  = sum(1 for e in pruned if e.label == 1)
    n_benign = sum(1 for e in pruned if e.label == 0)
    ratio    = n_phish / max(n_benign, 1)
    logger.info(f"Class balance: {n_phish} phishing | {n_benign} benign | ratio {ratio:.2f}")
    if ratio > 2.0:
        logger.warning("Dataset is >2× more phishing than benign — consider undersampling phish.")
    if ratio < 0.3:
        logger.warning("Dataset is <0.3 phishing ratio — consider oversampling or focal loss.")

    # Source breakdown
    from collections import Counter
    src_counts = Counter(e.source for e in pruned)
    logger.info("Source breakdown: " + ", ".join(f"{k}={v}" for k, v in src_counts.items()))

    return pruned


# ─────────────────────────────────────────────────────────────────────────────
# Master collection entrypoint
# ─────────────────────────────────────────────────────────────────────────────

async def collect_all(
    save_dir: str = "data/raw",
    phishtank_api_key: Optional[str] = None,
    n_tranco_top: int = 3_000,
    n_tranco_mid: int = 4_000,
    n_umbrella:   int = 3_000,
    max_phishtank: int = 10_000,
    max_phishstats: int = 15_000,
) -> list[Entry]:
    """
    Collect from all 5 sources, preprocess, save, and return.
    """
    if not HAS_AIOHTTP:
        raise ImportError("pip install aiohttp")

    Path(save_dir).mkdir(parents=True, exist_ok=True)
    headers = {"User-Agent": "PhishBP-Research/0.1 (academic, non-commercial)"}

    async with aiohttp.ClientSession(headers=headers) as session:
        phish1, phish2, phish3, benign1, benign2 = await asyncio.gather(
            fetch_openphish(session),
            fetch_phishtank(session, phishtank_api_key, max_phishtank),
            fetch_phishstats(session, max_entries=max_phishstats),
            fetch_tranco(session, n_tranco_top, n_tranco_mid),
            fetch_umbrella(session, n_umbrella),
        )

    all_entries = phish1 + phish2 + phish3 + benign1 + benign2
    logger.info(f"Raw collected: {len(all_entries)} total entries")
    all_entries = preprocess(all_entries)

    # Save to JSONL
    ts  = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out = Path(save_dir) / f"all_feeds_{ts}.jsonl"
    with open(out, "w") as f:
        for e in all_entries:
            f.write(json.dumps({
                "url": e.url, "domain": e.domain, "label": e.label,
                "source": e.source, "timestamp": e.timestamp.isoformat(),
                "extra": e.extra,
            }) + "\n")
    logger.info(f"Saved {len(all_entries)} preprocessed entries → {out}")
    return all_entries
