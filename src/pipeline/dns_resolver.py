"""
Async DNS resolver for PhishBP pipeline.

Resolves A, MX, NS, CNAME records for a list of domains.
Uses dnspython with asyncio for high throughput.

Usage
-----
>>> import asyncio
>>> from src.pipeline.dns_resolver import resolve_domains_bulk
>>> records = asyncio.run(resolve_domains_bulk(["google.com", "evil-phish.xyz"]))
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

try:
    import dns.asyncresolver
    import dns.exception
    import dns.rdatatype
    HAS_DNSPYTHON = True
except ImportError:
    HAS_DNSPYTHON = False
    logger.warning("dnspython not installed. DNS resolution unavailable.")


# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class DNSRecord:
    """All resolved DNS records for a single domain."""
    domain: str
    a_records:     list[str] = field(default_factory=list)   # IPv4 addresses
    aaaa_records:  list[str] = field(default_factory=list)   # IPv6 addresses
    mx_records:    list[str] = field(default_factory=list)   # Mail servers
    ns_records:    list[str] = field(default_factory=list)   # Nameservers
    cname_records: list[str] = field(default_factory=list)   # CNAME chain
    txt_records:   list[str] = field(default_factory=list)   # TXT (SPF, DKIM)
    resolve_time:  float     = 0.0                           # seconds
    error:         Optional[str] = None

    @property
    def resolved(self) -> bool:
        return self.error is None and bool(self.a_records or self.cname_records)


# ─────────────────────────────────────────────────────────────────────────────
# Single-domain resolver
# ─────────────────────────────────────────────────────────────────────────────

async def resolve_domain(
    domain: str,
    timeout: float = 5.0,
    nameservers: Optional[list[str]] = None,
) -> DNSRecord:
    """
    Resolve all record types for a single domain asynchronously.

    Parameters
    ----------
    domain      : fully-qualified domain name (e.g. "example.com")
    timeout     : per-query timeout in seconds
    nameservers : custom DNS servers (default: system resolver)
    """
    if not HAS_DNSPYTHON:
        return DNSRecord(domain=domain, error="dnspython not installed")

    record = DNSRecord(domain=domain)
    t0 = time.perf_counter()

    resolver = dns.asyncresolver.Resolver()
    resolver.timeout = timeout
    resolver.lifetime = timeout * 2
    if nameservers:
        resolver.nameservers = nameservers

    async def query(rdtype: str) -> list[str]:
        try:
            answer = await resolver.resolve(domain, rdtype)
            return [str(r) for r in answer]
        except (dns.exception.DNSException, Exception):
            return []

    # Run all record types concurrently
    results = await asyncio.gather(
        query("A"),
        query("AAAA"),
        query("MX"),
        query("NS"),
        query("CNAME"),
        query("TXT"),
        return_exceptions=False,
    )

    record.a_records     = results[0]
    record.aaaa_records  = results[1]
    # MX comes as "priority hostname." — extract just hostname
    record.mx_records    = [r.split(" ", 1)[-1].rstrip(".") for r in results[2]]
    record.ns_records    = [r.rstrip(".") for r in results[3]]
    record.cname_records = [r.rstrip(".") for r in results[4]]
    record.txt_records   = results[5]
    record.resolve_time  = time.perf_counter() - t0

    if not record.resolved:
        record.error = "NXDOMAIN or no A/CNAME records"

    return record


# ─────────────────────────────────────────────────────────────────────────────
# Bulk resolver with rate limiting and retry
# ─────────────────────────────────────────────────────────────────────────────

async def resolve_domains_bulk(
    domains: list[str],
    concurrency: int = 50,
    timeout: float = 5.0,
    max_retries: int = 2,
    nameservers: Optional[list[str]] = None,
    progress: bool = True,
) -> dict[str, DNSRecord]:
    """
    Resolve many domains concurrently with a semaphore rate-limiter.

    Parameters
    ----------
    concurrency : max simultaneous DNS queries
    max_retries : retry failed queries this many times (exponential backoff)

    Returns
    -------
    dict mapping domain → DNSRecord
    """
    semaphore = asyncio.Semaphore(concurrency)
    results: dict[str, DNSRecord] = {}
    total = len(domains)
    done = 0

    async def resolve_with_retry(domain: str) -> DNSRecord:
        nonlocal done
        async with semaphore:
            for attempt in range(max_retries + 1):
                rec = await resolve_domain(domain, timeout=timeout, nameservers=nameservers)
                if rec.resolved or rec.error == "dnspython not installed":
                    break
                if attempt < max_retries:
                    await asyncio.sleep(0.5 * (2 ** attempt))  # exponential backoff
            done += 1
            if progress and done % 100 == 0:
                logger.info(f"  DNS resolved {done}/{total} domains")
            return rec

    tasks = [resolve_with_retry(d) for d in domains]
    resolved = await asyncio.gather(*tasks, return_exceptions=False)
    for d, r in zip(domains, resolved):
        results[d] = r

    return results


# ─────────────────────────────────────────────────────────────────────────────
# IP → ASN lookup (using Team Cymru DNS)
# ─────────────────────────────────────────────────────────────────────────────

async def ip_to_asn(ip: str) -> Optional[str]:
    """
    Look up the ASN for an IP via Team Cymru's DNS service.
    Returns ASN string like "AS15169" or None on failure.

    Free, no API key required.
    Example: 8.8.8.8 → query 8.8.8.8.origin.asn.cymru.com
    """
    if not HAS_DNSPYTHON:
        return None
    try:
        parts = ip.split(".")
        if len(parts) != 4:
            return None
        reversed_ip = ".".join(reversed(parts))
        query_name = f"{reversed_ip}.origin.asn.cymru.com"
        resolver = dns.asyncresolver.Resolver()
        resolver.timeout = 5.0
        answer = await resolver.resolve(query_name, "TXT")
        # TXT record format: "asn | prefix | country | registry | allocated"
        txt = str(answer[0]).strip('"').split("|")
        return f"AS{txt[0].strip()}"
    except Exception:
        return None


async def ips_to_asns_bulk(
    ips: list[str],
    concurrency: int = 30,
) -> dict[str, Optional[str]]:
    """Bulk IP→ASN resolution."""
    semaphore = asyncio.Semaphore(concurrency)
    results = {}

    async def lookup(ip):
        async with semaphore:
            return ip, await ip_to_asn(ip)

    tasks = [lookup(ip) for ip in ips]
    resolved = await asyncio.gather(*tasks)
    for ip, asn in resolved:
        results[ip] = asn
    return results
