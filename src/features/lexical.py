"""
Lexical and DNS-based feature extractor for PhishBP.

Features computed per domain/URL (no labels used):
  Lexical (URL-level):
    - url_length          : total character count
    - domain_length       : length of registered domain
    - n_subdomains        : count of subdomains
    - has_ip_in_url       : URL uses raw IP instead of domain name
    - n_digits            : digit count in domain
    - digit_ratio         : n_digits / domain_length
    - n_hyphens           : hyphen count in domain
    - n_dots              : dot count in full URL
    - char_entropy        : Shannon entropy of domain characters
    - has_at_symbol       : @ in URL (before hostname)
    - has_double_slash    : // appears after scheme
    - is_https            : URL scheme is https
    - url_path_length     : length of path component
    - n_query_params      : number of query parameters
    - brand_in_subdomain  : known brand name in subdomain (typosquat indicator)
    - longest_word_length : longest token in domain split by - and .

  DNS-derived (domain-level, requires DNS records):
    - n_a_records         : number of A records
    - n_ns_records        : number of NS records
    - n_mx_records        : number of MX records
    - has_txt_spf         : TXT record contains SPF
    - has_cname           : domain has CNAME chain
    - ttl_a               : placeholder (requires raw TTL — set to -1 if unavailable)

All features are float32, NaN-safe (missing = 0.0 or -1.0 as noted).
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Optional
import numpy as np

try:
    import tldextract
    HAS_TLDEXTRACT = True
except ImportError:
    HAS_TLDEXTRACT = False

# Common brand names that appear in typosquatting attacks
KNOWN_BRANDS = {
    "paypal", "google", "microsoft", "apple", "amazon", "facebook",
    "netflix", "instagram", "linkedin", "twitter", "chase", "wellsfargo",
    "bankofamerica", "citibank", "ebay", "dropbox", "icloud", "outlook",
    "office365", "signin", "account", "secure", "login", "verify",
    "update", "confirm", "support", "helpdesk",
}

FEATURE_NAMES = [
    "url_length", "domain_length", "n_subdomains", "has_ip_in_url",
    "n_digits", "digit_ratio", "n_hyphens", "n_dots", "char_entropy",
    "has_at_symbol", "has_double_slash", "is_https", "url_path_length",
    "n_query_params", "brand_in_subdomain", "longest_word_length",
    # DNS features (populated when DNS records available)
    "n_a_records", "n_ns_records", "n_mx_records",
    "has_txt_spf", "has_cname",
]
N_FEATURES = len(FEATURE_NAMES)
FEATURE_INDEX = {name: i for i, name in enumerate(FEATURE_NAMES)}


# ─────────────────────────────────────────────────────────────────────────────
# Shannon entropy
# ─────────────────────────────────────────────────────────────────────────────

def _entropy(s: str) -> float:
    if not s:
        return 0.0
    counts = {}
    for c in s:
        counts[c] = counts.get(c, 0) + 1
    n = len(s)
    return -sum((v / n) * math.log2(v / n) for v in counts.values())


# ─────────────────────────────────────────────────────────────────────────────
# IP-in-URL detector
# ─────────────────────────────────────────────────────────────────────────────

_IP_PATTERN = re.compile(r"^\d{1,3}(\.\d{1,3}){3}$")

def _has_ip_in_url(url: str) -> bool:
    # Strip scheme
    url_no_scheme = re.sub(r"^https?://", "", url).split("/")[0].split(":")[0]
    return bool(_IP_PATTERN.match(url_no_scheme))


# ─────────────────────────────────────────────────────────────────────────────
# URL parser (without urllib — avoids import overhead)
# ─────────────────────────────────────────────────────────────────────────────

def _parse_url(url: str) -> dict:
    result = {"scheme": "", "hostname": "", "path": "", "query": "", "fragment": ""}
    try:
        from urllib.parse import urlparse, parse_qs
        p = urlparse(url)
        result["scheme"]   = p.scheme
        result["hostname"] = p.hostname or ""
        result["path"]     = p.path
        result["query"]    = p.query
        result["n_params"] = len(parse_qs(p.query))
    except Exception:
        pass
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Single URL/domain feature extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_url_features(url: str, dns_record=None) -> np.ndarray:
    """
    Extract feature vector for a URL.

    Parameters
    ----------
    url        : raw URL string
    dns_record : DNSRecord (optional) — if provided, DNS features are computed

    Returns
    -------
    feat : (N_FEATURES,) float32 numpy array
    """
    feat = np.zeros(N_FEATURES, dtype=np.float32)

    parsed = _parse_url(url)
    hostname = parsed["hostname"] or url.replace("https://", "").replace("http://", "").split("/")[0]

    # Registered domain via tldextract
    if HAS_TLDEXTRACT:
        ext = tldextract.extract(url)
        reg_domain  = f"{ext.domain}.{ext.suffix}" if ext.suffix else ext.domain
        subdomain   = ext.subdomain
        domain_part = ext.domain
    else:
        reg_domain  = hostname
        subdomain   = ""
        domain_part = hostname.split(".")[0] if "." in hostname else hostname

    # ── Lexical features ──────────────────────────────────────────────────────
    feat[FEATURE_INDEX["url_length"]]      = len(url)
    feat[FEATURE_INDEX["domain_length"]]   = len(reg_domain)
    feat[FEATURE_INDEX["n_subdomains"]]    = len(subdomain.split(".")) if subdomain else 0
    feat[FEATURE_INDEX["has_ip_in_url"]]   = float(_has_ip_in_url(url))
    feat[FEATURE_INDEX["n_digits"]]        = sum(c.isdigit() for c in domain_part)
    dlen = max(len(domain_part), 1)
    feat[FEATURE_INDEX["digit_ratio"]]     = feat[FEATURE_INDEX["n_digits"]] / dlen
    feat[FEATURE_INDEX["n_hyphens"]]       = domain_part.count("-")
    feat[FEATURE_INDEX["n_dots"]]          = url.count(".")
    feat[FEATURE_INDEX["char_entropy"]]    = _entropy(domain_part)
    feat[FEATURE_INDEX["has_at_symbol"]]   = float("@" in url)
    feat[FEATURE_INDEX["has_double_slash"]]= float("//" in url[7:])  # skip scheme's //
    feat[FEATURE_INDEX["is_https"]]        = float(parsed["scheme"] == "https")
    feat[FEATURE_INDEX["url_path_length"]] = len(parsed["path"])
    feat[FEATURE_INDEX["n_query_params"]]  = parsed.get("n_params", 0)

    # Brand in subdomain (typosquat indicator)
    sub_lower = subdomain.lower()
    feat[FEATURE_INDEX["brand_in_subdomain"]] = float(
        any(brand in sub_lower for brand in KNOWN_BRANDS)
    )

    # Longest word in domain (split on - and .)
    tokens = re.split(r"[-.]", domain_part)
    feat[FEATURE_INDEX["longest_word_length"]] = max((len(t) for t in tokens), default=0)

    # ── DNS features ──────────────────────────────────────────────────────────
    if dns_record is not None:
        feat[FEATURE_INDEX["n_a_records"]]  = len(dns_record.a_records)
        feat[FEATURE_INDEX["n_ns_records"]] = len(dns_record.ns_records)
        feat[FEATURE_INDEX["n_mx_records"]] = len(dns_record.mx_records)
        feat[FEATURE_INDEX["has_txt_spf"]]  = float(
            any("v=spf" in t.lower() for t in dns_record.txt_records)
        )
        feat[FEATURE_INDEX["has_cname"]]    = float(bool(dns_record.cname_records))

    return feat


# ─────────────────────────────────────────────────────────────────────────────
# Batch extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_features_batch(
    entries,           # list[FeedEntry]
    dns_records=None,  # dict[str, DNSRecord]
    verbose: bool = True,
) -> np.ndarray:
    """
    Extract features for a list of FeedEntry objects.

    Returns
    -------
    X : (N, N_FEATURES) float32 array
    """
    if dns_records is None:
        dns_records = {}

    X = np.zeros((len(entries), N_FEATURES), dtype=np.float32)
    for i, entry in enumerate(entries):
        rec = dns_records.get(entry.domain)
        X[i] = extract_url_features(entry.url, dns_record=rec)

    if verbose:
        print(f"Features extracted: {X.shape[0]} samples × {X.shape[1]} features")
        nan_count = np.isnan(X).sum()
        if nan_count > 0:
            print(f"  Warning: {nan_count} NaN values — will be replaced with 0")
        X = np.nan_to_num(X, nan=0.0)

    return X


# ─────────────────────────────────────────────────────────────────────────────
# Feature summary / diagnostics
# ─────────────────────────────────────────────────────────────────────────────

def feature_summary(X: np.ndarray, y: np.ndarray) -> None:
    """Print mean and std of each feature split by label."""
    print(f"\n{'Feature':<25}  {'Benign mean':>12}  {'Phish mean':>12}  {'Diff/std':>10}")
    print("-" * 65)
    benign = X[y == 0]
    phish  = X[y == 1]
    for i, name in enumerate(FEATURE_NAMES):
        b_mean = benign[:, i].mean() if len(benign) > 0 else 0
        p_mean = phish[:, i].mean()  if len(phish)  > 0 else 0
        pooled_std = X[:, i].std() + 1e-8
        effect = abs(p_mean - b_mean) / pooled_std
        marker = " ◀ HIGH" if effect > 0.5 else ""
        print(f"  {name:<23}  {b_mean:>12.4f}  {p_mean:>12.4f}  {effect:>10.3f}{marker}")
