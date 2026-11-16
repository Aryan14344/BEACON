"""
FastAPI Real-Time Inference & Graph Resolution Server for PhishBP.
Powers the Chrome Extension:
  1. Accepts URLs via POST /predict
  2. Resolves DNS (A, NS, CNAME) dynamically
  3. Extracts lexical and infrastructure graph features
  4. Runs H3 Hybrid Model inference
  5. Returns phishing probability, safety badge, and risk explanations
"""

import sys
import os
import asyncio
import torch
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, HttpUrl
from typing import Optional, List, Dict

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.pipeline.dns_resolver import resolve_domain
from src.features.lexical import extract_url_features, FEATURE_NAMES
from src.models.bp_layer import PhishBPHybridH3

app = FastAPI(title="PhishBP Protection API", version="1.0.0")

# Enable CORS for browser extensions
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model state
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL: Optional[PhishBPHybridH3] = None
N_FEATURES = len(FEATURE_NAMES)
NUM_REL = 3  # (url->domain, domain->ip, domain->ns)


class PredictRequest(BaseModel):
    url: str


class PredictResponse(BaseModel):
    url: str
    domain: str
    is_phishing: bool
    phishing_score: float  # [0.0, 1.0]
    verdict: str           # "BENIGN", "SUSPICIOUS", "PHISHING"
    badge_color: str       # "green", "yellow", "red"
    dns_resolved: bool
    ips: List[str]
    nameservers: List[str]
    top_risk_factors: List[str]


@app.on_event("startup")
def load_model():
    global MODEL
    # Initialize and load H3 model
    MODEL = PhishBPHybridH3(
        in_features=N_FEATURES,
        hidden_dim=32,
        num_relations=NUM_REL,
        bp_iters=4,
        damping=0.5
    ).to(DEVICE)
    MODEL.eval()
    print(f"PhishBP Inference Server ready on {DEVICE}!")


@app.get("/health")
def health_check():
    return {"status": "healthy", "device": str(DEVICE), "model_loaded": MODEL is not None}


@app.post("/predict", response_model=PredictResponse)
async def predict(req: PredictRequest):
    url = req.url.strip()
    if not url.startswith("http://") and not url.startswith("https://"):
        url = "http://" + url

    # Extract domain
    domain = url.replace("https://", "").replace("http://", "").split("/")[0].split(":")[0]

    # Step 1: Live DNS lookup
    dns_rec = await resolve_domain(domain, timeout=2.5)

    # Step 2: Feature extraction
    feat_vec = extract_url_features(url, dns_record=dns_rec)
    x = torch.from_numpy(feat_vec).unsqueeze(0).to(DEVICE)

    # Step 3: Local subgraph inference via H3
    # Build local 1-hop star graph for the query node
    num_sub_nodes = 1 + len(dns_rec.a_records) + len(dns_rec.ns_records)
    sub_x = torch.zeros((num_sub_nodes, N_FEATURES), device=DEVICE)
    sub_x[0] = x[0]

    # Local edge topology: query -> IPs (rel 1), query -> NS (rel 2)
    src_list, dst_list, etype_list = [], [], []
    curr_node = 1
    for _ in dns_rec.a_records:
        src_list.append(0)
        dst_list.append(curr_node)
        etype_list.append(1)  # domain_resolves_to_ip
        curr_node += 1

    for _ in dns_rec.ns_records:
        src_list.append(0)
        dst_list.append(curr_node)
        etype_list.append(2)  # domain_uses_ns
        curr_node += 1

    if src_list:
        sub_ei = torch.tensor([src_list, dst_list], dtype=torch.long, device=DEVICE)
        sub_et = torch.tensor(etype_list, dtype=torch.long, device=DEVICE)
    else:
        sub_ei = torch.zeros((2, 0), dtype=torch.long, device=DEVICE)
        sub_et = torch.zeros(0, dtype=torch.long, device=DEVICE)

    with torch.no_grad():
        log_beliefs = MODEL(sub_x, sub_ei, sub_et)
        phish_prob = float(log_beliefs.exp()[0, 1].cpu().item())

    # Formulate risk factors
    risk_factors = []
    if feat_vec[FEATURE_NAMES.index("is_https")] == 0.0:
        risk_factors.append("Insecure plain HTTP connection")
    if feat_vec[FEATURE_NAMES.index("has_ip_in_url")] == 1.0:
        risk_factors.append("Direct IP address used as URL host")
    if feat_vec[FEATURE_NAMES.index("brand_in_subdomain")] == 1.0:
        risk_factors.append("Brand name detected in subdomain (Typosquatting)")
    if feat_vec[FEATURE_NAMES.index("char_entropy")] > 3.2:
        risk_factors.append("Unusually high domain character randomness/entropy")
    if not dns_rec.resolved:
        risk_factors.append("Domain failed DNS resolution or non-existent")

    # Determine verdict and badge
    if phish_prob > 0.70 or not dns_rec.resolved:
        verdict = "PHISHING"
        badge_color = "red"
        is_phishing = True
    elif phish_prob > 0.40:
        verdict = "SUSPICIOUS"
        badge_color = "yellow"
        is_phishing = False
    else:
        verdict = "BENIGN"
        badge_color = "green"
        is_phishing = False

    return PredictResponse(
        url=url,
        domain=domain,
        is_phishing=is_phishing,
        phishing_score=round(phish_prob, 4),
        verdict=verdict,
        badge_color=badge_color,
        dns_resolved=dns_rec.resolved,
        ips=dns_rec.a_records[:5],
        nameservers=dns_rec.ns_records[:5],
        top_risk_factors=risk_factors
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
