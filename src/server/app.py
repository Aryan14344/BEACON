"""
PhishBP — FastAPI Real-Time Inference Server
Powers the Chrome Extension with:
  - Live async DNS resolution (A, NS, CNAME records)
  - 21-dim lexical + DNS feature extraction
  - H3 Hybrid (Learnable BP layer) inference on GPU
  - Swagger UI at http://127.0.0.1:8000/docs
"""

import sys
import os
from contextlib import asynccontextmanager
from typing import Optional, List

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ── Path setup ────────────────────────────────────────────────────────────────
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.pipeline.dns_resolver import resolve_domain
from src.features.lexical import extract_url_features, FEATURE_NAMES
from src.models.bp_layer import PhishBPHybridH3

# ── Constants ─────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_FEATURES = len(FEATURE_NAMES)
NUM_REL = 3        # url→domain(0), domain→ip(1), domain→ns(2)
MODEL: Optional[PhishBPHybridH3] = None


# ── Pydantic schemas ──────────────────────────────────────────────────────────

class PredictRequest(BaseModel):
    url: str
    model_config = {"json_schema_extra": {"example": {"url": "https://google.com"}}}


class PredictResponse(BaseModel):
    url: str
    domain: str
    is_phishing: bool
    phishing_score: float
    verdict: str        # "BENIGN" | "SUSPICIOUS" | "PHISHING"
    badge_color: str    # "green" | "yellow" | "red"
    dns_resolved: bool
    ips: List[str]
    nameservers: List[str]
    top_risk_factors: List[str]


# ── Lifespan (modern FastAPI startup / shutdown) ───────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    global MODEL
    MODEL = PhishBPHybridH3(
        in_features=N_FEATURES,
        hidden_dim=32,
        num_relations=NUM_REL,
        bp_iters=4,
        damping=0.5,
    ).to(DEVICE)

    ckpt = os.path.join(REPO_ROOT, "checkpoints", "h3_trained.pt")
    if os.path.exists(ckpt):
        MODEL.load_state_dict(torch.load(ckpt, map_location=DEVICE, weights_only=True))
        print(f"Loaded checkpoint: {ckpt}")
    else:
        print("WARNING: No checkpoint found — running with random weights.")

    MODEL.eval()
    print(f"PhishBP server ready on {DEVICE} ✓")
    yield
    print("PhishBP server shut down.")


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="PhishBP Protection API",
    version="1.0.0",
    description=(
        "Real-time phishing detection via Belief Propagation × Heterogeneous GNN.\n\n"
        "POST a URL to `/predict` and receive a verdict, phishing score, and DNS telemetry."
    ),
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/", tags=["Info"])
def root():
    return {
        "name": "PhishBP Protection API",
        "version": "1.0.0",
        "device": str(DEVICE),
        "model_loaded": MODEL is not None,
        "docs": "http://127.0.0.1:8000/docs",
        "usage": "POST /predict  with  {'url': 'https://example.com'}",
    }


@app.get("/health", tags=["Info"])
def health():
    return {
        "status": "healthy",
        "device": str(DEVICE),
        "model_loaded": MODEL is not None,
        "features": N_FEATURES,
    }


@app.post("/predict", response_model=PredictResponse, tags=["Inference"])
async def predict(req: PredictRequest):
    """
    Score a URL for phishing risk.

    1. Resolves live DNS records (A, NS, CNAME).
    2. Extracts 21-dim lexical + DNS feature vector.
    3. Builds a local star subgraph (domain → IPs + NSes).
    4. Runs the trained H3 model (encoder → BP message passing).
    5. Returns a verdict, risk score, and explanation.
    """
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet. Try again in a moment.")

    url = req.url.strip()
    if not (url.startswith("http://") or url.startswith("https://")):
        url = "http://" + url

    # Extract domain
    domain = url.replace("https://", "").replace("http://", "").split("/")[0].split(":")[0].split("@")[-1]

    # --- Step 1: Live DNS resolution ---
    dns_rec = await resolve_domain(domain, timeout=3.0)

    # --- Step 2: Feature extraction ---
    feat_vec = extract_url_features(url, dns_record=dns_rec)
    x = torch.from_numpy(feat_vec).to(DEVICE)

    # --- Step 3: Local star subgraph ---
    n_ips = len(dns_rec.a_records)
    n_ns  = len(dns_rec.ns_records)
    num_sub = 1 + n_ips + n_ns    # query node + IP nodes + NS nodes

    sub_x = torch.zeros((num_sub, N_FEATURES), device=DEVICE)
    sub_x[0] = x                  # query node carries all features

    src_l, dst_l, et_l = [], [], []
    for i in range(n_ips):
        src_l.append(0); dst_l.append(1 + i);        et_l.append(1)  # domain→ip
    for i in range(n_ns):
        src_l.append(0); dst_l.append(1 + n_ips + i); et_l.append(2) # domain→ns

    if src_l:
        sub_ei = torch.tensor([src_l, dst_l], dtype=torch.long, device=DEVICE)
        sub_et = torch.tensor(et_l,            dtype=torch.long, device=DEVICE)
    else:
        sub_ei = torch.zeros((2, 0), dtype=torch.long, device=DEVICE)
        sub_et = torch.zeros(0,      dtype=torch.long, device=DEVICE)

    # --- Step 4: Inference ---
    with torch.no_grad():
        # 1. Feature encoder output for query URL
        query_log_prior = F.log_softmax(MODEL.encoder(x.unsqueeze(0)), dim=-1)[0]
        feature_phish_prob = float(query_log_prior.exp()[1].cpu())

        # 2. Local BP graph message passing
        # Neighbor nodes (IPs, NS) have unknown labels -> assign uninformative neutral log-prior
        # [log(0.5), log(0.5)] so they don't artificially force the query node to benign
        if num_sub > 1 and sub_ei.shape[1] > 0:
            log_priors = torch.full((num_sub, 2), float(np.log(0.5)), device=DEVICE)
            log_priors[0] = query_log_prior
            log_beliefs = MODEL.bp_layer(log_priors, sub_ei, sub_et)
            graph_phish_prob = float(log_beliefs.exp()[0, 1].cpu())
        else:
            graph_phish_prob = feature_phish_prob

        # Base model phishing probability
        phish_prob = 0.6 * graph_phish_prob + 0.4 * feature_phish_prob

    # --- Step 5: Risk factors & Heuristic Calibration ---
    fi = {name: i for i, name in enumerate(FEATURE_NAMES)}
    risks = []
    heuristic_penalty = 0.0

    if feat_vec[fi["is_https"]] == 0.0:
        risks.append("Insecure HTTP connection (no TLS encryption)")
        heuristic_penalty += 0.25

    if feat_vec[fi["has_ip_in_url"]] == 1.0:
        risks.append("Raw IP address used as hostname")
        heuristic_penalty += 0.35

    if feat_vec[fi["brand_in_subdomain"]] == 1.0:
        risks.append("Brand keyword in subdomain (typosquatting)")
        heuristic_penalty += 0.40

    if feat_vec[fi["char_entropy"]] > 3.2:
        risks.append("High character entropy (random-looking domain)")
        heuristic_penalty += 0.20

    if feat_vec[fi["n_hyphens"]] > 2.0:
        risks.append("Excessive hyphens in domain name")
        heuristic_penalty += 0.15

    if feat_vec[fi["digit_ratio"]] > 0.35:
        risks.append("High digit ratio in domain name")
        heuristic_penalty += 0.20

    if not dns_rec.resolved:
        risks.append("Domain failed DNS resolution (NXDOMAIN / unresolvable)")
        heuristic_penalty += 0.50

    # Calibrate final risk score: model probability combined with observed security risks
    final_score = min(1.0, max(0.0, phish_prob + heuristic_penalty))

    # --- Verdict ---
    if final_score >= 0.65 or not dns_rec.resolved:
        verdict, badge, is_phish = "PHISHING",   "red",    True
    elif final_score >= 0.25 or len(risks) > 0:
        verdict, badge, is_phish = "SUSPICIOUS", "yellow", False
    else:
        verdict, badge, is_phish = "BENIGN",     "green",  False

    return PredictResponse(
        url=url,
        domain=domain,
        is_phishing=is_phish,
        phishing_score=round(final_score, 4),
        verdict=verdict,
        badge_color=badge,
        dns_resolved=dns_rec.resolved,
        ips=dns_rec.a_records[:5],
        nameservers=dns_rec.ns_records[:5],
        top_risk_factors=risks,
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=False)
