"""
Realistic graph-structure and evasion attacks on DNS/phishing graphs for RQ3:
1. MimicIP: Attacker maps malicious domains to benign IP addresses (mimicking popular hosting / CDN IPs).
2. URL-Padding: Attacker pads phishing URLs with legitimate keywords or subdomains to distort lexical priors.
3. Edge-Perturbation: Multi-domain graph perturbation (decoupling malicious infrastructure or injecting spurious links).
"""

import copy
import random
import torch
from torch import Tensor
from torch_geometric.data import HeteroData


def apply_mimicip_attack(
    data: HeteroData,
    attack_ratio: float = 0.3,
    seed: int = 42
) -> HeteroData:
    """
    MimicIP Attack:
    Attacker points their malicious domains to benign IP addresses.
    Threat model: Attacker controls DNS A records of their own domains.
    """
    torch.manual_seed(seed)
    attacked = copy.deepcopy(data)

    domain_y = attacked['domain'].y
    phish_domains = (domain_y == 1).nonzero(as_tuple=True)[0]
    benign_ips = (attacked['ip'].y == 0).nonzero(as_tuple=True)[0]

    if len(phish_domains) == 0 or len(benign_ips) == 0:
        return attacked

    # Select target phishing domains to perturb
    num_to_attack = max(1, int(len(phish_domains) * attack_ratio))
    perm = torch.randperm(len(phish_domains))
    target_domains = phish_domains[perm[:num_to_attack]]

    edge_type = ('domain', 'resolves_to', 'ip')
    ei = attacked[edge_type].edge_index

    # For each target domain, redirect its IP links or add links to benign IPs
    new_src = []
    new_dst = []
    for d in target_domains:
        # Pick 1-2 benign IPs to mimic
        chosen_ips = benign_ips[torch.randint(0, len(benign_ips), (2,))]
        for ip in chosen_ips:
            new_src.append(d.item())
            new_dst.append(ip.item())

    if new_src:
        attack_edges = torch.tensor([new_src, new_dst], dtype=torch.long, device=ei.device)
        attacked[edge_type].edge_index = torch.cat([ei, attack_edges], dim=1)

    return attacked


def apply_url_padding_attack(
    X_features: Tensor,
    y: Tensor,
    feature_names: list[str],
    attack_ratio: float = 0.3,
    seed: int = 42
) -> Tensor:
    """
    URL-Padding Attack:
    Attacker appends benign words, forces HTTPS, or pads URL paths to deceive feature-based models.
    Threat model: Attacker controls URL string structure.
    """
    torch.manual_seed(seed)
    X_attacked = X_features.clone()

    phish_mask = (y == 1)
    phish_indices = phish_mask.nonzero(as_tuple=True)[0]
    num_attack = int(len(phish_indices) * attack_ratio)

    perm = torch.randperm(len(phish_indices))
    selected = phish_indices[perm[:num_attack]]

    # Indices of manipulated features
    f_idx = {name: i for i, name in enumerate(feature_names)}
    
    if "is_https" in f_idx:
        X_attacked[selected, f_idx["is_https"]] = 1.0  # Force HTTPS
    if "url_length" in f_idx:
        X_attacked[selected, f_idx["url_length"]] += 30.0  # Pad path length
    if "brand_in_subdomain" in f_idx:
        X_attacked[selected, f_idx["brand_in_subdomain"]] = 0.0  # Conceal brand keyword
    if "has_ip_in_url" in f_idx:
        X_attacked[selected, f_idx["has_ip_in_url"]] = 0.0  # Use hostname instead of IP

    return X_attacked
