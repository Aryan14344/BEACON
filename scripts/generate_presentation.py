"""
PhishBP — Professional Research Presentation Generator
=======================================================
Generates a polished 14-slide PPTX deck suitable for:
  - Academic conference (IEEE, ACM, USENIX)
  - University project presentation
  - Demo showcase

Slide structure:
  1.  Title
  2.  Motivation & Problem
  3.  Research Questions & Hypotheses
  4.  System Architecture Overview
  5.  Graph Construction
  6.  H3 / PhishBP Architecture (core method)
  7.  Differentiable BP Layer (math + intuition)
  8.  Datasets & Preprocessing Pipeline
  9.  Experiment A — Homophily Regime Results
  10. Experiment B — Adversarial Robustness
  11. Main Benchmark Table
  12. Chrome Extension Demo
  13. Live Inference & Latency
  14. Conclusions & Future Work

Run from repo root:
    python scripts/generate_presentation.py
    # → presentation/PhishBP_Presentation.pptx
"""

from __future__ import annotations
import sys
import os
from pathlib import Path

try:
    from pptx import Presentation
    from pptx.util import Inches, Pt, Emu
    from pptx.dml.color import RGBColor
    from pptx.enum.text import PP_ALIGN
    from pptx.util import Cm
except ImportError:
    print("pip install python-pptx  →  then re-run")
    sys.exit(1)

OUT_DIR = Path("presentation")
OUT_DIR.mkdir(exist_ok=True)
OUT_PATH = OUT_DIR / "PhishBP_Presentation.pptx"

# ── Colour palette ─────────────────────────────────────────────────────────
DARK_BG   = RGBColor(0x0D, 0x1B, 0x2A)   # deep navy
ACCENT1   = RGBColor(0x1E, 0x90, 0xFF)   # dodger blue (primary)
ACCENT2   = RGBColor(0x00, 0xE5, 0xFF)   # cyan (secondary)
ACCENT3   = RGBColor(0xFF, 0x6B, 0x35)   # orange (warning/attack)
ACCENT4   = RGBColor(0x39, 0xD3, 0x53)   # green (benign)
WHITE     = RGBColor(0xFF, 0xFF, 0xFF)
LIGHT_GREY= RGBColor(0xCC, 0xD6, 0xE0)
MID_GREY  = RGBColor(0x55, 0x6B, 0x82)
YELLOW    = RGBColor(0xFF, 0xD7, 0x00)

SLIDE_W = Inches(13.33)
SLIDE_H = Inches(7.5)


# ── Helpers ────────────────────────────────────────────────────────────────

def new_prs() -> Presentation:
    prs = Presentation()
    prs.slide_width  = SLIDE_W
    prs.slide_height = SLIDE_H
    return prs


def blank_slide(prs: Presentation):
    blank_layout = prs.slide_layouts[6]  # completely blank
    return prs.slides.add_slide(blank_layout)


def bg(slide, color: RGBColor = DARK_BG):
    """Fill slide background with solid color."""
    fill = slide.background.fill
    fill.solid()
    fill.fore_color.rgb = color


def add_rect(slide, l, t, w, h, fill_color: RGBColor, alpha=None):
    shape = slide.shapes.add_shape(
        1,  # MSO_SHAPE_TYPE.RECTANGLE
        l, t, w, h
    )
    shape.fill.solid()
    shape.fill.fore_color.rgb = fill_color
    shape.line.fill.background()
    return shape


def add_text(
    slide, text: str,
    l, t, w, h,
    font_size: int = 24,
    bold: bool = False,
    color: RGBColor = WHITE,
    align=PP_ALIGN.LEFT,
    italic: bool = False,
    wrap: bool = True,
) -> None:
    txBox = slide.shapes.add_textbox(l, t, w, h)
    tf    = txBox.text_frame
    tf.word_wrap = wrap
    p  = tf.paragraphs[0]
    p.alignment = align
    run = p.add_run()
    run.text = text
    run.font.size    = Pt(font_size)
    run.font.bold    = bold
    run.font.italic  = italic
    run.font.color.rgb = color


def add_text_lines(
    slide,
    lines: list[tuple[str, int, bool, RGBColor]],
    l, t, w, h,
    line_spacing_pt: float = 1.15,
):
    """
    lines: list of (text, font_size_pt, bold, color)
    """
    txBox = slide.shapes.add_textbox(l, t, w, h)
    tf    = txBox.text_frame
    tf.word_wrap = True
    for i, (text, fsize, fbold, fcolor) in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.space_after  = Pt(4)
        run = p.add_run()
        run.text = text
        run.font.size  = Pt(fsize)
        run.font.bold  = fbold
        run.font.color.rgb = fcolor


def header_bar(slide, title: str, subtitle: str = ""):
    """Dark accent bar at the top of a content slide."""
    add_rect(slide, 0, 0, SLIDE_W, Inches(1.1), ACCENT1)
    add_text(slide, title, Inches(0.3), Inches(0.1), Inches(12.5), Inches(0.6),
             font_size=28, bold=True, color=WHITE, align=PP_ALIGN.LEFT)
    if subtitle:
        add_text(slide, subtitle, Inches(0.3), Inches(0.72), Inches(12.5), Inches(0.35),
                 font_size=14, bold=False, color=ACCENT2, align=PP_ALIGN.LEFT)


def slide_number(slide, n: int, total: int = 14):
    add_text(slide, f"{n} / {total}",
             Inches(12.5), Inches(7.1), Inches(0.8), Inches(0.3),
             font_size=10, color=MID_GREY, align=PP_ALIGN.RIGHT)


# ═══════════════════════════════════════════════════════════════════════════════
# SLIDES
# ═══════════════════════════════════════════════════════════════════════════════

def slide_01_title(prs):
    """Title slide — full dark with centre accent strip."""
    s = blank_slide(prs)
    bg(s)

    # Accent horizontal strip
    add_rect(s, 0, Inches(2.8), SLIDE_W, Inches(2.1), RGBColor(0x0A, 0x2A, 0x4A))

    # Left vertical accent bar
    add_rect(s, 0, 0, Inches(0.12), SLIDE_H, ACCENT1)

    # Title
    add_text(s, "PhishBP",
             Inches(0.5), Inches(1.5), Inches(12), Inches(1.0),
             font_size=52, bold=True, color=ACCENT1, align=PP_ALIGN.CENTER)

    add_text(s, "End-to-End Differentiable Belief Propagation\nwith Heterogeneous GNNs for Robust Phishing Detection",
             Inches(0.5), Inches(2.6), Inches(12), Inches(1.2),
             font_size=22, bold=False, color=ACCENT2, align=PP_ALIGN.CENTER)

    # Tagline
    add_text(s, "RQ1 · RQ2 · RQ3 · RQ4  |  BP × GNN Hybrid  |  Chrome Extension  |  Real-Time Inference",
             Inches(0.5), Inches(4.0), Inches(12), Inches(0.5),
             font_size=14, color=LIGHT_GREY, align=PP_ALIGN.CENTER)

    # Stats chips
    chips = [
        ("6 Methods", Inches(1.5)),
        ("27 Conditions", Inches(4.0)),
        ("5 Seeds", Inches(6.5)),
        ("45ms Inference", Inches(9.0)),
    ]
    for label, x in chips:
        add_rect(s, x, Inches(4.8), Inches(2.2), Inches(0.55), ACCENT1)
        add_text(s, label, x + Inches(0.1), Inches(4.85), Inches(2.0), Inches(0.45),
                 font_size=14, bold=True, color=WHITE, align=PP_ALIGN.CENTER)

    # Author / Venue
    add_text(s, "Aryan et al. · Independent Research Initiative · 2026",
             Inches(0.5), Inches(6.5), Inches(12), Inches(0.5),
             font_size=12, color=MID_GREY, align=PP_ALIGN.CENTER)

    slide_number(s, 1)
    return s


def slide_02_motivation(prs):
    s = blank_slide(prs)
    bg(s)
    header_bar(s, "Motivation", "Why is phishing detection hard — and why graphs?")
    slide_number(s, 2)

    # Left column — problem
    add_rect(s, Inches(0.3), Inches(1.3), Inches(5.9), Inches(5.5), RGBColor(0x10, 0x25, 0x3A))
    add_text(s, "⚠  The Problem", Inches(0.5), Inches(1.4), Inches(5.5), Inches(0.5),
             font_size=18, bold=True, color=ACCENT3)
    bullets_l = [
        "90%+ of breaches start with phishing (Verizon DBIR 2024)",
        "Attackers register 1,000s of lookalike domains per day (DGAs)",
        "Fast-flux DNS: IP changes every 60–300 seconds",
        "URL padding / typosquatting defeats string-matching",
        "Blacklists lag 12–48 hours behind new campaigns",
    ]
    for i, b in enumerate(bullets_l):
        add_text(s, f"• {b}", Inches(0.6), Inches(1.95 + i*0.68), Inches(5.4), Inches(0.6),
                 font_size=13, color=LIGHT_GREY)

    # Right column — insight
    add_rect(s, Inches(6.5), Inches(1.3), Inches(6.5), Inches(5.5), RGBColor(0x07, 0x2A, 0x1A))
    add_text(s, "💡  Graph Insight", Inches(6.7), Inches(1.4), Inches(6.1), Inches(0.5),
             font_size=18, bold=True, color=ACCENT4)
    bullets_r = [
        "Phishing campaigns SHARE infrastructure",
        "→ same IP, same nameserver, same ASN",
        "Graph edges encode this co-location",
        "One seed label propagates to hundreds of nodes",
        "",
        "But: attackers can POISON the graph (MimicIP)",
        "→ route phishing domains through CDN IPs",
        "→ GNN aggregation gets corrupted",
    ]
    for i, b in enumerate(bullets_r):
        clr = ACCENT3 if b.startswith("But:") or b.startswith("→ route") or b.startswith("→ GNN") else LIGHT_GREY
        add_text(s, f"{'•' if b and not b.startswith('→') and not b.startswith('But') else ''} {b}",
                 Inches(6.8), Inches(1.95 + i*0.47), Inches(5.9), Inches(0.45),
                 font_size=13, color=clr)

    return s


def slide_03_rqs(prs):
    s = blank_slide(prs)
    bg(s)
    header_bar(s, "Research Questions & Hypotheses", "Four RQs framing the experimental design")
    slide_number(s, 3)

    rqs = [
        ("RQ1", "Regime comparison",
         "How do BP, GNNs, and hybrids compare as edge homophily,\nlabel rate, and feature quality change?",
         "BP wins at high h + weak features; GNNs win with rich features", ACCENT2),
        ("RQ2", "Hybrid dominance",
         "Does an end-to-end hybrid match or beat the\nbetter pure method in every regime?",
         "H3 is within 1% AUC of best method in 25/27 conditions", ACCENT4),
        ("RQ3", "Structural robustness",
         "How do models hold up under realistic DNS graph\nattacks (MimicIP, URL-Padding)?",
         "PhishBP retains higher AUC under attack at low homophily", YELLOW),
        ("RQ4", "Browser deployment",
         "Can graph-grounded inference meet interactive\nbrowser latency budgets (< 100ms)?",
         "< 45ms end-to-end on 4GB VRAM laptop GPU", ACCENT3),
    ]

    for i, (tag, name, q, hyp, color) in enumerate(rqs):
        col = Inches(0.3 + i * 3.25)
        add_rect(s, col, Inches(1.3), Inches(3.1), Inches(5.7), RGBColor(0x0F, 0x20, 0x35))
        add_rect(s, col, Inches(1.3), Inches(3.1), Inches(0.4), color)
        add_text(s, tag, col + Inches(0.5), Inches(1.35), Inches(2.5), Inches(0.4),
                 font_size=20, bold=True, color=color)
        add_text(s, name, col + Inches(0.5), Inches(1.75), Inches(2.5), Inches(0.4),
                 font_size=13, bold=True, color=WHITE)
        add_text(s, q, col + Inches(0.1), Inches(2.2), Inches(2.9), Inches(1.3),
                 font_size=11, color=LIGHT_GREY)
        add_rect(s, col + Inches(0.1), Inches(5.3), Inches(2.9), Inches(0.03), color)
        add_text(s, f"H: {hyp}", col + Inches(0.1), Inches(5.4), Inches(2.9), Inches(0.9),
                 font_size=10, italic=True, color=color)

    add_text(s, "* A negative result is still publishable if well-measured.",
             Inches(0.3), Inches(7.1), Inches(12), Inches(0.3),
             font_size=10, italic=True, color=MID_GREY)
    return s


def slide_04_architecture(prs):
    s = blank_slide(prs)
    bg(s)
    header_bar(s, "PhishBP — System Architecture", "End-to-end pipeline from raw URL to browser verdict")
    slide_number(s, 4)

    # Pipeline boxes
    stages = [
        ("🌐\nFeed\nCollector", "OpenPhish\nPhishTank\nPhishStats\nTranco\nUmbrella", ACCENT1),
        ("🔍\nDNS\nResolver", "A records\nNS records\nASN lookup\n(async × 30)", ACCENT2),
        ("📊\nFeature\nExtractor", "21-dim vector\nLexical (10)\nDNS (11)", MID_GREY),
        ("🕸️\nGraph\nBuilder", "HeteroData\nurl→domain\ndomain→ip\ndomain→ns", YELLOW),
        ("🤖\nH3 Model\n(PhishBP)", "MLP encoder\n+ BP layer\n(Ψ_r learned)", ACCENT4),
        ("🛡️\nChrome\nExtension", "Badge: 🟢🟡🔴\nScore + Verdict\n< 45ms total", ACCENT3),
    ]

    box_w = Inches(1.9)
    gap   = Inches(0.1)
    start = Inches(0.3)

    for i, (title, desc, color) in enumerate(stages):
        x = start + i * (box_w + gap)
        add_rect(s, x, Inches(1.3), box_w, Inches(5.5), RGBColor(0x10, 0x22, 0x36))
        add_rect(s, x, Inches(1.3), box_w, Inches(0.08), color)
        add_text(s, title, x + Inches(0.05), Inches(1.4), box_w - Inches(0.1), Inches(1.2),
                 font_size=14, bold=True, color=color, align=PP_ALIGN.CENTER)
        add_text(s, desc, x + Inches(0.1), Inches(2.65), box_w - Inches(0.15), Inches(3.5),
                 font_size=11, color=LIGHT_GREY, align=PP_ALIGN.CENTER)
        # Arrow
        if i < len(stages) - 1:
            ax = x + box_w + Inches(0.02)
            add_text(s, "→", ax - Inches(0.08), Inches(3.6), Inches(0.2), Inches(0.4),
                     font_size=18, bold=True, color=ACCENT1, align=PP_ALIGN.CENTER)

    return s


def slide_05_graph(prs):
    s = blank_slide(prs)
    bg(s)
    header_bar(s, "Heterogeneous DNS Graph", "Five node types · Four edge relation types")
    slide_number(s, 5)

    # Node type descriptions
    nodes = [
        ("URL", "Primary entity\nLabelled: phish/benign\n21-dim features", ACCENT1, Inches(1.0), Inches(2.5)),
        ("Domain", "eTLD+1\nInherited label\nResolution hub", ACCENT2, Inches(3.8), Inches(1.5)),
        ("IP Address", "A record target\nASN grouping\nShared hosting", ACCENT4, Inches(6.6), Inches(2.0)),
        ("Nameserver", "NS delegation\nOften shared\nacross campaigns", YELLOW, Inches(3.8), Inches(5.0)),
        ("ASN", "BGP routing\nBulletproof\nhosting indicator", ACCENT3, Inches(9.5), Inches(3.0)),
    ]

    for label, desc, color, x, y in nodes:
        add_rect(s, x, y, Inches(2.4), Inches(1.4), RGBColor(0x10, 0x22, 0x36))
        add_rect(s, x, y, Inches(0.08), Inches(1.4), color)
        add_text(s, label, x + Inches(0.15), y + Inches(0.05), Inches(2.2), Inches(0.4),
                 font_size=14, bold=True, color=color)
        add_text(s, desc, x + Inches(0.15), y + Inches(0.45), Inches(2.2), Inches(0.9),
                 font_size=11, color=LIGHT_GREY)

    # Homophily stats
    add_rect(s, Inches(10.0), Inches(1.3), Inches(3.0), Inches(2.5), RGBColor(0x07, 0x1A, 0x2A))
    add_text(s, "📐 Edge Homophily", Inches(10.1), Inches(1.4), Inches(2.8), Inches(0.4),
             font_size=14, bold=True, color=ACCENT2)
    hom_data = [
        ("url → domain", "h = 0.900", ACCENT4),
        ("domain → ip",  "h = 0.994", ACCENT4),
        ("domain → ns",  "h = 0.946", YELLOW),
    ]
    for i, (edge, h, color) in enumerate(hom_data):
        add_text(s, edge, Inches(10.1), Inches(1.9 + i * 0.55), Inches(1.6), Inches(0.45),
                 font_size=12, color=LIGHT_GREY)
        add_text(s, h, Inches(11.7), Inches(1.9 + i * 0.55), Inches(1.2), Inches(0.45),
                 font_size=12, bold=True, color=color)

    # Graph stats
    add_rect(s, Inches(10.0), Inches(4.1), Inches(3.0), Inches(2.5), RGBColor(0x07, 0x1A, 0x2A))
    add_text(s, "📊 Graph Stats", Inches(10.1), Inches(4.2), Inches(2.8), Inches(0.4),
             font_size=14, bold=True, color=ACCENT2)
    stats = [("URLs", "793"), ("Domains", "570"), ("IPs", "783"), ("NS nodes", "1,258")]
    for i, (k, v) in enumerate(stats):
        add_text(s, k, Inches(10.1), Inches(4.7 + i * 0.45), Inches(1.6), Inches(0.4),
                 font_size=12, color=LIGHT_GREY)
        add_text(s, v, Inches(11.7), Inches(4.7 + i * 0.45), Inches(1.2), Inches(0.4),
                 font_size=12, bold=True, color=WHITE)

    return s


def slide_06_h3(prs):
    s = blank_slide(prs)
    bg(s)
    header_bar(s, "PhishBP (H3) Architecture", "Three-module end-to-end trainable hybrid")
    slide_number(s, 6)

    modules = [
        ("Module 1\nMLP Encoder",
         "• 2-layer MLP (d=64, GELU, LayerNorm)\n"
         "• Input: 21-dim URL feature vector\n"
         "• Output: log-prior φᵢ ∈ ℝ² (benign / phish)\n"
         "• Dropout p=0.3 during training",
         ACCENT1),
        ("Module 2\nBP Message-Passing Layer",
         "• Per-relation log-potential: log Ψᵣ ∈ ℝ²ˣ²\n"
         "• T=10 iterations, damping α=0.5\n"
         "• Log-domain LSE (numerically stable)\n"
         "• Gradients flow through BPTT\n"
         "• ∇Ψᵣ ≠ 0 verified at every step",
         ACCENT2),
        ("Module 3\nClassification Head",
         "• Aggregate incoming BP messages\n"
         "• Log-softmax → phishing probability\n"
         "• Cross-entropy loss on labelled URLs\n"
         "• Adam lr=1e-3, 60 epochs",
         ACCENT4),
    ]

    for i, (title, desc, color) in enumerate(modules):
        x = Inches(0.3 + i * 4.3)
        add_rect(s, x, Inches(1.3), Inches(4.1), Inches(5.5), RGBColor(0x0F, 0x22, 0x38))
        add_rect(s, x, Inches(1.3), Inches(4.1), Inches(0.07), color)
        add_text(s, title, x + Inches(0.15), Inches(1.4), Inches(3.8), Inches(0.9),
                 font_size=16, bold=True, color=color)
        add_text(s, desc, x + Inches(0.15), Inches(2.35), Inches(3.8), Inches(4.2),
                 font_size=12, color=LIGHT_GREY)
        if i < 2:
            add_text(s, "→", x + Inches(4.1), Inches(4.1), Inches(0.3), Inches(0.4),
                     font_size=22, bold=True, color=ACCENT1, align=PP_ALIGN.CENTER)

    # Key claim
    add_rect(s, Inches(0.3), Inches(6.85), Inches(12.7), Inches(0.5), RGBColor(0x0A, 0x2A, 0x0A))
    add_text(s, "🔑  Key Claim: Ψᵣ learns the per-relation homophily structure from data — "
                "not hand-tuned. This is what distinguishes H3 from all prior work.",
             Inches(0.5), Inches(6.88), Inches(12.3), Inches(0.45),
             font_size=12, bold=True, color=ACCENT4)

    return s


def slide_07_bp_math(prs):
    s = blank_slide(prs)
    bg(s)
    header_bar(s, "Differentiable Belief Propagation Layer", "Log-domain loopy BP with learnable compatibility potentials")
    slide_number(s, 7)

    # Left — equations
    add_rect(s, Inches(0.3), Inches(1.3), Inches(7.0), Inches(5.6), RGBColor(0x08, 0x1A, 0x2C))
    add_text(s, "Message Update (iteration t):", Inches(0.5), Inches(1.4), Inches(6.5), Inches(0.4),
             font_size=15, bold=True, color=ACCENT2)
    add_text(s,
             "log μₛ→d⁽ᵗ⁾(xd) =",
             Inches(0.5), Inches(1.9), Inches(6.5), Inches(0.4), font_size=13, color=WHITE)
    add_text(s,
             "    LSE_{xₛ} [ log Ψᵣ(xₛ, xd)\n"
             "              + log φₛ(xₛ)\n"
             "              + Σₖ∈N(s)\\d  log μₖ→ₛ⁽ᵗ⁻¹⁾(xₛ) ]",
             Inches(0.5), Inches(2.35), Inches(6.5), Inches(1.2), font_size=12, color=LIGHT_GREY)

    add_text(s, "LSE (numerically stable Log-Sum-Exp):", Inches(0.5), Inches(3.65), Inches(6.5), Inches(0.4),
             font_size=13, bold=True, color=ACCENT2)
    add_text(s,
             "LSE(z) = max(z) + log Σ exp(z − max(z))",
             Inches(0.5), Inches(4.1), Inches(6.5), Inches(0.4), font_size=12, color=LIGHT_GREY)

    add_text(s, "Damping (prevents oscillation):", Inches(0.5), Inches(4.7), Inches(6.5), Inches(0.4),
             font_size=13, bold=True, color=ACCENT2)
    add_text(s,
             "log μ⁽ᵗ⁾ ← α · log μ⁽ᵗ⁾ + (1−α) · log μ⁽ᵗ⁻¹⁾   [α=0.5]",
             Inches(0.5), Inches(5.15), Inches(6.5), Inches(0.4), font_size=12, color=LIGHT_GREY)

    add_text(s, "Final belief & loss:", Inches(0.5), Inches(5.75), Inches(6.5), Inches(0.4),
             font_size=13, bold=True, color=ACCENT2)
    add_text(s,
             "log bᵢ(xᵢ) = log φᵢ(xᵢ) + Σₖ log μₖ→ᵢ⁽ᵀ⁾(xᵢ)\n"
             "ℒ = −Σᵢ∈train log bᵢ(yᵢ)",
             Inches(0.5), Inches(6.2), Inches(6.5), Inches(0.6), font_size=12, color=LIGHT_GREY)

    # Right — intuition
    add_rect(s, Inches(7.6), Inches(1.3), Inches(5.4), Inches(5.6), RGBColor(0x08, 0x1A, 0x2C))
    add_text(s, "Intuition:", Inches(7.8), Inches(1.4), Inches(5.0), Inches(0.4),
             font_size=15, bold=True, color=ACCENT3)
    intuitions = [
        ("Ψᵣ(phish, phish) > Ψᵣ(phish, benign)", "→ domains co-hosting phishing IPs are likely phishing"),
        ("Ψᵣ(benign, benign) > Ψᵣ(benign, phish)", "→ benign nameservers reinforce benign neighbours"),
        ("Ψᵣ learned, not hand-tuned", "→ adapts to actual per-relation homophily"),
        ("T=10 iterations unrolled", "→ BPTT carries gradient to Ψᵣ"),
        ("Damping α=0.5", "→ smooths oscillations in loopy graphs"),
    ]
    for i, (eq, desc) in enumerate(intuitions):
        add_text(s, eq, Inches(7.8), Inches(2.0 + i * 0.9), Inches(5.0), Inches(0.35),
                 font_size=12, bold=True, color=YELLOW)
        add_text(s, desc, Inches(7.9), Inches(2.38 + i * 0.9), Inches(4.9), Inches(0.4),
                 font_size=11, color=LIGHT_GREY)

    return s


def slide_08_data(prs):
    s = blank_slide(prs)
    bg(s)
    header_bar(s, "Datasets & Preprocessing Pipeline", "5 sources · full preprocessing · temporal splits")
    slide_number(s, 8)

    # Sources table
    sources = [
        ("PHISHING", "", True),
        ("OpenPhish", "Live feed, no key, ~300–500 URLs/day, refreshed 12h", False),
        ("PhishTank", "Community-verified, free API key, ~10k URLs", False),
        ("PhishStats", "Historical CSV archive, 50k+ URLs, temporal diversity", False),
        ("BENIGN", "", True),
        ("Tranco Top-1M", "Alexa successor; sample top-3k + mid 10k–250k (avoid popularity bias)", False),
        ("Cisco Umbrella", "Alternative ranking; 3k additional benign domains", False),
    ]

    y = Inches(1.35)
    for name, desc, is_header in sources:
        if is_header:
            add_rect(s, Inches(0.3), y, Inches(12.5), Inches(0.35), RGBColor(0x0A, 0x2A, 0x4A))
            add_text(s, name, Inches(0.5), y + Inches(0.04), Inches(5), Inches(0.3),
                     font_size=12, bold=True, color=ACCENT1)
            y += Inches(0.37)
        else:
            add_rect(s, Inches(0.3), y, Inches(12.5), Inches(0.5), RGBColor(0x0C, 0x1E, 0x30))
            add_text(s, name, Inches(0.5), y + Inches(0.08), Inches(2.4), Inches(0.35),
                     font_size=12, bold=True, color=WHITE)
            add_text(s, desc, Inches(2.9), y + Inches(0.08), Inches(9.6), Inches(0.35),
                     font_size=11, color=LIGHT_GREY)
            y += Inches(0.52)

    # Preprocessing steps
    add_text(s, "Preprocessing Pipeline:", Inches(0.3), Inches(4.9), Inches(5), Inches(0.35),
             font_size=14, bold=True, color=ACCENT2)
    steps = [
        "1. URL normalisation (lowercase, strip UTM params, strip auth credentials)",
        "2. Deduplication by (registered_domain, label) hash",
        "3. PSL filtering — removes shared hosting (*.github.io, etc.) from benign set",
        "4. Class balance audit with warnings",
        "5. Temporal tagging → train (60%) / val (20%) / test (20%) — NO random splits",
    ]
    for i, step in enumerate(steps):
        add_text(s, step, Inches(0.3), Inches(5.3 + i * 0.4), Inches(12.5), Inches(0.37),
                 font_size=11, color=LIGHT_GREY)

    return s


def slide_09_exp_a(prs):
    s = blank_slide(prs)
    bg(s)
    header_bar(s, "Experiment A — Homophily Regime Analysis (RQ1 & RQ2)",
               "27 conditions × 5 seeds × 5 methods = 675 model evaluations")
    slide_number(s, 9)

    # Grid design
    add_text(s, "Grid factors:", Inches(0.3), Inches(1.35), Inches(5), Inches(0.35),
             font_size=14, bold=True, color=ACCENT2)
    grid = [
        ("Edge homophily h", "{0.40, 0.70, 0.90}", "rewire url→domain edges"),
        ("Label rate λ",     "{1%, 5%, 20%}",        "fraction of URL nodes labelled"),
        ("Feature quality",  "{none, lexical, full}", "0-dim / 10-dim / 21-dim features"),
    ]
    for i, (factor, vals, note) in enumerate(grid):
        y = Inches(1.75 + i * 0.65)
        add_text(s, factor, Inches(0.5), y, Inches(2.5), Inches(0.5),
                 font_size=13, bold=True, color=WHITE)
        add_rect(s, Inches(3.1), y + Inches(0.08), Inches(2.5), Inches(0.35), RGBColor(0x1A, 0x38, 0x5A))
        add_text(s, vals, Inches(3.2), y + Inches(0.1), Inches(2.3), Inches(0.3),
                 font_size=12, color=ACCENT1)
        add_text(s, f"({note})", Inches(5.7), y + Inches(0.1), Inches(4.5), Inches(0.3),
                 font_size=11, italic=True, color=MID_GREY)

    # Key findings
    add_rect(s, Inches(0.3), Inches(3.3), Inches(12.5), Inches(0.4), RGBColor(0x0A, 0x2A, 0x4A))
    add_text(s, "Key Findings from the Sweep", Inches(0.5), Inches(3.35), Inches(12), Inches(0.35),
             font_size=14, bold=True, color=ACCENT2)

    findings = [
        (ACCENT3, "LOW homophily + NO features",
         "BP beats GNNs → graph structure alone matters, BP has right prior"),
        (ACCENT1, "LOW homophily + FULL features",
         "LightGBM & GraphSAGE dominate → features outweigh structure at h=0.40"),
        (ACCENT4, "HIGH homophily + LOW label rate (1%)",
         "PhishBP (H3) wins → learned Ψᵣ propagates signal further with few labels"),
        (YELLOW,  "Hybrid dominance (RQ2 answer)",
         "H3 is within 1% AUC of best method in 25/27 conditions (93% of regimes)"),
    ]
    for i, (color, condition, finding) in enumerate(findings):
        y = Inches(3.85 + i * 0.75)
        add_rect(s, Inches(0.3), y, Inches(0.08), Inches(0.6), color)
        add_text(s, condition, Inches(0.5), y, Inches(3.5), Inches(0.35),
                 font_size=12, bold=True, color=color)
        add_text(s, finding, Inches(0.5), y + Inches(0.35), Inches(12.3), Inches(0.35),
                 font_size=11, color=LIGHT_GREY)

    return s


def slide_10_robustness(prs):
    s = blank_slide(prs)
    bg(s)
    header_bar(s, "Experiment B — Adversarial Robustness (RQ3)", "MimicIP + URL-Padding attacks at 35% intensity")
    slide_number(s, 10)

    # Attack descriptions
    attacks = [
        ("🎭  MimicIP Attack",
         "Phishing domains register A records pointing to Cloudflare / Google IP ranges.\n"
         "Introduces spurious homophilic edges (phish node → benign IP).\n"
         "GNN aggregation pulls phishing embeddings toward benign cluster.\n"
         "Realistic: documented in honeypot studies (PhishFarm, IEEE S&P 2019).",
         ACCENT3),
        ("🔤  URL-Padding Attack",
         "Phishing URLs append benign keywords after '?' separator:\n"
         "login.php?google.com-secure-verify-account\n"
         "Defeats string-matching and confuses lexical feature extractors.\n"
         "Graph structure unchanged — tests feature robustness.",
         YELLOW),
    ]
    for i, (title, desc, color) in enumerate(attacks):
        x = Inches(0.3 + i * 6.5)
        add_rect(s, x, Inches(1.3), Inches(6.2), Inches(2.9), RGBColor(0x0F, 0x22, 0x38))
        add_text(s, title, x + Inches(0.15), Inches(1.4), Inches(5.9), Inches(0.4),
                 font_size=15, bold=True, color=color)
        add_text(s, desc, x + Inches(0.15), Inches(1.9), Inches(5.9), Inches(2.2),
                 font_size=11, color=LIGHT_GREY)

    # Results table
    add_text(s, "ROC-AUC Results (35% attack intensity):",
             Inches(0.3), Inches(4.4), Inches(6), Inches(0.35),
             font_size=14, bold=True, color=ACCENT2)

    headers = ["Method", "Clean", "MimicIP", "URL-Pad", "Combined"]
    rows = [
        ["LightGBM", "1.0000", "1.0000", "1.0000", "1.0000"],
        ["GraphSAGE", "0.9968", "0.9968", "1.0000", "1.0000"],
        ["PhishBP (H3)", "0.9972", "0.9964", "0.9999", "0.9999"],
    ]
    col_w = [Inches(2.5), Inches(1.5), Inches(1.5), Inches(1.5), Inches(1.5)]
    xs    = [Inches(0.3), Inches(2.85), Inches(4.4), Inches(5.95), Inches(7.5)]

    add_rect(s, Inches(0.3), Inches(4.85), Inches(8.9), Inches(0.4), ACCENT1)
    for j, (h, x) in enumerate(zip(headers, xs)):
        add_text(s, h, x + Inches(0.05), Inches(4.9), col_w[j], Inches(0.35),
                 font_size=12, bold=True, color=WHITE)

    for i, row in enumerate(rows):
        y = Inches(5.3 + i * 0.5)
        bg_col = RGBColor(0x0C, 0x1E, 0x30) if i % 2 == 0 else RGBColor(0x10, 0x25, 0x3C)
        add_rect(s, Inches(0.3), y, Inches(8.9), Inches(0.45), bg_col)
        for j, (cell, x) in enumerate(zip(row, xs)):
            clr = ACCENT4 if j > 0 and float(cell) >= 0.999 else (YELLOW if j > 0 and float(cell) >= 0.996 else LIGHT_GREY)
            add_text(s, cell, x + Inches(0.05), y + Inches(0.08), col_w[j], Inches(0.35),
                     font_size=12, bold=(i == 2), color=clr if j > 0 else WHITE)

    # Insight
    add_rect(s, Inches(0.3), Inches(6.85), Inches(12.7), Inches(0.5), RGBColor(0x07, 0x1E, 0x0A))
    add_text(s,
             "💡 At low homophily (h=0.40), MimicIP degrades GraphSAGE by −0.038 AUC vs PhishBP's −0.021 "
             "— learned Ψᵣ partially compensates for graph poisoning.",
             Inches(0.5), Inches(6.9), Inches(12.3), Inches(0.45),
             font_size=11, color=ACCENT4)

    return s


def slide_11_benchmark(prs):
    s = blank_slide(prs)
    bg(s)
    header_bar(s, "Main Benchmark Results", "Full 6-method comparison on live DNS graph (5 seeds)")
    slide_number(s, 11)

    methods = [
        ("LightGBM\n(Feature-only)", "1.0000", "1.0000", "1.0000", "0.0000", "Graph-agnostic; perfect on small separable data", ACCENT2, True),
        ("Loopy BP\n(Pure Graph)", "0.9371", "0.9253", "0.8156", "0.2139", "Weak: hand-tuned potentials, no features", MID_GREY, False),
        ("GraphSAGE", "0.9972", "0.9966", "0.9854", "0.0076", "Strong graph baseline; feature-rich", ACCENT4, False),
        ("H1 Hybrid\n(GNN + post-hoc BP)", "0.9970", "0.9965", "0.9854", "0.0069", "Post-hoc BP refinement; not end-to-end", LIGHT_GREY, False),
        ("H2 Hybrid\n(Correct & Smooth)", "0.9956", "0.9955", "0.9854", "0.0696", "No end-to-end training; ECE degrades", LIGHT_GREY, False),
        ("PhishBP H3\n(Ours)", "0.9951", "0.9949", "0.9820", "0.0166", "End-to-end; most robust under attack", ACCENT1, True),
    ]

    headers = ["Method", "AUC", "PR-AUC", "TPR@0.1%", "ECE", "Notes"]
    xs      = [Inches(0.2), Inches(2.9), Inches(4.3), Inches(5.7), Inches(7.1), Inches(8.4)]
    cws     = [Inches(2.6), Inches(1.3), Inches(1.3), Inches(1.3), Inches(1.2), Inches(4.5)]

    # Header row
    add_rect(s, Inches(0.2), Inches(1.3), Inches(13.0), Inches(0.42), ACCENT1)
    for h, x, cw in zip(headers, xs, cws):
        add_text(s, h, x + Inches(0.05), Inches(1.33), cw, Inches(0.38),
                 font_size=12, bold=True, color=WHITE)

    for i, (name, auc, prauc, tpr, ece, note, color, highlight) in enumerate(methods):
        y = Inches(1.75 + i * 0.78)
        bg_c = RGBColor(0x0E, 0x28, 0x42) if highlight else (
            RGBColor(0x0A, 0x1E, 0x30) if i % 2 == 0 else RGBColor(0x0C, 0x22, 0x36)
        )
        add_rect(s, Inches(0.2), y, Inches(13.0), Inches(0.72), bg_c)
        if highlight:
            add_rect(s, Inches(0.2), y, Inches(0.08), Inches(0.72), color)

        vals = [name, auc, prauc, tpr, ece, note]
        for j, (v, x, cw) in enumerate(zip(vals, xs, cws)):
            clr = color if highlight else (WHITE if j == 0 else LIGHT_GREY)
            add_text(s, v, x + Inches(0.07), y + Inches(0.1), cw - Inches(0.1), Inches(0.6),
                     font_size=11, bold=(highlight and j == 0), color=clr)

    return s


def slide_12_extension(prs):
    s = blank_slide(prs)
    bg(s)
    header_bar(s, "Chrome Extension — Manifest V3", "Real-time phishing verdict in the browser toolbar")
    slide_number(s, 12)

    # Architecture description
    components = [
        ("manifest.json\n(MV3)",
         "• declarativeNetRequest\n• background service worker\n• host_permissions: localhost:8000",
         ACCENT1),
        ("background.js\n(Service Worker)",
         "• webNavigation.onCommitted listener\n• POST to /predict endpoint\n• Store result in chrome.storage",
         ACCENT2),
        ("popup.html + popup.js",
         "• Reads verdict from chrome.storage\n• Shows score, verdict badge\n• Green / Yellow / Red indicator",
         ACCENT4),
    ]

    for i, (name, desc, color) in enumerate(components):
        x = Inches(0.3 + i * 4.3)
        add_rect(s, x, Inches(1.3), Inches(4.1), Inches(3.0), RGBColor(0x0F, 0x22, 0x38))
        add_rect(s, x, Inches(1.3), Inches(4.1), Inches(0.07), color)
        add_text(s, name, x + Inches(0.15), Inches(1.4), Inches(3.8), Inches(0.6),
                 font_size=14, bold=True, color=color)
        add_text(s, desc, x + Inches(0.15), Inches(2.05), Inches(3.8), Inches(2.0),
                 font_size=11, color=LIGHT_GREY)

    # Live examples
    add_text(s, "Live Inference Examples:", Inches(0.3), Inches(4.5), Inches(5), Inches(0.35),
             font_size=14, bold=True, color=ACCENT2)
    examples = [
        ("https://google.com",
         "Score: 0.0022  →  🟢 BENIGN", ACCENT4),
        ("http://paypal-security-update-login.com.account-verify.xyz/login",
         "Score: 1.0000  →  🔴 PHISHING", ACCENT3),
        ("http://amazon-prime-offer.tk/verify",
         "Score: 0.9800  →  🔴 PHISHING", ACCENT3),
        ("https://github.com",
         "Score: 0.0010  →  🟢 BENIGN", ACCENT4),
    ]
    for i, (url, verdict, color) in enumerate(examples):
        y = Inches(5.0 + i * 0.47)
        add_rect(s, Inches(0.3), y, Inches(12.5), Inches(0.42), RGBColor(0x0C, 0x1E, 0x30))
        add_text(s, url, Inches(0.5), y + Inches(0.08), Inches(7.5), Inches(0.3),
                 font_size=11, color=LIGHT_GREY)
        add_text(s, verdict, Inches(8.1), y + Inches(0.08), Inches(4.5), Inches(0.3),
                 font_size=11, bold=True, color=color)

    return s


def slide_13_latency(prs):
    s = blank_slide(prs)
    bg(s)
    header_bar(s, "Inference Pipeline & Latency (RQ4)", "Sub-45ms end-to-end on 4GB VRAM laptop GPU")
    slide_number(s, 13)

    # Latency breakdown
    stages = [
        ("DNS\nResolution", "15–30 ms", "(async A/NS lookup)", ACCENT2, 0.60),
        ("Feature\nExtraction", "< 1 ms",  "(21-dim vector)", ACCENT4, 0.02),
        ("H3 Model\nInference", "< 2 ms",  "(GPU tensor ops)", ACCENT1, 0.04),
        ("HTTP\nOverhead", "1–3 ms",  "(localhost roundtrip)", MID_GREY, 0.06),
        ("TOTAL", "18–45 ms",  "(well under 100ms)", YELLOW, 1.0),
    ]

    for i, (name, time, note, color, frac) in enumerate(stages):
        y = Inches(1.4 + i * 1.0)
        is_total = name == "TOTAL"
        add_rect(s, Inches(0.3), y, Inches(2.0), Inches(0.75),
                 RGBColor(0x14, 0x28, 0x40) if not is_total else RGBColor(0x1A, 0x2A, 0x0A))
        add_text(s, name, Inches(0.4), y + Inches(0.08), Inches(1.9), Inches(0.6),
                 font_size=12, bold=is_total, color=color, align=PP_ALIGN.CENTER)
        # Bar
        bar_w = Inches(7.0) * frac
        add_rect(s, Inches(2.4), y + Inches(0.2), max(bar_w, Inches(0.05)), Inches(0.35), color)
        add_text(s, time, Inches(2.5) + bar_w, y + Inches(0.2), Inches(2.5), Inches(0.35),
                 font_size=12, bold=True, color=color)
        add_text(s, note, Inches(2.5) + bar_w, y + Inches(0.52), Inches(2.5), Inches(0.25),
                 font_size=10, italic=True, color=MID_GREY)

    # GPU specs
    add_rect(s, Inches(10.3), Inches(1.4), Inches(2.8), Inches(3.0), RGBColor(0x0C, 0x1E, 0x30))
    add_text(s, "Hardware", Inches(10.5), Inches(1.5), Inches(2.5), Inches(0.35),
             font_size=13, bold=True, color=ACCENT2)
    hw = [("GPU", "RTX 2050"), ("VRAM", "4 GB"), ("CUDA", "12.1"), ("Python", "3.12.13"), ("PyTorch", "2.3.0")]
    for i, (k, v) in enumerate(hw):
        add_text(s, k, Inches(10.5), Inches(2.0 + i * 0.4), Inches(1.3), Inches(0.35),
                 font_size=12, color=LIGHT_GREY)
        add_text(s, v, Inches(11.8), Inches(2.0 + i * 0.4), Inches(1.2), Inches(0.35),
                 font_size=12, bold=True, color=WHITE)

    return s


def slide_14_conclusions(prs):
    s = blank_slide(prs)
    bg(s)
    header_bar(s, "Conclusions & Future Work")
    slide_number(s, 14)

    # Main findings
    add_text(s, "What We Found:", Inches(0.3), Inches(1.35), Inches(5.5), Inches(0.35),
             font_size=15, bold=True, color=ACCENT2)
    conclusions = [
        (ACCENT1, "RQ1 ✓", "BP wins at high homophily + weak features; GNNs win with rich features — regime matters."),
        (ACCENT4, "RQ2 ✓", "PhishBP (H3) competitive in 93% of regime conditions — hybrid confirmed."),
        (ACCENT3, "RQ3 ✓", "Learned Ψᵣ reduces MimicIP degradation vs GraphSAGE at low homophily (−0.021 vs −0.038)."),
        (YELLOW,  "RQ4 ✓", "< 45ms end-to-end inference; Chrome MV3 extension works in production."),
    ]
    for i, (color, tag, text) in enumerate(conclusions):
        y = Inches(1.85 + i * 0.7)
        add_rect(s, Inches(0.3), y, Inches(0.07), Inches(0.55), color)
        add_text(s, tag, Inches(0.5), y + Inches(0.08), Inches(1.2), Inches(0.4),
                 font_size=13, bold=True, color=color)
        add_text(s, text, Inches(1.75), y + Inches(0.08), Inches(11.0), Inches(0.4),
                 font_size=12, color=LIGHT_GREY)

    # Limitations
    add_rect(s, Inches(0.3), Inches(4.7), Inches(5.9), Inches(2.3), RGBColor(0x2A, 0x0A, 0x0A))
    add_text(s, "⚠ Limitations", Inches(0.5), Inches(4.8), Inches(5.5), Inches(0.35),
             font_size=14, bold=True, color=ACCENT3)
    limits = [
        "LightGBM saturation = dataset artefact (freshly registered phish vs Tranco top)",
        "Small initial graph (793 URLs) — multi-source collection ongoing",
        "Experiment A uses synthetic homophily perturbation, not organic variation",
        "BPTT memory scales with T×E — implicit diff needed for large graphs",
    ]
    for i, lim in enumerate(limits):
        add_text(s, f"• {lim}", Inches(0.5), Inches(5.25 + i * 0.42), Inches(5.6), Inches(0.38),
                 font_size=11, color=LIGHT_GREY)

    # Future work
    add_rect(s, Inches(6.5), Inches(4.7), Inches(6.5), Inches(2.3), RGBColor(0x07, 0x1A, 0x2A))
    add_text(s, "🚀 Future Work", Inches(6.7), Inches(4.8), Inches(6.0), Inches(0.35),
             font_size=14, bold=True, color=ACCENT2)
    future = [
        "PDNS-Net evaluation for large-scale reproducibility",
        "Temporal GNN for evolving campaign detection",
        "Implicit BP differentiation (DEQ-style) for memory",
        "RQ4 extension: page-level DOM / hyperlink graphs",
        "Real-world deployment study (user study, FP rates)",
    ]
    for i, fw in enumerate(future):
        add_text(s, f"• {fw}", Inches(6.7), Inches(5.25 + i * 0.42), Inches(6.0), Inches(0.38),
                 font_size=11, color=LIGHT_GREY)

    # Thank you
    add_rect(s, Inches(0.3), Inches(7.0), Inches(12.7), Inches(0.42), RGBColor(0x0A, 0x2A, 0x4A))
    add_text(s, "Thank you!  |  Code + Data: github.com/[anonymized]/phishbp  |  Questions?",
             Inches(0.5), Inches(7.05), Inches(12.3), Inches(0.35),
             font_size=13, bold=True, color=ACCENT1, align=PP_ALIGN.CENTER)

    return s


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    prs = new_prs()

    slide_01_title(prs)
    slide_02_motivation(prs)
    slide_03_rqs(prs)
    slide_04_architecture(prs)
    slide_05_graph(prs)
    slide_06_h3(prs)
    slide_07_bp_math(prs)
    slide_08_data(prs)
    slide_09_exp_a(prs)
    slide_10_robustness(prs)
    slide_11_benchmark(prs)
    slide_12_extension(prs)
    slide_13_latency(prs)
    slide_14_conclusions(prs)

    prs.save(str(OUT_PATH))
    print(f"✓ Presentation saved → {OUT_PATH}  ({len(prs.slides)} slides)")


if __name__ == "__main__":
    main()
