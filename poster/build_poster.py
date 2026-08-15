"""Build the AGU poster (60 x 40 in) with reportlab.

    python3 poster/build_poster.py

Geometry is measured off the supplied Canva template so the result drops into
the same board: black header band with the world-map texture, lime title strip,
three unequal columns split by hairline rules, coral section headers, and the
template's own section order -- References and Acknowledgement live at the foot
of column three.

The flow engine reports how full each column is, so overflow is caught here
rather than at the print shop.
"""
from __future__ import annotations

from pathlib import Path

from reportlab.lib.colors import HexColor, white
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas

ROOT = Path(__file__).resolve().parent.parent
ASSETS = ROOT / "poster/assets"
FIGS_PAPER = ROOT / "paper/figures"
FIGS_POSTER = ROOT / "poster/generated"
OUT_PDF = ROOT / "FSI-SLM_AGU_poster_60x40.pdf"

# --------------------------------------------------------------------------- #
# Page geometry, in points (72 pt = 1 in). Template mediabox is 4320 x 2880.
# --------------------------------------------------------------------------- #
W, H = 4320.0, 2880.0

HEADER_H = 400.0
STRIP_H = 96.0
BODY_TOP = H - HEADER_H - STRIP_H

COL_X = [0.0, 0.3111 * W, 0.6069 * W, W]   # divider positions from the template
PAD = 46.0                                  # inner padding per column
GUTTER_RULE = 2.2

BOTTOM_MARGIN = 30.0

DRY = False        # autofit passes skip image embedding and chrome
_IMG_SIZE = {}

# --------------------------------------------------------------------------- #
# Colours sampled from the template render
# --------------------------------------------------------------------------- #
BLACK = HexColor("#292929")
LIME = HexColor("#DDEB54")
CORAL = HexColor("#F47051")
INK = HexColor("#1A1A1A")
GREY = HexColor("#5C6266")
RULE = HexColor("#000000")

# --------------------------------------------------------------------------- #
# Fonts
# --------------------------------------------------------------------------- #
GF = "/usr/share/fonts/truetype/google-fonts"
LT = "/usr/share/fonts/truetype/lato"
pdfmetrics.registerFont(TTFont("Poppins", f"{GF}/Poppins-Regular.ttf"))
pdfmetrics.registerFont(TTFont("Poppins-Bold", f"{GF}/Poppins-Bold.ttf"))
pdfmetrics.registerFont(TTFont("Poppins-Med", f"{GF}/Poppins-Medium.ttf"))
pdfmetrics.registerFont(TTFont("Lato", f"{LT}/Lato-Regular.ttf"))
pdfmetrics.registerFont(TTFont("Lato-Bold", f"{LT}/Lato-Bold.ttf"))
pdfmetrics.registerFont(TTFont("Lato-It", f"{LT}/Lato-Italic.ttf"))
pdfmetrics.registerFont(TTFont("Lato-BoldIt", f"{LT}/Lato-BoldItalic.ttf"))

BODY_F, BODY_S, BODY_LEAD = "Lato", 28.0, 34.0
CAP_F, CAP_S, CAP_LEAD = "Lato-It", 22.0, 27.0
HEAD_F, HEAD_S = "Poppins-Bold", 52.0

TITLE_L1 = "A Fine-Tuned Small Language Model for Interpreting"
TITLE_L2 = "Fluid–Structure Interaction Experiments"
SUBTITLE = ("A U-Net surrogate stands in for the solver and the language model "
            "reads what comes out — the whole digital twin runs on one laptop")


# --------------------------------------------------------------------------- #
# Rich-text helpers: **bold** inside a paragraph
# --------------------------------------------------------------------------- #
def _tokens(text):
    """Split on ** into (chunk, is_bold) pairs."""
    out, bold = [], False
    for part in text.split("**"):
        if part:
            out.append((part, bold))
        bold = not bold
    return out


BOLD_OF = {"Lato": "Lato-Bold", "Lato-It": "Lato-BoldIt",
           "Poppins": "Poppins-Bold", "Poppins-Med": "Poppins-Bold"}


def _words(text):
    """Split into words, where a word is a list of (text, bold) runs.

    A word boundary is a space in the *source* string, so punctuation that
    follows a bold run stays welded to it -- otherwise '**record**.' renders
    as 'record .'
    """
    words, cur = [], []
    for chunk, bold in _tokens(text):
        parts = chunk.split(" ")
        for i, part in enumerate(parts):
            if i > 0:                      # a real space ended the last word
                if cur:
                    words.append(cur)
                cur = []
            if part:
                cur.append((part, bold))
    if cur:
        words.append(cur)
    return words


def _wid(word, font, size):
    return sum(pdfmetrics.stringWidth(t, BOLD_OF.get(font, font) if b else font,
                                      size) for t, b in word)


def wrap_rich(text, width, font, size):
    """Greedy wrap that respects bold runs. Returns a list of lines of words."""
    space = pdfmetrics.stringWidth(" ", font, size)
    lines, cur, cw = [], [], 0.0
    for word in _words(text):
        ww = _wid(word, font, size)
        add = ww if not cur else ww + space
        if cur and cw + add > width:
            lines.append(cur)
            cur, cw = [word], ww
        else:
            cur.append(word)
            cw += add
    if cur:
        lines.append(cur)
    return lines


# --------------------------------------------------------------------------- #
class Column:
    """Top-down flow inside one column."""

    def __init__(self, c, x0, x1, top, name, fig_scale=1.0, text_w=None):
        self.c = c
        self.x = x0 + PAD
        self.w = (x1 - x0) - 2 * PAD
        self.y = top
        self.top = top
        self.name = name
        self.fig_scale = fig_scale
        # A 22 in measure runs to 150 characters a line, which is unreadable.
        # Long-measure columns set a cap; the block is then centred in the
        # column so the figure above it stays optically aligned.
        self.text_w = min(text_w or self.w, self.w)

    # -- primitives -------------------------------------------------------- #
    def space(self, dy):
        self.y -= dy

    def header(self, text, gap_before=26, gap_after=14, size=HEAD_S,
               x=None, w=None):
        self.y -= gap_before
        xx = self.x if x is None else x
        ww = self.w if w is None else w
        self.c.setFont(HEAD_F, size)
        self.c.setFillColor(CORAL)
        self.c.drawCentredString(xx + ww / 2, self.y - size, text)
        self.y -= size + gap_after

    def para(self, text, size=BODY_S, lead=BODY_LEAD, font=BODY_F,
             color=INK, gap_after=20, indent=0.0):
        measure = min(self.text_w, self.w)
        x0 = self.x + (self.w - measure) / 2
        lines = wrap_rich(text, measure - indent, font, size)
        space = pdfmetrics.stringWidth(" ", font, size)
        self.c.setFillColor(color)
        for line in lines:
            xx = x0 + indent
            self.y -= lead
            for word in line:
                for t, bold in word:
                    f = BOLD_OF.get(font, font) if bold else font
                    self.c.setFont(f, size)
                    self.c.drawString(xx, self.y, t)
                    xx += pdfmetrics.stringWidth(t, f, size)
                xx += space
        self.y -= gap_after

    def bullets(self, items, size=BODY_S, lead=BODY_LEAD, gap_after=20,
                numbered=False):
        marker_w = 46.0
        for i, item in enumerate(items):
            mark = f"{i+1}." if numbered else "▪"
            self.c.setFont("Lato-Bold", size)
            self.c.setFillColor(CORAL)
            self.c.drawString(self.x + (self.w - min(self.text_w, self.w)) / 2,
                              self.y - lead, mark)
            self.para(item, size=size, lead=lead, gap_after=10,
                      indent=marker_w)
        self.y -= gap_after - 10

    def figure(self, path, caption=None, width=None, gap_before=8,
               gap_after=26, cap_size=CAP_S):
        if path not in _IMG_SIZE:
            from PIL import Image
            _IMG_SIZE[path] = Image.open(path).size
        iw, ih = _IMG_SIZE[path]
        w = (width or self.w) * self.fig_scale
        w = min(w, self.w)
        h = w * ih / iw
        self.y -= gap_before + h
        if not DRY:
            self.c.drawImage(ImageReader(str(path)), self.x + (self.w - w) / 2,
                             self.y, width=w, height=h, mask="auto")
        if caption:
            self.y -= 12
            self.para(caption, size=cap_size, lead=cap_size * 1.24,
                      font=CAP_F, color=GREY, gap_after=0)
        self.y -= gap_after

    def figure_pair(self, left, right, gutter=44, gap_before=8, gap_after=26,
                    cap_size=CAP_S):
        """Two figures side by side, captions beneath each."""
        x0, w0, y0 = self.x, self.w, self.y
        half = (w0 - gutter) / 2
        ends = []
        tw = self.text_w
        for i, (path, cap) in enumerate((left, right)):
            self.x = x0 + i * (half + gutter)
            self.w = half
            self.text_w = half
            self.y = y0
            self.figure(path, caption=cap, gap_before=gap_before,
                        gap_after=0, cap_size=cap_size)
            ends.append(self.y)
        self.x, self.w, self.text_w = x0, w0, tw
        self.y = min(ends) - gap_after

    def two_up(self, left, right, size=BODY_S, lead=BODY_LEAD, head_size=46,
               gutter=56, gap_after=10):
        """Two half-width sections side by side, as in the template."""
        x0, w0, y0 = self.x, self.w, self.y
        half = (w0 - gutter) / 2
        ends = []
        tw = self.text_w
        for i, (title, text) in enumerate((left, right)):
            self.x = x0 + i * (half + gutter)
            self.w = half
            self.text_w = half
            self.y = y0
            self.header(title, gap_before=0, gap_after=12, size=head_size)
            self.para(text, size=size, lead=lead, gap_after=0)
            ends.append(self.y)
        self.x, self.w, self.text_w = x0, w0, tw
        self.y = min(ends) - gap_after

    def rule(self, gap=16):
        self.y -= gap
        self.c.setStrokeColor(HexColor("#C9CDD0"))
        self.c.setLineWidth(1.6)
        self.c.line(self.x, self.y, self.x + self.w, self.y)
        self.y -= gap

    def report(self, bottom, quiet=False):
        used = self.top - self.y
        avail = self.top - bottom
        pct = 100 * used / avail
        flag = "  <-- OVERFLOW" if self.y < bottom else ""
        if not quiet:
            print(f"  {self.name:9s} used {used:7.0f} / {avail:7.0f} pt "
                  f"({pct:5.1f}%)  slack {self.y - bottom:7.0f}{flag}")
        return self.y - bottom


# --------------------------------------------------------------------------- #
def draw_chrome(c):
    """Header band, logos, title, lime strip, column rules."""
    # -- black header with the template's world-map texture ---------------- #
    hdr = ASSETS / "header_bg.png"
    if hdr.exists():
        c.drawImage(ImageReader(str(hdr)), 0, H - HEADER_H, width=W,
                    height=HEADER_H, mask="auto")
    else:
        c.setFillColor(BLACK)
        c.rect(0, H - HEADER_H, W, HEADER_H, stroke=0, fill=1)

    # -- logos, top right, matching the template stack --------------------- #
    from PIL import Image

    def logo(path, x_right, y_center, h):
        iw, ih = Image.open(path).size
        w = h * iw / ih
        c.setFillColor(white)
        c.rect(x_right - w - 12, y_center - h / 2 - 12, w + 24, h + 24,
               stroke=0, fill=1)
        c.drawImage(ImageReader(str(path)), x_right - w, y_center - h / 2,
                    width=w, height=h, mask="auto")

    logo(ASSETS / "logo_agu.png", 300, H - HEADER_H / 2, 132)
    logo(ASSETS / "logo_tompkins.png", W - 340, H - HEADER_H / 2, 132)
    logo(ASSETS / "logo_brown.png", W - 60, H - HEADER_H / 2, 132)

    # -- author block, inboard of the logos, as in the template ------------- #
    ax = W - 1290
    ay = H - 92
    c.setFillColor(white)
    c.setFont("Poppins-Bold", 36)
    c.drawString(ax, ay, "*Kumar, Angad")
    c.setFont("Lato", 30)
    c.drawString(ax, ay - 42, "angadkumar16ak@gmail.com")
    c.setFont("Lato-It", 30)
    c.drawString(ax, ay - 80, "Obra Tompkins High School, Texas, USA")
    c.setFont("Poppins-Bold", 36)
    c.drawString(ax, ay - 156, "Shukla, Khemraj")
    c.setFont("Lato", 30)
    c.drawString(ax, ay - 198, "khemraj_shukla@brown.edu")
    c.setFont("Lato-It", 30)
    c.drawString(ax, ay - 236, "Brown University, Rhode Island, USA")
    c.setFont("Lato-It", 24)
    c.setFillColor(HexColor("#B9C0C5"))
    c.drawString(ax, ay - 282, "*presenting author")

    # -- title, centred in the space the logos and authors leave ------------ #
    tl, tr = 460.0, W - 1330.0
    tc = (tl + tr) / 2
    avail = tr - tl

    ts = 82.0
    while max(pdfmetrics.stringWidth(TITLE_L1, "Poppins", ts),
              pdfmetrics.stringWidth(TITLE_L2, "Poppins", ts)) > avail:
        ts -= 1.0
    c.setFillColor(white)
    c.setFont("Poppins", ts)
    c.drawCentredString(tc, H - 150, TITLE_L1)
    c.drawCentredString(tc, H - 150 - ts * 1.24, TITLE_L2)

    ss = 40.0
    while pdfmetrics.stringWidth(SUBTITLE, "Lato-It", ss) > avail:
        ss -= 0.5
    c.setFont("Lato-It", ss)
    c.setFillColor(HexColor("#DFE3E5"))
    c.drawCentredString(tc, H - 336, SUBTITLE)

    # -- QR to the repository, in the dead space under the AGU mark --------- #
    qr = ASSETS / "qr_repo.png"
    if qr.exists():
        qs = 94.0
        qx, qy = 112.0, H - 380.0
        c.setFillColor(white)
        c.rect(qx - 8, qy - 8, qs + 16, qs + 16, stroke=0, fill=1)
        c.drawImage(ImageReader(str(qr)), qx, qy, width=qs, height=qs,
                    mask="auto")
        c.setFillColor(white)
        c.setFont("Poppins-Bold", 22)
        c.drawString(qx + qs + 22, qy + qs - 26, "CODE  &  DATA")
        c.setFont("Lato", 20)
        c.setFillColor(HexColor("#C9CFD3"))
        c.drawString(qx + qs + 22, qy + qs - 54,
                     "github.com/AngadKumar16/FSI-SLM")
        c.drawString(qx + qs + 22, qy + qs - 80,
                     "REST API  ·  MCP server  ·  weights")

    # -- lime headline strip ------------------------------------------------ #
    y0 = H - HEADER_H - STRIP_H
    c.setFillColor(LIME)
    c.rect(0, y0, W, STRIP_H, stroke=0, fill=1)
    c.setStrokeColor(RULE)
    c.setLineWidth(2.4)
    c.line(0, y0, W, y0)

    parts = [("1,928", " solver cases"), ("107", " airfoil sections"),
             ("R² = 0.998", " lift, unseen shapes"),
             ("2.3%", " velocity-field error"),
             ("0.86", " lock-in F1 score"), ("1", " laptop, no GPU")]
    sep = "     •     "
    bold_s, reg_s = 46.0, 40.0
    total = 0.0
    for i, (b, r) in enumerate(parts):
        total += pdfmetrics.stringWidth(b, "Poppins-Bold", bold_s)
        total += pdfmetrics.stringWidth(r, "Lato", reg_s)
        if i < len(parts) - 1:
            total += pdfmetrics.stringWidth(sep, "Lato", reg_s)
    x = (W - total) / 2
    ty = y0 + STRIP_H / 2 - 16
    for i, (b, r) in enumerate(parts):
        c.setFillColor(HexColor("#1B2A0B"))
        c.setFont("Poppins-Bold", bold_s)
        c.drawString(x, ty, b)
        x += pdfmetrics.stringWidth(b, "Poppins-Bold", bold_s)
        c.setFont("Lato", reg_s)
        c.drawString(x, ty, r)
        x += pdfmetrics.stringWidth(r, "Lato", reg_s)
        if i < len(parts) - 1:
            c.setFillColor(HexColor("#5B6B20"))
            c.drawString(x, ty, sep)
            x += pdfmetrics.stringWidth(sep, "Lato", reg_s)

    # -- vertical column rules --------------------------------------------- #
    c.setStrokeColor(RULE)
    c.setLineWidth(GUTTER_RULE)
    for x in COL_X[1:3]:
        c.line(x, BOTTOM_MARGIN + 26, x, y0)


# --------------------------------------------------------------------------- #
def build(scales=(1.0, 1.0, 1.0), out=None, quiet=False):
    c = canvas.Canvas(str(out or OUT_PDF), pagesize=(W, H))
    c.setTitle(f"{TITLE_L1} {TITLE_L2}")
    c.setAuthor("Angad Kumar; Khemraj Shukla")
    if not DRY:
        draw_chrome(c)

    top = BODY_TOP - 30
    bottom = BOTTOM_MARGIN
    slacks = []

    # ===================== COLUMN 1 -- why it matters ==================== #
    c1 = Column(c, COL_X[0], COL_X[1], top, "column 1", scales[0])
    c1.header("MOTIVATION", gap_before=6)
    c1.para(
        "Fluid–structure interaction (FSI) decides when a hydrofoil or a "
        "turbine blade shakes itself apart. Reading what it produces — "
        "particle image velocimetry (PIV) fields, force and accelerometer "
        "traces — takes an expert, and **the expert is the bottleneck**. A "
        "frontier model talks about the physics fluently, invents what it "
        "does not know, and needs a data centre to do it.")
    c1.figure(FIGS_POSTER / "fig_regimes.png",
              caption="**F1.** The five flow–vibration regimes the "
                      "interpreter must tell apart — the expert judgement "
                      "being automated.")

    c1.header("OBJECTIVES")
    c1.bullets([
        "Reduce any FSI experiment, measured or simulated, to **one "
        "machine-readable JSON record**.",
        "Fine-tune a **small** language model to read the physics and flag "
        "likely sensor noise.",
        "**Ground** each answer in published fluid dynamics by retrieval, so "
        "the model cites rather than invents.",
        "Predict fields, forces **and** structural response from geometry, "
        "replacing the solver.",
        "Serve all of it from an ordinary laptop over REST and MCP.",
    ], numbered=True)

    c1.header("DATA")
    c1.para(
        "107 NACA sections, 18 angles of attack each at Re = 5×10⁵ — a "
        "1,926-case design grid, **1,928 solved cases** on disk (1,881 panel, "
        "35 RANS, 12 URANS). Three solvers write **one common tensor "
        "contract**, chosen by incidence: Hess–Smith panel with "
        "Beddoes–Leishman stall correction at any angle, steady RANS below "
        "stall, URANS above. A chordwise Euler–Bernoulli beam turns surface "
        "pressure into load and displacement, so every case carries structure "
        "as well as flow. **NACA 0009 is held out entirely.**")
    c1.figure(FIGS_PAPER / "fig2_dataset.png",
              caption="**F2.** (a) Splits hold out whole **shapes**, so "
                      "NACA 0009 (circled) is pinned to test and every number "
                      "reported for it is a generalisation result. (b) Where "
                      "the 18 angles per section sit. (c) Target fidelity, "
                      "log scale; the 12 URANS cases are not plotted.")

    c1.header("A REAL PIV EXPERIMENT")
    c1.para(
        "200 DaVis PIV frames of a NACA 0009 at **α = 17.2°**, 15 Hz on a "
        "152 × 127 grid, through the **unchanged** feature extractor: "
        "measured and simulated share one code path.")
    c1.figure(FIGS_POSTER / "fig_p_realpiv.png",
              caption="**F3.** Channels a PIV-only export cannot carry are "
                      "written as zero and **flagged**, never invented — and "
                      "those zeros provoke an over-confident lock-in call, "
                      "which the deterministic validator **overrules**. A "
                      "broad wake and a flat POD spectrum are separated flow, "
                      "not lock-in.")
    slacks.append(c1.report(bottom, quiet))

    # ===================== COLUMN 2 -- how it works ====================== #
    c2 = Column(c, COL_X[1], COL_X[2], top, "column 2", scales[1])
    c2.header("METHODS", gap_before=6)
    c2.para(
        "A 13-channel raster carries the geometry, including the **inviscid "
        "panel solution on the same grid**, so the 11.59 M U-Net predicts the "
        "**correction** to potential flow, not the flow. α, Re and fidelity "
        "enter by FiLM; attention sits at the 8×8 bottleneck because "
        "circulation is global. The reader is **Phi-4-mini** (3.8 B) plus a "
        "46 MB LoRA adapter.")
    c2.figure(FIGS_PAPER / "fig1_system.png",
              caption="**F4.** The twin end to end. The bridge turns a "
                      "steady field into a time-resolved record, and eleven "
                      "fault detectors label dropout, clipping and mains hum "
                      "before the model sees anything. **The dashed red edge "
                      "is the one unimplemented link;** F3 is the measured "
                      "path running beside it.")
    c2.figure(FIGS_POSTER / "fig_p_serve.png",
              caption="**F5.** Retrieval-augmented generation (RAG) over "
                      "**1,840 passages** puts the top three into the "
                      "prompt, so an answer carries what it leaned on. A CLI, "
                      "a **FastAPI REST** endpoint and a **Model Context "
                      "Protocol (MCP)** server of twelve tools all reach the "
                      "same pipeline — which is what lets an agent drive the "
                      "twin.")

    c2.header("ANALOGY: WHY MULTI-FIDELITY")
    c2.para(
        "A panel solve costs 0.13 s at any angle; a steady RANS solve costs "
        "four minutes and only converges below stall; a URANS case costs "
        "hours. Train on the cheap tier alone and you get a fluent "
        "approximation with a **structural blind spot**, because the beam "
        "amplifies the leading-edge pressure difference — exactly where the "
        "solvers disagree. The cheap tier buys coverage; the expensive tier "
        "buys truth. The same argument sets the model size: a frontier model "
        "is a library with no lab notebook.")

    c2.header("REFERENCES", gap_before=16, gap_after=10, size=44)
    refs = [
        "Ronneberger, O. et al. (2015) U-Net. MICCAI.   ·   Thuerey, N. et "
        "al. (2020) Deep learning methods for RANS simulations of airfoil "
        "flows. AIAA J. 58(1).",
        "Lewis, P. et al. (2020) Retrieval-augmented generation for "
        "knowledge-intensive NLP tasks. NeurIPS.   ·   Leishman & Beddoes "
        "(1989) A semi-empirical model for dynamic stall. JAHS 34(3).",
        "Hess & Smith (1967) Calculation of potential flow about arbitrary "
        "bodies. Prog. Aerospace Sci. 8.   ·   Abbott & von Doenhoff (1959) "
        "Theory of Wing Sections. Dover.",
        "Anthropic (2024) Model Context Protocol specification.",
    ]
    for r in refs:
        c2.para(r, size=21, lead=25, gap_after=2)

    c2.header("ACKNOWLEDGEMENT", gap_before=14, gap_after=10, size=44)
    c2.para(
        "With thanks to the AGU Bright STaRS program. Built with Python, "
        "PyTorch, OpenFOAM, MLX and FAISS.   Code, data and weights: "
        "**github.com/AngadKumar16/FSI-SLM**", size=23, lead=28)
    slacks.append(c2.report(bottom, quiet))

    # ===================== COLUMN 3 -- results =========================== #
    c3 = Column(c, COL_X[2], COL_X[3], top, "column 3", scales[2],
                text_w=1210.0)
    c3.header("RESULTS", gap_before=6)
    c3.figure(FIGS_PAPER / "fig5_fields.png",
              caption="**F6.** Solver, surrogate and error on NACA 0009 at "
                      "α = 5°, a section held out entirely. Relative RMSE "
                      "0.007 (u), 0.029 (v), 0.049 (p), 0.024 (load), 0.093 "
                      "(displacement); error concentrates in the near-wall "
                      "layer and the wake. Across **293 held-out cases on 17 "
                      "unseen sections**, scored against "
                      "the panel targets it was trained on: R² = 0.998 lift, "
                      "0.995 drag, 0.991 moment, 0.968 peak deflection, "
                      "0.786 trailing-edge twist.")
    c3.figure(FIGS_POSTER / "fig_mf_gain.png",
              caption="**F7.** Panel-only training reproduces RANS velocity "
                      "and drag but not the structural channels. Adding 47 "
                      "high-fidelity cases on **training-split** sections "
                      "closes most of that gap while every **field** channel "
                      "stays within 3.1% of the panel baseline. The honest "
                      "cost: panel-test drag MAE regresses 0.00230 → 0.00314; "
                      "lift MAE improves 11%.")
    c3.figure(FIGS_PAPER / "fig9_interpreter.png",
              caption="**F8.** The interpreter on 50 held-out records. Regime "
                      "classification is saturated at this sample size — a "
                      "fact about the test set as much as the model — so the "
                      "discriminating metric is lock-in detection: recall "
                      "1.000, precision 0.750. It **never misses a lock-in** "
                      "and over-calls eight — a bias the rule-based labeller "
                      "shares, so it was inherited, not invented.")

    c3.space(10)
    c3.two_up(
        ("SUMMARY",
         "The sensor half predicts velocity, pressure, load and displacement "
         "fields plus lift, drag and moment on shapes it has never seen. The "
         "interpreter half reads the resulting JSON record, names the regime, "
         "flags sensor faults and cites the literature it used — and, on the "
         "real PIV run, is overruled when it over-reaches. Both halves run on "
         "one laptop, over REST and MCP."),
        ("FUTURE WORK",
         "Acquire the structural channel alongside PIV so the real records "
         "carry displacement, not zeros. Finish the URANS sweep so the "
         "deep-stall branch becomes trainable. Close the structural loop so "
         "deflection feeds back into the flow. Add calibrated uncertainty on "
         "every field, and a harder interpreter test set now that regime "
         "accuracy has saturated."),
        size=25, lead=30)

    slacks.append(c3.report(bottom, quiet))

    c.showPage()
    c.save()
    if not quiet:
        print(f"\nwrote {out or OUT_PDF}  ({W/72:.0f} x {H/72:.0f} in)")
        if min(slacks) < 0:
            print("!! a column overflows -- fix before printing")
    return slacks


def autofit(target=26.0):
    global DRY
    """Shrink figures per column until every column fits with a little air.

    Column height is linear in figure width, so a couple of secant steps
    converge. Solved numerically rather than guessed by eye.
    """
    DRY = True
    scales = [1.0, 1.0, 1.0]
    for it in range(120):
        slacks = build(tuple(scales), out="/tmp/_fit.pdf", quiet=True)
        if all(target - 8 <= s for s in slacks) and max(slacks) < 260:
            break
        for i, s in enumerate(slacks):
            if s < target:
                scales[i] *= 0.975
            elif s > 240 and scales[i] < 0.999:
                scales[i] *= 1.010
        scales = [min(1.0, max(0.45, x)) for x in scales]
    DRY = False
    print("fig scale per column:",
          " ".join(f"{x:.3f}" for x in scales))
    return tuple(scales)


if __name__ == "__main__":
    sc = autofit()
    build(sc)
