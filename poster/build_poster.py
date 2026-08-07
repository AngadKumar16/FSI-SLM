"""Build the AGU poster (60 x 40 in) with reportlab.

    python3 poster/build_poster.py

Geometry is measured off the supplied Canva template so the result drops into
the same board: black header band with the world-map texture, lime title strip,
three unequal columns split by hairline rules, coral section headers.

The flow engine reports how full each column is, so overflow is caught here
rather than at the print shop.
"""
from __future__ import annotations

import textwrap
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

# The template's black band is 0.1833*H and the lime strip 0.0396*H. That
# leaves too little body height for this much content, so the band is tightened
# while keeping the same three-part structure and proportions.
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

    def __init__(self, c, x0, x1, top, name, fig_scale=1.0):
        self.c = c
        self.x = x0 + PAD
        self.w = (x1 - x0) - 2 * PAD
        self.y = top
        self.top = top
        self.name = name
        self.fig_scale = fig_scale

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
        lines = wrap_rich(text, self.w - indent, font, size)
        space = pdfmetrics.stringWidth(" ", font, size)
        self.c.setFillColor(color)
        for line in lines:
            xx = self.x + indent
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
            self.c.drawString(self.x, self.y - lead, mark)
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
        """Two figures side by side, captions beneath each.

        Short wide figures waste a lot of height when stacked in a wide
        column; paired, they use the full measure instead of sitting in a
        pool of white space.
        """
        x0, w0, y0 = self.x, self.w, self.y
        half = (w0 - gutter) / 2
        ends = []
        for i, (path, cap) in enumerate((left, right)):
            self.x = x0 + i * (half + gutter)
            self.w = half
            self.y = y0
            self.figure(path, caption=cap, gap_before=gap_before,
                        gap_after=0, cap_size=cap_size)
            ends.append(self.y)
        self.x, self.w = x0, w0
        self.y = min(ends) - gap_after

    def two_up(self, left, right, size=BODY_S, lead=BODY_LEAD, head_size=46,
               gutter=56, gap_after=10):
        """Two half-width sections side by side, as in the template."""
        x0, w0, y0 = self.x, self.w, self.y
        half = (w0 - gutter) / 2
        ends = []
        for i, (title, text) in enumerate((left, right)):
            self.x = x0 + i * (half + gutter)
            self.w = half
            self.y = y0
            self.header(title, gap_before=0, gap_after=12, size=head_size)
            self.para(text, size=size, lead=lead, gap_after=0)
            ends.append(self.y)
        self.x, self.w = x0, w0
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

    # -- title, centred in the space the logos and authors leave ------------ #
    tl, tr = 460.0, W - 1330.0
    tc = (tl + tr) / 2
    c.setFillColor(white)
    c.setFont("Poppins", 82)
    c.drawCentredString(tc, H - 150, "FSI-SLM: An End-to-End Digital Twin")
    c.drawCentredString(tc, H - 252, "that Reads Fluid–Structure Interaction Experiments")
    c.setFont("Lato-It", 40)
    c.setFillColor(HexColor("#DFE3E5"))
    c.drawCentredString(
        tc, H - 336,
        "A U-Net flow surrogate and a small language model that interprets the "
        "result — the whole pipeline runs on one laptop")

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
             ("F1 = 0.86", " lock-in detection"), ("1", " laptop, no GPU")]
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
        c.line(x, 0, x, y0)


# --------------------------------------------------------------------------- #
def build(scales=(1.0, 1.0, 1.0), out=None, quiet=False):
    c = canvas.Canvas(str(out or OUT_PDF), pagesize=(W, H))
    c.setTitle("FSI-SLM: An End-to-End Digital Twin that Reads "
               "Fluid-Structure Interaction Experiments")
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
        "Fluid–structure interaction decides when a hydrofoil, a riser or a "
        "turbine blade shakes itself apart. Reading what it produces — PIV "
        "fields, force traces, accelerometers — takes an expert, and the "
        "expert is the bottleneck. A frontier language model discusses the "
        "physics fluently, **invents** what it does not know, and needs a "
        "data centre to do it.")
    c1.figure(FIGS_POSTER / "fig_regimes.png",
              caption="**F1.** The five flow–vibration regimes the "
                      "interpreter must tell apart. Separating these by eye "
                      "is the expert judgement we are automating.")

    c1.header("OBJECTIVES")
    c1.bullets([
        "Reduce a raw FSI experiment to **one machine-readable record**.",
        "Fine-tune a **small** language model to interpret the physics and "
        "flag likely sensor noise.",
        "**Ground** every answer in published fluid dynamics by retrieval, "
        "so the model cites rather than invents.",
        "Predict fields, forces **and** structural response straight from "
        "geometry, replacing the solver.",
        "Serve all of it from a laptop over REST and MCP.",
    ], numbered=True)

    c1.header("DATA")
    c1.para(
        "107 NACA sections at Re = 5×10⁵, 18 angles of attack each. Three "
        "solvers write **one common tensor contract**, chosen per case by "
        "incidence: a Hess–Smith panel method with Beddoes–Leishman stall "
        "correction at any angle, steady RANS below stall, URANS above it "
        "where no steady solution exists. A chordwise Euler–Bernoulli beam "
        "turns surface pressure into load and displacement, so every case "
        "carries structure as well as flow.")
    c1.figure(FIGS_PAPER / "fig2_dataset.png",
              caption="**F2.** (a) The 107 sections by maximum thickness and "
                      "its chordwise location. Splits hold out whole "
                      "**shapes**, so NACA 0009 (circled) is pinned to test "
                      "and every number reported for it is a generalisation "
                      "result. (b) Angle sampling tightens from 2° to 1° "
                      "through stall, where the physics changes fastest. "
                      "(c) Target fidelity, log scale.")
    c1.figure(FIGS_PAPER / "fig6_rans.png",
              caption="**F3.** Are the targets right? Steady-RANS validation "
                      "on the held-out section: 15 OpenFOAM cases, k-ω SST, "
                      "324 k cells. The converged branch gives a lift slope "
                      "of 0.095/deg, 86% of thin-airfoil 2π, and a drag "
                      "minimum of 0.0078 at α = 0, inside the published "
                      "0.006–0.008 band. Every case at α ≥ 10° hit the "
                      "iteration cap: separated flow has no steady solution "
                      "to converge **to**. **RANS validation is claimed for "
                      "α ≤ 8° only**; the failed cases stay in the manifest "
                      "with their residuals.")
    slacks.append(c1.report(bottom, quiet))

    # ===================== COLUMN 2 -- how it works ====================== #
    c2 = Column(c, COL_X[1], COL_X[2], top, "column 2", scales[1])
    c2.header("METHODS", gap_before=6)
    c2.figure(FIGS_PAPER / "fig1_system.png",
              caption="**F4.** The twin end to end. The U-Net learns geometry "
                      "→ fields and coefficients; the twin bridge turns a "
                      "steady field into a time-resolved record, and eleven "
                      "fault detectors label PIV dropout, ADC clipping and "
                      "mains hum before the model sees anything. A LoRA-tuned "
                      "Phi-4-mini interprets it, grounded by retrieval over "
                      "16 papers. **The dashed red edge is the one "
                      "unimplemented link:** a PIV rig would replace the "
                      "synthetic experiment, and nothing downstream changes.")
    c2.figure(FIGS_PAPER / "fig3_network.png",
              caption="**F5.** (a) Six of the thirteen input channels. "
                      "pot_u, pot_v, pot_p are the inviscid panel solution on "
                      "the same raster, so the network predicts the "
                      "**correction** to potential flow rather than the flow "
                      "— stagnation, circulation and the far field arrive "
                      "already exact. (b) 11.59 M parameters, self-attention "
                      "at the 8×8 bottleneck because circulation is a global "
                      "constraint; α, Re and fidelity enter by FiLM.")
    c2.header("ANALOGY: WHY SMALL + GROUNDED")
    c2.para(
        "A frontier model is a vast library with no lab notebook: fluent, "
        "unfalsifiable, data-centre-sized. Ours is a junior lab partner "
        "trained on **this** rig with the textbook open beside it — 46 MB "
        "of LoRA weights over a 3.8 B model and a FAISS index it must cite "
        "from. The U-Net is a learned wind tunnel: four minutes of RANS "
        "collapses to milliseconds, so a new case is a lookup, not an "
        "overnight solve.")
    c2.header("REFERENCES", gap_before=18, gap_after=10, size=44)
    refs = [
        "Ronneberger, O., Fischer, P., Brox, T. (2015) U-Net: convolutional "
        "networks for biomedical image segmentation. MICCAI.",
        "Thuerey, N. et al. (2020) Deep learning methods for Reynolds-averaged "
        "Navier–Stokes simulations of airfoil flows. AIAA J. 58(1).",
        "Lewis, P. et al. (2020) Retrieval-augmented generation for "
        "knowledge-intensive NLP tasks. NeurIPS.",
        "Leishman, J. G., Beddoes, T. S. (1989) A semi-empirical model for "
        "dynamic stall. J. American Helicopter Society 34(3).",
    ]
    for r in refs:
        c2.para(r, size=20, lead=24, gap_after=2)

    c2.header("ACKNOWLEDGEMENT", gap_before=16, gap_after=10, size=44)
    c2.para(
        "Thanks to Khemraj Shukla (Brown University) and the AGU Bright STaRS "
        "program. Built with Python, PyTorch, OpenFOAM, MLX and FAISS.   Code "
        "and data: **github.com/AngadKumar16/FSI-SLM**", size=23, lead=28)
    slacks.append(c2.report(bottom, quiet))

    # ===================== COLUMN 3 -- results =========================== #
    c3 = Column(c, COL_X[2], COL_X[3], top, "column 3", scales[2])
    c3.header("RESULTS", gap_before=6)
    c3.figure(FIGS_PAPER / "fig5_fields.png",
              caption="**F6.** Solver, surrogate and error on NACA 0009 at "
                      "α = 5° — a section held out of training entirely. "
                      "Relative RMSE 0.007 (u), 0.029 (v), 0.049 (p), 0.024 "
                      "(load), 0.093 (displacement); error concentrates in "
                      "the near-wall layer and the wake. Across 293 held-out "
                      "cases on 17 unseen sections: **R² = 0.998 for lift**, "
                      "0.995 for drag.")
    c3.figure(FIGS_POSTER / "fig_mf_gain.png",
              caption="**F7.** Panel-only training reproduces RANS velocity "
                      "and drag but not the structural channels — the "
                      "chordwise beam amplifies the leading-edge pressure "
                      "difference, exactly where the solvers disagree. Adding "
                      "47 high-fidelity cases on **training-split** sections "
                      "closes most of the gap at no cost to panel accuracy "
                      "— every field channel stays within 3.1% of the "
                      "panel-trained baseline.")
    c3.figure(FIGS_PAPER / "fig9_interpreter.png",
              caption="**F8.** The interpreter on 50 held-out records. The "
                      "discriminating metric is lock-in detection: recall "
                      "1.000, precision 0.750 — it **never misses a lock-in** "
                      "and over-calls eight. The rule-based labeller carries "
                      "the same bias, so this was inherited, not invented.")

    c3.space(10)
    c3.two_up(
        ("SUMMARY",
         "The sensor half predicts velocity, pressure, load and displacement "
         "fields plus lift, drag and moment on shapes it has never seen. The "
         "interpreter half reads the record, names the regime, flags sensor "
         "faults and cites what it used. Both run on one laptop."),
        ("FUTURE WORK",
         "Finish the URANS sweep so the deep-stall branch becomes trainable. "
         "Connect the PIV rig — the one dashed edge in F4. Close the "
         "structural loop so deflection feeds back into the flow. Add "
         "calibrated uncertainty on every field."),
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
    for it in range(60):
        slacks = build(tuple(scales), out="/tmp/_fit.pdf", quiet=True)
        if all(target - 8 <= s for s in slacks) and max(slacks) < 260:
            break
        for i, s in enumerate(slacks):
            if s < target:
                scales[i] *= 0.965
            elif s > 240 and scales[i] < 0.999:
                scales[i] *= 1.012
        scales = [min(1.0, max(0.55, x)) for x in scales]
    DRY = False
    print("fig scale per column:",
          " ".join(f"{x:.3f}" for x in scales))
    return tuple(scales)


if __name__ == "__main__":
    sc = autofit()
    build(sc)
