"""Build the AGU poster (60 x 40 in) with reportlab -- v2, plain-language edit.

    python3 poster/build_poster_v2.py

Same Canva template geometry as build_poster.py: black header band with the
world-map texture, lime hook strip, three unequal columns split by hairline
rules, coral section headers.  What changed is the writing: short bullets,
no hedging, one idea a line, and the physical specimen photograph at the top
of MOTIVATION so a reader knows what the poster is about before reading a word.

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
OUT_PDF = ROOT / "poster/output/FSI-SLM_AGU_poster_60x40_v2.pdf"

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

BODY_F, BODY_S, BODY_LEAD = "Lato", 29.0, 35.0
CAP_F, CAP_S, CAP_LEAD = "Lato-It", 22.0, 27.0
HEAD_F, HEAD_S = "Poppins-Bold", 52.0

TITLE_L1 = "A Fine-Tuned Small Language Model for Interpreting"
TITLE_L2 = "Fluid–Structure Interaction Experiments"
SUBTITLE = ("PIV experiment  ·  U-Net flow surrogate  ·  fine-tuned small "
            "language model  ·  retrieval grounding  ·  REST and MCP")
HOOK = ("One laptop predicts the flow around a wing and explains what the "
        "measurement means — work that used to need a cluster and a specialist.")


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
    """Split into words, where a word is a list of (text, bold) runs."""
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
            self.para(item, size=size, lead=lead, gap_after=12,
                      indent=marker_w)
        self.y -= gap_after - 12

    def stats(self, items, cols=3, box_h=126, gap=14, gap_after=26,
              num_s=48, lab_s=20):
        """Grid of lime number tiles: the six things to remember."""
        rows = (len(items) + cols - 1) // cols
        bw = (self.w - gap * (cols - 1)) / cols
        for r in range(rows):
            ytop = self.y
            for cidx in range(cols):
                k = r * cols + cidx
                if k >= len(items):
                    break
                num, lab = items[k]
                bx = self.x + cidx * (bw + gap)
                by = ytop - box_h
                if not DRY:
                    self.c.setFillColor(LIME)
                    self.c.setStrokeColor(RULE)
                    self.c.setLineWidth(1.8)
                    self.c.rect(bx, by, bw, box_h, stroke=1, fill=1)
                    self.c.setFillColor(HexColor("#1B2A0B"))
                    self.c.setFont("Poppins-Bold", num_s)
                    self.c.drawCentredString(bx + bw / 2, by + box_h - num_s - 12,
                                             num)
                    self.c.setFillColor(HexColor("#33401A"))
                    for j, line in enumerate(lab.split("\n")):
                        self.c.setFont("Lato", lab_s)
                        self.c.drawCentredString(bx + bw / 2,
                                                 by + 34 - j * (lab_s + 4), line)
            self.y = ytop - box_h - gap
        self.y -= gap_after - gap

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

    def two_up(self, left, right, size=BODY_S, lead=BODY_LEAD, head_size=46,
               gutter=56, gap_after=10, bullets=False):
        """Two half-width sections side by side, as in the template."""
        x0, w0, y0 = self.x, self.w, self.y
        half = (w0 - gutter) / 2
        ends = []
        tw = self.text_w
        for i, (title, body) in enumerate((left, right)):
            self.x = x0 + i * (half + gutter)
            self.w = half
            self.text_w = half
            self.y = y0
            self.header(title, gap_before=0, gap_after=12, size=head_size)
            if bullets:
                self.bullets(body, size=size, lead=lead, gap_after=0)
            else:
                self.para(body, size=size, lead=lead, gap_after=0)
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
    """Header band, logos, title, lime hook strip, column rules."""
    hdr = ASSETS / "header_bg.png"
    if hdr.exists():
        c.drawImage(ImageReader(str(hdr)), 0, H - HEADER_H, width=W,
                    height=HEADER_H, mask="auto")
    else:
        c.setFillColor(BLACK)
        c.rect(0, H - HEADER_H, W, HEADER_H, stroke=0, fill=1)

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

    # -- lime hook strip: one plain sentence, no jargon --------------------- #
    y0 = H - HEADER_H - STRIP_H
    c.setFillColor(LIME)
    c.rect(0, y0, W, STRIP_H, stroke=0, fill=1)
    c.setStrokeColor(RULE)
    c.setLineWidth(2.4)
    c.line(0, y0, W, y0)
    hs = 44.0
    while pdfmetrics.stringWidth(HOOK, "Poppins-Med", hs) > W - 220:
        hs -= 0.5
    c.setFillColor(HexColor("#1B2A0B"))
    c.setFont("Poppins-Med", hs)
    c.drawCentredString(W / 2, y0 + STRIP_H / 2 - hs * 0.36, HOOK)

    # -- vertical column rules --------------------------------------------- #
    c.setStrokeColor(RULE)
    c.setLineWidth(GUTTER_RULE)
    for x in COL_X[1:3]:
        c.line(x, BOTTOM_MARGIN + 26, x, y0)



# --------------------------------------------------------------------------- #
REFS = [
    "Ronneberger, O. et al. (2015) U-Net. MICCAI.   ·   Thuerey, N. et "
    "al. (2020) Deep learning methods for RANS simulations of airfoil "
    "flows. AIAA J. 58(1).",
    "Lewis, P. et al. (2020) Retrieval-augmented generation for "
    "knowledge-intensive NLP tasks. NeurIPS.   ·   Leishman & Beddoes "
    "(1989) A semi-empirical model for dynamic stall. JAHS 34(3).",
    "Hess & Smith (1967) Calculation of potential flow about arbitrary "
    "bodies. Prog. Aerospace Sci. 8.   ·   Abbott & von Doenhoff (1959) "
    "Theory of Wing Sections. Dover.   ·   Anthropic (2024) Model "
    "Context Protocol specification.",
]


def draw_refs(col, air=0.0):
    """References + acknowledgement, drawn wherever the column currently is."""
    col.header("REFERENCES", gap_before=14 + air, gap_after=10, size=44)
    for r in REFS:
        col.para(r, size=21, lead=25, gap_after=2)
    col.header("ACKNOWLEDGEMENT", gap_before=14 + air, gap_after=10, size=44)
    col.para(
        "With thanks to the AGU Bright STaRS program. Built with Python, "
        "PyTorch, OpenFOAM, MLX and FAISS.   Code, data and weights: "
        "**github.com/AngadKumar16/FSI-SLM**", size=23, lead=28)


def refs_height(x0, x1):
    """Height of the refs block, measured on a throwaway canvas."""
    import io
    scratch = canvas.Canvas(io.BytesIO(), pagesize=(W, H))
    probe = Column(scratch, x0, x1, 0.0, "probe")
    draw_refs(probe)
    return -probe.y


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

    # ===================== COLUMN 1 -- the problem ======================= #
    # Column 1 is the shortest, so its leftover space is spread evenly between
    # sections as extra air rather than dumped at the foot as one dead band.
    def col1_body(col, air):
        col.header("MOTIVATION", gap_before=6 + air)
        col.figure(ASSETS / "specimen_naca0009.png", width=col.w * 0.94,
                   gap_before=0, gap_after=16,
                   caption="**The specimen.** A NACA 0009 section, 9% thick, "
                           "pinned through the hole. Every measured number on "
                           "this poster comes from this part.")
        col.bullets([
            "Moving fluid shakes structures. Enough shaking breaks them.",
            "A **PIV** camera answers this with thousands of velocity vectors "
            "per frame.",
            "Turning those vectors into a verdict takes an expert. **Experts "
            "are the bottleneck.**",
            "A frontier chatbot talks about the physics, invents the rest, "
            "and needs a data centre.",
        ])
        col.figure(FIGS_POSTER / "fig_regimes.png",
                   caption="**F1.** The call the expert makes. Only the last "
                           "three are dangerous. This is the judgement being "
                           "automated.")

        col.header("WHAT WE BUILT", gap_before=26 + air)
        col.bullets([
            "One **JSON record** for any run — measured or simulated.",
            "A **small** language model, fine-tuned to read that record.",
            "**Retrieval** from published papers, so it cites instead of "
            "inventing.",
            "A network that predicts flow **and** structure from shape alone, "
            "replacing the solver.",
            "All of it on **one laptop**, reachable over REST and MCP.",
        ], numbered=True)

        col.header("DATA", gap_before=26 + air)
        col.bullets([
            "107 airfoil sections × 18 angles at Re = 5×10⁵. **1,928 cases "
            "solved.**",
            "Three solvers, one output format: panel 0.13 s, RANS 4 min, "
            "URANS hours.",
            "A beam model turns surface pressure into deflection, so every "
            "case carries structure.",
            "**NACA 0009 never appears in training** — every number quoted "
            "for it is generalisation.",
        ])

    import io
    probe = Column(canvas.Canvas(io.BytesIO(), pagesize=(W, H)),
                   COL_X[0], COL_X[1], top, "probe", scales[0])
    col1_body(probe, 0.0)
    draw_refs(probe)
    slack1 = probe.y - bottom
    air = max(0.0, slack1) / 5.0

    c1 = Column(c, COL_X[0], COL_X[1], top, "column 1", scales[0])
    col1_body(c1, air)
    draw_refs(c1, air)
    c1.report(bottom, quiet)
    # autofit must see the pre-air slack, or column 1 always looks exactly full
    slacks.append(slack1)

    # ===================== COLUMN 2 -- how it works ====================== #
    c2 = Column(c, COL_X[1], COL_X[2], top, "column 2", scales[1])

    c2.header("METHODS", gap_before=6)
    c2.bullets([
        "The network reads **13 channels**: the shape, the flow conditions, "
        "and the ideal-flow solution on the same grid.",
        "So it learns the **correction** to ideal flow, not the flow itself. "
        "11.59 M parameters.",
        "The reader is **Phi-4-mini** (3.8 B) with a 46 MB LoRA adapter.",
        "Answers are grounded by retrieval: 16 papers, 1,840 passages, top "
        "three above a 0.30 score enter the prompt.",
        "**Eleven fault detectors** label dropout, clipping and mains hum "
        "before the model sees anything.",
        "Served three ways from one laptop — CLI, **FastAPI REST**, and an "
        "**MCP** server of twelve tools an agent can drive.",
    ])
    c2.figure(FIGS_PAPER / "fig1_system.png",
              caption="**F2.** The twin, end to end. **The dashed red edge is "
                      "the one link not yet built** — F3 is the measured path "
                      "running beside it.")

    c2.header("A REAL MEASUREMENT")
    c2.para(
        "200 PIV frames of the specimen at **α = 17.2°**, pushed through the "
        "**same unchanged code** as the simulations.")
    c2.figure(FIGS_POSTER / "fig_p_realpiv.png", gap_before=2,
              caption="**F3.** PIV cannot supply the force channels, so they "
                      "are written as zero and flagged — never invented. "
                      "Those zeros push the language model into a wrong "
                      "lock-in call, and the rule-based validator "
                      "**overrules it**.")

    c2.header("WHY TWO KINDS OF SOLVER")
    c2.bullets([
        "Cheap solves are fast and cover every angle. Expensive solves are "
        "right and rare.",
        "Train on cheap alone: the **flow** comes out right, the "
        "**structure** does not.",
        "The beam amplifies the leading-edge pressure difference — exactly "
        "where the two solvers disagree.",
        "**47 expensive cases close most of that gap.** Cheap buys coverage; "
        "expensive buys truth.",
    ])
    slacks.append(c2.report(bottom, quiet))

    # ===================== COLUMN 3 -- results =========================== #
    c3 = Column(c, COL_X[2], COL_X[3], top, "column 3", scales[2],
                text_w=1210.0)

    c3.header("RESULTS", gap_before=6, gap_after=18)
    c3.stats([
        ("R² = 0.998", "lift, on shapes it\nhas never seen"),
        ("2.3%", "velocity-field\nerror"),
        ("293", "held-out cases,\n17 unseen sections"),
        ("0.86", "lock-in F1\n(recall 1.000)"),
        ("47", "expensive cases\nclose the gap"),
        ("1", "laptop, no GPU\nanywhere"),
    ], box_h=118)
    c3.figure(FIGS_PAPER / "fig5_fields.png",
              caption="**F4.** Solver, network and error on NACA 0009 at "
                      "α = 5°, a section held out entirely. **Error "
                      "concentrates in the near-wall layer and the wake**, "
                      "exactly where it should. Relative RMSE 0.007 (u), "
                      "0.029 (v), 0.049 (p), 0.024 (load), 0.093 (deflection), "
                      "against the panel targets the network trained on.")
    c3.figure(FIGS_POSTER / "fig_mf_gain.png",
              caption="**F5.** The 47 expensive cases close the structural "
                      "gap while every **field** channel stays within 3.1% of "
                      "the cheap-only baseline. Honest cost: drag error "
                      "regresses 0.00230 → 0.00314; lift error improves 11%.")
    c3.figure(FIGS_PAPER / "fig9_interpreter.png",
              caption="**F6.** The reader on 50 held-out records. Regime "
                      "classification is **saturated at this sample size**, "
                      "so the metric that discriminates is lock-in detection: "
                      "it never misses one and over-calls eight — a bias "
                      "inherited from the rule-based labeller.")

    c3.space(6)
    c3.two_up(
        ("SUMMARY", [
            "Predicts velocity, pressure, load and deflection on shapes it "
            "has never seen.",
            "Reads the record, names the regime, flags faults, cites its "
            "sources.",
            "On the real run the validator **caught the model over-reaching**.",
            "Both halves run on one laptop, over REST and MCP.",
        ]),
        ("FUTURE WORK", [
            "Measure deflection alongside PIV, so real records stop "
            "carrying zeros.",
            "Finish the URANS sweep to make the deep-stall branch trainable.",
            "Close the loop: let deflection feed back into the flow.",
            "Calibrated uncertainty, and a harder interpreter test set.",
        ]),
        size=25, lead=30, bullets=True)

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
    """Shrink figures per column until every column fits with a little air."""
    DRY = True
    scales = [1.0, 1.0, 1.0]
    for it in range(160):
        slacks = build(tuple(scales), out="/tmp/_fit.pdf", quiet=True)
        if all(target - 8 <= s for s in slacks) and max(slacks) < 260:
            break
        for i, s in enumerate(slacks):
            if s < target:
                scales[i] *= 0.975
            elif s > 240 and scales[i] < 0.999:
                scales[i] *= 1.010
        scales = [min(1.0, max(0.40, x)) for x in scales]
    DRY = False
    print("fig scale per column:", " ".join(f"{x:.3f}" for x in scales))
    return tuple(scales)


if __name__ == "__main__":
    sc = autofit()
    build(sc)
