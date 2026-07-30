"""Shared poster style + palette."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

NAVY   = "#0B2545"
BLUE   = "#134074"
TEAL   = "#1B9AAA"
AQUA   = "#3FC1C9"
CORAL  = "#EF6F45"
GOLD   = "#F2B134"
SAND   = "#F6F4EF"
INK    = "#12232E"
GREY   = "#5B6B73"
GREEN  = "#3DA35D"
PURPLE = "#6C5CE7"
RED    = "#D7263D"

REGIME_COLORS = {
    "attached":     "#3DA35D",
    "separated":    "#1B9AAA",
    "viv_lockin":   "#EF6F45",
    "galloping":    "#6C5CE7",
    "stall_flutter":"#D7263D",
}
REGIME_LABELS = {
    "attached": "Attached",
    "separated": "Separated",
    "viv_lockin": "VIV lock-in",
    "galloping": "Galloping",
    "stall_flutter": "Stall flutter",
}

def apply():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 15,
        "axes.titlesize": 19,
        "axes.titleweight": "bold",
        "axes.labelsize": 16,
        "axes.labelcolor": INK,
        "axes.edgecolor": GREY,
        "axes.linewidth": 1.1,
        "xtick.color": INK, "ytick.color": INK,
        "text.color": INK,
        "figure.dpi": 140,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
    })
