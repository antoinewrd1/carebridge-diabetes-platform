"""Shared plotting style for all CareBridge figures."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from carebridge.config import REPORTS

NAVY = "#1F3864"
MID = "#8FA5C7"
LIGHT = "#DCE3EF"
ACCENT = "#A6462E"
SLATE = "#44546A"

OUT = REPORTS / "figures"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.size": 9,
    "axes.edgecolor": SLATE,
    "axes.labelcolor": SLATE,
    "xtick.color": SLATE,
    "ytick.color": SLATE,
    "axes.titlesize": 10,
    "axes.titleweight": "bold",
    "axes.titlecolor": NAVY,
    "figure.facecolor": "white",
})


def clean(ax, grid_axis="y"):
    """Gridlines behind the data, no box around the plot."""
    ax.grid(axis=grid_axis, color="#DDDDDD", linewidth=0.7)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def save(fig, name: str) -> None:
    """Write a figure and report it, so a skipped figure is visible."""
    path = OUT / name
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  [fig] {name}")