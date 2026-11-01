#!/usr/bin/env python3
"""Make the three result figures for TR-EN-CodeSwitch-LID.

Reads:  results/results_table.csv (figures 1 and 2), results/predictions/*_test.csv
        and data/splits/test.txt (figure 3).
Makes:  results/figures/fig1_macro_f1.png, fig2_data_size.png, fig3_confusion.png
Run:    python src/plot_figures.py   (from the top folder of the repo, no arguments)
Only reads inputs and writes the three PNGs. Missing runs/files are skipped with a message.
"""

import re
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no window needed, we only save files
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# ----------------------------------------------------------------------------
# Settings (change here)
# ----------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
TABLE = REPO_ROOT / "results" / "results_table.csv"
PRED_DIR = REPO_ROOT / "results" / "predictions"
TEST_SPLIT = REPO_ROOT / "data" / "splits" / "test.txt"
OUT_DIR = REPO_ROOT / "results" / "figures"

BASELINE_RUN = "berturk_gen0"
BEST_RUN = "berturk_gen5000"  # chosen on validation (see results/ibrahim_size_choice.md)

LABELS = ["TR", "EN", "MIXED", "NE", "AMBIGUOUS", "OTHER"]
SIZES = [0, 2500, 5000, 10000]
PAPER_BERTURK = 0.4734
DPI = 200
EXPECTED_TEST_ROWS = 1373

GREY, BLUE, ORANGE = "#9E9E9E", "#4C72B0", "#DD8452"
FAMILY_COLOURS = {"berturk": BLUE, "xlmr": "#55A868", "crf": "#8172B3"}
FAMILY_NAMES = {"berturk": "BERTurk", "xlmr": "XLM-R", "crf": "CRF"}
FIXED_NAMES = {
    "majority": "Majority class",
    "rules_wordfreq": "Rules (wordfreq)",
    "rules_zemberek": "Rules (Zemberek)",
    "gpt4o_0shot": "GPT-4o zero-shot",
    "gpt4o_3shot": "GPT-4o 3-shot (paper examples)",
    "gpt4o_3shot-ours": "GPT-4o 3-shot (our examples)",
}
GEN_PATTERN = re.compile(r"^(berturk|xlmr|crf)_gen(\d+)$")
NUMERIC_COLS = [
    "test_macro_f1", "ci95_low", "ci95_high", "f1_MIXED",
]

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
})


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def fmt3(v):
    """3 decimals, rounding .5 up (0.4295 -> 0.430), the same as the README table."""
    return str(Decimal(str(v)).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP))


def size_label(n):
    """0 -> '0', 2500 -> '2.5k', 5000 -> '5k', 10000 -> '10k'."""
    if n == 0:
        return "0"
    if n % 1000 == 0:
        return f"{n // 1000}k"
    return f"{n / 1000:g}k"


def parse_gen_run(run):
    """Return (family, size) for names like 'berturk_gen2500', else None."""
    m = GEN_PATTERN.match(str(run))
    if not m:
        return None
    return m.group(1), int(m.group(2))


def display_name(run):
    """Readable name for a run. Unknown names are returned unchanged."""
    run = str(run)
    if run in FIXED_NAMES:
        return FIXED_NAMES[run]
    parsed = parse_gen_run(run)
    if parsed:
        family, size = parsed
        return f"{FAMILY_NAMES[family]} + {size_label(size)}"
    return run


def colour_group(run):
    """'base' (grey), 'without' (blue) or 'with' (orange) our generated data."""
    run = str(run)
    if run == "majority" or run.startswith("rules_"):
        return "base"
    if run in ("gpt4o_0shot", "gpt4o_3shot"):
        return "without"
    if run == "gpt4o_3shot-ours":
        return "with"
    parsed = parse_gen_run(run)
    if parsed:
        return "without" if parsed[1] == 0 else "with"
    return "base"  # unknown run: neutral grey


def load_table():
    """Read results_table.csv; return a DataFrame or None if it is not usable."""
    if not TABLE.exists():
        print("results/results_table.csv not found. Run: python src/score_all.py")
        return None
    try:
        df = pd.read_csv(TABLE)
    except Exception as exc:  # unreadable or empty file
        print(f"skipped results table: cannot read it ({exc})")
        return None
    if "run" not in df.columns:
        print("skipped results table: no 'run' column")
        return None
    for col in NUMERIC_COLS:
        if col not in df.columns:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors="coerce")  # empty cell -> NaN
    return df


def read_test_docs():
    """Set of doc ids in data/splits/test.txt (None if the file is missing)."""
    if not TEST_SPLIT.exists():
        print(f"warning: {TEST_SPLIT.relative_to(REPO_ROOT)} not found, using all rows")
        return None
    docs = set()
    for line in TEST_SPLIT.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            docs.add(line)
    return docs


def save(fig, name):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / name
    fig.savefig(path, dpi=DPI, facecolor="white")
    plt.close(fig)
    print(f"saved results/figures/{name}")


def asymmetric_error(values, lows, highs):
    """Return a (2, n) array of non-negative errors, or None if no interval at all."""
    values, lows, highs = (np.asarray(a, dtype=float) for a in (values, lows, highs))
    has = ~(np.isnan(lows) | np.isnan(highs))
    if not has.any():
        return None
    left = np.where(has, np.maximum(values - lows, 0), 0.0)
    right = np.where(has, np.maximum(highs - values, 0), 0.0)
    return np.vstack([left, right])


# ----------------------------------------------------------------------------
# Figure 1: horizontal bars, all models, best on top
# ----------------------------------------------------------------------------
def fig1_macro_f1(df):
    rows = df[df["test_macro_f1"].notna()].copy()
    for run in df.loc[df["test_macro_f1"].isna(), "run"]:
        print(f"fig1: skipped {run} (no test_macro_f1)")
    if rows.empty:
        print("fig1: skipped (no rows with a test score)")
        return
    rows = rows.sort_values("test_macro_f1", ascending=True)  # best ends up on top

    n = len(rows)
    fig, ax = plt.subplots(figsize=(9, 0.45 * n + 1.5))
    colours = {"base": GREY, "without": BLUE, "with": ORANGE}
    ypos = np.arange(n)
    values = rows["test_macro_f1"].to_numpy()
    bar_colours = [colours[colour_group(r)] for r in rows["run"]]
    ax.barh(ypos, values, color=bar_colours, height=0.7)

    for y, (_, r) in zip(ypos, rows.iterrows()):
        v, lo, hi = r["test_macro_f1"], r["ci95_low"], r["ci95_high"]
        right_edge = v
        if not (np.isnan(lo) or np.isnan(hi)):
            ax.errorbar(v, y, xerr=[[max(v - lo, 0)], [max(hi - v, 0)]],
                        fmt="none", ecolor="black", elinewidth=1, capsize=3)
            right_edge = max(hi, v)
        ax.text(right_edge + 0.012, y, fmt3(v), va="center", ha="left", fontsize=10,
                bbox=dict(facecolor="white", edgecolor="none", pad=1.0))

    ax.axvline(PAPER_BERTURK, color="black", linestyle="--", linewidth=0.8)
    ax.set_yticks(ypos)
    ax.set_yticklabels([display_name(r) for r in rows["run"]])
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.6, n - 0.4)
    ax.set_xlabel("Macro-F1 on the TurEngMix test set (25 posts, 1,373 words)")
    ax.set_title("Language identification: all models")

    handles = [
        Patch(color=GREY, label="Simple baselines"),
        Patch(color=BLUE, label="Without our data"),
        Patch(color=ORANGE, label="With our data"),
        Line2D([0], [0], color="black", linestyle="--", linewidth=0.8,
               label=f"TurEngMix paper, BERTurk ({PAPER_BERTURK:.3f})"),
    ]
    ax.legend(handles=handles, loc="lower right", frameon=True)
    fig.tight_layout()
    save(fig, "fig1_macro_f1.png")


# ----------------------------------------------------------------------------
# Figure 2: effect of the amount of generated data
# ----------------------------------------------------------------------------
def family_points(df, family, value_col, with_ci):
    """Sorted points (size, value, lo, hi) for one model family."""
    pts = []
    for _, r in df.iterrows():
        parsed = parse_gen_run(r["run"])
        if not parsed or parsed[0] != family or parsed[1] not in SIZES:
            continue
        v = r[value_col]
        if pd.isna(v):
            print(f"fig2: skipped {r['run']} for {value_col} (no value)")
            continue
        lo = r["ci95_low"] if with_ci else np.nan
        hi = r["ci95_high"] if with_ci else np.nan
        pts.append((parsed[1], v, lo, hi))
    return sorted(pts)


def fig2_data_size(df):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    panels = [
        (axes[0], "test_macro_f1", True, "Overall (macro-F1)"),
        (axes[1], "f1_MIXED", False, "MIXED words (F1)"),
    ]
    drawn = []
    for ax, col, with_ci, title in panels:
        for family, colour in FAMILY_COLOURS.items():
            pts = family_points(df, family, col, with_ci)
            if len(pts) < 2:
                if pts or any(parse_gen_run(r) and parse_gen_run(r)[0] == family
                              for r in df["run"]):
                    print(f"fig2: skipped {FAMILY_NAMES[family]} in '{title}' "
                          f"(needs at least 2 points)")
                continue
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            err = asymmetric_error(ys, [p[2] for p in pts], [p[3] for p in pts]) \
                if with_ci else None
            ax.errorbar(xs, ys, yerr=err, color=colour, marker="o", linewidth=1.8,
                        markersize=6, capsize=3, elinewidth=1)
            if family not in drawn:
                drawn.append(family)
        ax.set_xticks(SIZES)
        ax.set_xticklabels([size_label(s) for s in SIZES])
        ax.set_xlabel("Generated sentences added to training")
        ax.set_ylabel("F1")
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.3)

    axes[0].axhline(PAPER_BERTURK, color="black", linestyle="--", linewidth=0.8)
    axes[0].text(0.98, PAPER_BERTURK, "TurEngMix paper, BERTurk",
                 transform=axes[0].get_yaxis_transform(),
                 ha="right", va="bottom", fontsize=9)
    axes[1].set_ylim(0, 1)

    if not drawn:
        print("fig2: skipped (no model family has 2 or more points)")
        plt.close(fig)
        return
    handles = [Line2D([0], [0], color=FAMILY_COLOURS[f], marker="o", linewidth=1.8,
                      label=FAMILY_NAMES[f]) for f in drawn]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False)
    fig.suptitle("Does adding generated data help on real text?", fontsize=13)
    fig.tight_layout(rect=[0, 0.07, 1, 0.95])
    save(fig, "fig2_data_size.png")


# ----------------------------------------------------------------------------
# Figure 3: confusion matrices, baseline vs best run
# ----------------------------------------------------------------------------
def confusion_counts(run, test_docs):
    """Sum 6x6 count matrices over all seed files of a run (None if no files)."""
    files = sorted(PRED_DIR.glob(f"{run}_seed*_test.csv"))
    if not files:
        single = PRED_DIR / f"{run}_test.csv"
        files = [single] if single.exists() else []
    if not files:
        print(f"fig3: skipped {run} (no prediction files in results/predictions/)")
        return None

    index = {lab: i for i, lab in enumerate(LABELS)}
    counts = np.zeros((len(LABELS), len(LABELS)), dtype=int)
    used = 0
    for f in files:
        try:
            d = pd.read_csv(f, dtype=str, keep_default_na=False)
        except Exception as exc:
            print(f"fig3: skipped file {f.name} ({exc})")
            continue
        if not {"lid", "pred_lid"} <= set(d.columns):
            print(f"fig3: skipped file {f.name} (needs columns lid and pred_lid)")
            continue
        if test_docs is not None and "doc_id" in d.columns:
            d = d[d["doc_id"].str.strip().isin(test_docs)]
        if len(d) != EXPECTED_TEST_ROWS:
            print(f"warning: {f.name} has {len(d)} test rows, expected {EXPECTED_TEST_ROWS}")
        for gold, pred in zip(d["lid"].str.strip(), d["pred_lid"].str.strip()):
            if gold in index and pred in index:  # ignore labels outside the six
                counts[index[gold], index[pred]] += 1
        used += 1
    return counts if used else None


def draw_confusion(ax, counts, title, cmap):
    """Draw one matrix (row-percent colours, percent + count text); return the image."""
    row_sums = counts.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        pct = np.where(row_sums > 0, counts / row_sums * 100, np.nan)
    im = ax.imshow(np.ma.masked_invalid(pct), cmap=cmap, vmin=0, vmax=100)
    for i in range(len(LABELS)):
        for j in range(len(LABELS)):
            if row_sums[i, 0] == 0:
                ax.text(j, i, "–", ha="center", va="center", color="black", fontsize=10)
                continue
            colour = "white" if pct[i, j] > 50 else "black"
            ax.text(j, i, f"{pct[i, j]:.0f}%\n({counts[i, j]})", ha="center",
                    va="center", color=colour, fontsize=10)
    ax.set_xticks(range(len(LABELS)))
    ax.set_yticks(range(len(LABELS)))
    ax.set_xticklabels(LABELS, rotation=45, ha="right")
    ax.set_yticklabels(LABELS)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("Correct label")
    ax.set_title(title)
    return im


def fig3_confusion(df):
    test_docs = read_test_docs()
    scores = {}
    if df is not None:
        scores = dict(zip(df["run"], df["test_macro_f1"]))

    panels = []
    for run in (BASELINE_RUN, BEST_RUN):
        counts = confusion_counts(run, test_docs)
        if counts is None:
            continue
        title = display_name(run)
        score = scores.get(run)
        if score is not None and not pd.isna(score):
            title += f" (macro-F1 {fmt3(score)})"
        panels.append((counts, title))
    if not panels:
        print("fig3: skipped (no prediction files for either run)")
        return

    n = len(panels)
    fig, axes = plt.subplots(1, n + 1, figsize=(13 if n == 2 else 7.5, 5.5),
                             gridspec_kw={"width_ratios": [1] * n + [0.05]})
    axes = np.atleast_1d(axes)
    cmap = plt.get_cmap("Blues").copy()
    cmap.set_bad("white")
    im = None
    for ax, (counts, title) in zip(axes[:n], panels):
        im = draw_confusion(ax, counts, title, cmap)
    fig.colorbar(im, cax=axes[n], label="% of the row")
    fig.suptitle("Where the models go wrong (test set, all seeds pooled)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    save(fig, "fig3_confusion.png")


# ----------------------------------------------------------------------------
def main():
    df = load_table()
    if df is not None:
        for name, func in (("fig1", fig1_macro_f1), ("fig2", fig2_data_size)):
            try:
                func(df)
            except Exception as exc:  # never stop because of one bad figure
                plt.close("all")
                print(f"{name}: skipped ({type(exc).__name__}: {exc})")
    try:
        fig3_confusion(df)
    except Exception as exc:
        plt.close("all")
        print(f"fig3: skipped ({type(exc).__name__}: {exc})")


if __name__ == "__main__":
    main()
