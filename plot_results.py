#!/usr/bin/env python3
# plot_results.py
#
# You asked for ONLY 3 PNG figures:
#   (1) MAIN: 128K context, 2 panels (WikiText / PG19)
#   (2) APPENDIX: WikiText only, 3 panels (32K / 64K / 128K)
#   (3) APPENDIX: PG19 only, 3 panels (32K / 64K / 128K)
#
# All figures include FA2 reference dashed lines:
#   - vertical: x = speedup_FA2 (typically 1.0)
#   - horizontal: y = ppl_FA2
#
# Read:
#   /work/results/aggregate/results.csv
# Write:
#   /work/results/aggregate/figures_png/*.png
#
# Run:
#   python plot_results.py

import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


CSV_PATH = Path("/work/results/aggregate/results.csv")
OUT_DIR = Path("/work/results/aggregate/figures")


# ----------------------------
# Helpers
# ----------------------------
def ensure_outdir():
    OUT_DIR.mkdir(parents=True, exist_ok=True)


def save_png(fig, name: str):
    ensure_outdir()
    path = OUT_DIR / f"{name}.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    print(f"[saved] {path}")


def pick_col(df: pd.DataFrame, candidates: list[str]) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(f"Missing columns. Tried: {candidates}\nAvailable: {list(df.columns)}")


def norm_method(m: str) -> str:
    m0 = str(m).strip().lower()
    if m0 in ["fa2", "flashattention2", "flashattention-2", "flash_attention_2", "flash_attn_2"]:
        return "FA2"
    if m0 in ["hip", "ppl hip", "hip-attention", "hip_attention"]:
        return "HiP"
    if m0 in ["adahip", "ada hip", "adaptive hip", "adahip (ours)", "adahip_ours"]:
        return "AdaHiP"
    return str(m).strip()


def dataset_key(name: str) -> str:
    n = str(name).strip().lower()
    if "wiki" in n:
        return "WikiText"
    if "pg19" in n or "pg-19" in n:
        return "PG19"
    return str(name).strip()


def parse_stride_to_int(s) -> int | None:
    if pd.isna(s):
        return None
    if isinstance(s, (int, np.integer)):
        return int(s)
    if isinstance(s, float) and np.isfinite(s):
        return int(s)

    t = str(s).strip().lower()
    if re.fullmatch(r"\d+", t):
        return int(t)
    m = re.fullmatch(r"(\d+(?:\.\d+)?)\s*k", t)
    if m:
        return int(float(m.group(1)) * 1024)
    t2 = t.replace("_", "")
    if re.fullmatch(r"\d+", t2):
        return int(t2)
    return None


def stride_label(s_int: int) -> str:
    if s_int == 32 * 1024:
        return "32K context"
    if s_int == 64 * 1024:
        return "64K context"
    if s_int == 128 * 1024:
        return "128K context"
    if s_int % 1024 == 0:
        return f"{s_int//1024}K context"
    return f"{s_int} tokens"


def robust_ylim(ax, yvals: np.ndarray):
    yvals = yvals[np.isfinite(yvals)]
    if yvals.size == 0:
        return
    lo = float(np.quantile(yvals, 0.02))
    hi = float(np.quantile(yvals, 0.98))
    if hi - lo < 1e-6:
        lo -= 0.01
        hi += 0.01
    pad = 0.18 * (hi - lo)
    ax.set_ylim(lo - pad, hi + pad)


def robust_xlim(ax, xvals: np.ndarray):
    xvals = xvals[np.isfinite(xvals)]
    if xvals.size == 0:
        return
    lo = float(np.min(xvals))
    hi = float(np.max(xvals))
    if hi - lo < 1e-6:
        lo -= 0.05
        hi += 0.05
    pad = 0.10 * (hi - lo)
    ax.set_xlim(lo - pad, hi + pad)


def get_fa2_baseline(sub: pd.DataFrame, col_speedup: str, col_ppl: str) -> tuple[float, float | None]:
    fa2 = sub[sub["method_norm"] == "FA2"]
    if fa2.empty:
        return 1.0, None
    x = float(np.median(fa2[col_speedup].to_numpy()))
    y = float(np.median(fa2[col_ppl].to_numpy()))
    return x, y


def add_fa2_reference_lines(ax, fa2_speedup: float, fa2_ppl: float | None):
    ax.axvline(fa2_speedup, linestyle="--", linewidth=1.2, alpha=0.85)
    if fa2_ppl is not None and np.isfinite(fa2_ppl):
        ax.axhline(fa2_ppl, linestyle="--", linewidth=1.2, alpha=0.85)


def scatter_by_method(ax, sub: pd.DataFrame, col_speedup: str, col_ppl: str, marker_map: dict[str, str]):
    for m in ["FA2", "HiP", "AdaHiP"]:
        ssub = sub[sub["method_norm"] == m]
        if ssub.empty:
            continue
        ax.scatter(
            ssub[col_speedup].to_numpy(),
            ssub[col_ppl].to_numpy(),
            marker=marker_map.get(m, "o"),
            alpha=0.85,
            label=m,
            edgecolors="none",
        )


def fig_legend_outside(fig, axes_list, y_anchor: float, ncol_max: int = 4):
    handles_all, labels_all = [], []
    for ax in axes_list:
        h, l = ax.get_legend_handles_labels()
        handles_all += h
        labels_all += l

    uniq = {}
    for h, l in zip(handles_all, labels_all):
        if l not in uniq:
            uniq[l] = h

    fig.legend(
        list(uniq.values()),
        list(uniq.keys()),
        loc="upper center",
        ncol=min(ncol_max, len(uniq)),
        frameon=True,
        bbox_to_anchor=(0.5, 1.00),
    )


# ----------------------------
# Load + normalize
# ----------------------------
if not CSV_PATH.exists():
    raise FileNotFoundError(f"CSV not found: {CSV_PATH}")

df = pd.read_csv(CSV_PATH)

col_dataset = pick_col(df, ["dataset", "Dataset"])
col_stride = pick_col(df, ["stride", "Stride"])
col_method = pick_col(df, ["method", "Method"])
col_ppl = pick_col(df, ["ppl", "ppl_mean", "PPL", "PPL_mean"])
col_speedup = pick_col(df, ["speedup", "speedup_mean", "Speedup", "Speedup_mean"])

df = df.copy()
df["dataset_key"] = df[col_dataset].apply(dataset_key)
df["method_norm"] = df[col_method].apply(norm_method)
df["stride_int"] = df[col_stride].apply(parse_stride_to_int)

df[col_ppl] = pd.to_numeric(df[col_ppl], errors="coerce")
df[col_speedup] = pd.to_numeric(df[col_speedup], errors="coerce")

df = df.dropna(subset=["dataset_key", "method_norm", "stride_int", col_ppl, col_speedup])

marker_map = {"FA2": "o", "HiP": "s", "AdaHiP": "^"}

stride_list = [s for s in [32 * 1024, 64 * 1024, 128 * 1024] if s in set(df["stride_int"])]
if not stride_list:
    stride_list = sorted(df["stride_int"].unique().tolist())


# ----------------------------
# (1) MAIN: 128K, 2 panels (WikiText / PG19)
# ----------------------------
stride_main = 128 * 1024
df128 = df[df["stride_int"] == stride_main].copy()
if df128.empty:
    raise RuntimeError("No rows for 128K stride found in results.csv (stride_int == 131072).")

datasets_main = [d for d in ["WikiText", "PG19"] if d in set(df128["dataset_key"])]
if not datasets_main:
    datasets_main = sorted(df128["dataset_key"].unique().tolist())

fig, axes = plt.subplots(1, len(datasets_main), figsize=(7.2 * len(datasets_main), 4.9), sharey=False)
if len(datasets_main) == 1:
    axes = [axes]

for ax, dset in zip(axes, datasets_main):
    sub = df128[df128["dataset_key"] == dset].copy()
    if sub.empty:
        ax.set_title(f"{dset} (no data)")
        ax.axis("off")
        continue

    fa2_x, fa2_y = get_fa2_baseline(sub, col_speedup, col_ppl)

    scatter_by_method(ax, sub, col_speedup, col_ppl, marker_map)
    add_fa2_reference_lines(ax, fa2_x, fa2_y)

    robust_xlim(ax, sub[col_speedup].to_numpy())
    robust_ylim(ax, sub[col_ppl].to_numpy())

    ax.set_title(f"{dset} (128K context)")
    ax.set_xlabel("Speedup relative to FA2 (higher is faster)")
    ax.set_ylabel("Perplexity (lower is better)")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)

fig_legend_outside(fig, axes, y_anchor=1.18, ncol_max=3)
fig.suptitle("Trade-off Plot: Perplexity vs Speedup (128K Context)", y=1.04)
fig.tight_layout()
save_png(fig, "main_128k_ppl_vs_speedup")
plt.close(fig)


# ----------------------------
# (2) APPENDIX: WikiText only, 3 panels (32K/64K/128K)
# ----------------------------
df_wiki = df[df["dataset_key"] == "WikiText"].copy()
if df_wiki.empty:
    print("[warn] No WikiText rows found. Skipping appendix_wikitext_by_stride.")
else:
    fig, axes = plt.subplots(1, len(stride_list), figsize=(6.6 * len(stride_list), 4.9), sharey=False)
    if len(stride_list) == 1:
        axes = [axes]

    for ax, s in zip(axes, stride_list):
        sub = df_wiki[df_wiki["stride_int"] == s].copy()
        if sub.empty:
            ax.set_title(f"WikiText / {stride_label(s)} (no data)")
            ax.axis("off")
            continue

        fa2_x, fa2_y = get_fa2_baseline(sub, col_speedup, col_ppl)

        scatter_by_method(ax, sub, col_speedup, col_ppl, marker_map)
        add_fa2_reference_lines(ax, fa2_x, fa2_y)

        robust_xlim(ax, sub[col_speedup].to_numpy())
        robust_ylim(ax, sub[col_ppl].to_numpy())

        ax.set_title(f"WikiText ({stride_label(s)})")
        ax.set_xlabel("Speedup relative to FA2 (higher is faster)")
        ax.set_ylabel("Perplexity (lower is better)")
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)

    fig_legend_outside(fig, axes, y_anchor=1.18, ncol_max=3)
    fig.suptitle("Appendix: WikiText — Perplexity vs Speedup Across Context Lengths", y=1.04)
    fig.tight_layout()
    save_png(fig, "appendix_wikitext_by_stride_ppl_vs_speedup")
    plt.close(fig)


# ----------------------------
# (3) APPENDIX: PG19 only, 3 panels (32K/64K/128K)
# ----------------------------
df_pg = df[df["dataset_key"] == "PG19"].copy()
if df_pg.empty:
    print("[warn] No PG19 rows found. Skipping appendix_pg19_by_stride.")
else:
    fig, axes = plt.subplots(1, len(stride_list), figsize=(6.6 * len(stride_list), 4.9), sharey=False)
    if len(stride_list) == 1:
        axes = [axes]

    for ax, s in zip(axes, stride_list):
        sub = df_pg[df_pg["stride_int"] == s].copy()
        if sub.empty:
            ax.set_title(f"PG19 / {stride_label(s)} (no data)")
            ax.axis("off")
            continue

        fa2_x, fa2_y = get_fa2_baseline(sub, col_speedup, col_ppl)

        scatter_by_method(ax, sub, col_speedup, col_ppl, marker_map)
        add_fa2_reference_lines(ax, fa2_x, fa2_y)

        robust_xlim(ax, sub[col_speedup].to_numpy())
        robust_ylim(ax, sub[col_ppl].to_numpy())

        ax.set_title(f"PG19 ({stride_label(s)})")
        ax.set_xlabel("Speedup relative to FA2 (higher is faster)")
        ax.set_ylabel("Perplexity (lower is better)")
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)

    fig_legend_outside(fig, axes, y_anchor=1.18, ncol_max=3)
    fig.suptitle("Appendix: PG19 — Perplexity vs Speedup Across Context Lengths", y=1.04)
    fig.tight_layout()
    save_png(fig, "appendix_pg19_by_stride_ppl_vs_speedup")
    plt.close(fig)

print("\nDone.")
print(f"PNGs saved to: {OUT_DIR}")
