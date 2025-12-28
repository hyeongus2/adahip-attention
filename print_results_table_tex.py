#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv
import re
from pathlib import Path


CSV_PATH = Path("results/aggregate/results.csv")


def _norm_dataset(s: str) -> str:
    s0 = (s or "").strip()
    key = s0.lower().replace("-", "").replace("_", "").replace(" ", "")
    if key in {"wikitext", "wiki", "wt"}:
        return "WikiText"
    if key in {"pg19", "projectgutenberg19", "gutenberg19"}:
        return "PG19"
    return s0 if s0 else "Unknown"


def _norm_method(s: str) -> str:
    t = (s or "").strip()
    tl = t.lower().replace(" ", "").replace("_", "").replace("-", "")
    if tl in {"fa2", "flashattention2", "flashattn2"}:
        return "FA2"
    if tl in {"hip"}:
        return "HiP"
    if tl in {"adahip"}:
        return "AdaHiP"
    return t if t else "Unknown"


def _parse_stride_to_k(s: str) -> int:
    t = (s or "").strip().lower()
    m = re.fullmatch(r"(\d+)\s*k", t)
    if m:
        return int(m.group(1))
    m = re.fullmatch(r"(\d+)", t)
    if m:
        v = int(m.group(1))
        if v >= 1000:
            return int(round(v / 1024))
        return v
    m = re.search(r"(\d+)", t)
    return int(m.group(1)) if m else 0


def _stride_label(k: int) -> str:
    return f"{k}k" if k else "-"


def _to_int_or_none(s: str):
    t = (s or "").strip()
    if t == "" or t == "-" or t.lower() == "none":
        return None
    try:
        return int(float(t))
    except Exception:
        return None


def _to_float_or_none(s: str):
    t = (s or "").strip()
    if t == "" or t == "-" or t.lower() == "none":
        return None
    try:
        return float(t)
    except Exception:
        # sometimes like "1.23x"
        m = re.search(r"([0-9]+(\.[0-9]+)?)", t)
        return float(m.group(1)) if m else None


def _lower_keys(row: dict) -> dict:
    return {str(k).strip().lower(): k for k in row.keys()}


def _pick_first_existing(row: dict, candidates: list[str]) -> str:
    lk = _lower_keys(row)
    for c in candidates:
        if c.lower() in lk:
            return row[lk[c.lower()]]
    return ""


def _auto_find_col(fieldnames: list[str], include_any: list[str], exclude_any: list[str] | None = None) -> str | None:
    exclude_any = exclude_any or []
    f_lower = [(f, f.strip().lower()) for f in fieldnames]
    for orig, low in f_lower:
        ok = any(tok in low for tok in include_any) and not any(tok in low for tok in exclude_any)
        if ok:
            return orig
    return None


def _parse_speedup(s: str):
    v = _to_float_or_none(s)
    return v


def main():
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"CSV not found: {CSV_PATH}")

    with CSV_PATH.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise RuntimeError("CSV has no header row.")

        fieldnames = reader.fieldnames

        # --- auto-detect main columns (strict: if not found -> error) ---
        dataset_col = _auto_find_col(fieldnames, include_any=["dataset"], exclude_any=[])
        if dataset_col is None:
            # allow 'data' or 'corpus' fallback
            dataset_col = _auto_find_col(fieldnames, include_any=["corpus", "data"], exclude_any=[])
        if dataset_col is None:
            raise RuntimeError(f"Cannot find dataset column in headers: {fieldnames}")

        stride_col = _auto_find_col(fieldnames, include_any=["stride"], exclude_any=[])
        if stride_col is None:
            stride_col = _auto_find_col(fieldnames, include_any=["context"], exclude_any=["text", "length"])
        if stride_col is None:
            raise RuntimeError(f"Cannot find stride/context column in headers: {fieldnames}")

        method_col = _auto_find_col(fieldnames, include_any=["method"], exclude_any=[])
        if method_col is None:
            method_col = _auto_find_col(fieldnames, include_any=["attn", "algo"], exclude_any=[])
        if method_col is None:
            raise RuntimeError(f"Cannot find method column in headers: {fieldnames}")

        # PPL column: look for ppl/perplexity (most important)
        ppl_col = _auto_find_col(fieldnames, include_any=["ppl"], exclude_any=["std", "var", "stderr"])
        if ppl_col is None:
            ppl_col = _auto_find_col(fieldnames, include_any=["perplex"], exclude_any=["std", "var", "stderr"])
        if ppl_col is None:
            raise RuntimeError(f"Cannot find PPL column in headers: {fieldnames}")

        # speedup column
        speedup_col = _auto_find_col(fieldnames, include_any=["speedup"], exclude_any=[])
        if speedup_col is None:
            speedup_col = _auto_find_col(fieldnames, include_any=["tokens_per_sec", "tok_per_sec", "tps"], exclude_any=[])
            # If only TPS exists, user likely has separate FA2 TPS; we can't safely derive speedup without more info.
            if speedup_col is not None:
                raise RuntimeError(
                    "Found TPS-like column but no speedup column. "
                    "Your CSV must contain speedup vs FA2 to print this table safely."
                )
        if speedup_col is None:
            raise RuntimeError(f"Cannot find speedup column in headers: {fieldnames}")

        # optional hyperparams
        k_col = _auto_find_col(fieldnames, include_any=["k"], exclude_any=["block", "stride", "size"])
        alpha_col = _auto_find_col(fieldnames, include_any=["alpha"], exclude_any=[])
        power_col = _auto_find_col(fieldnames, include_any=["power"], exclude_any=[])

        rows = []
        for r in reader:
            dataset = _norm_dataset(r.get(dataset_col, ""))
            stride_k = _parse_stride_to_k(r.get(stride_col, ""))
            stride = _stride_label(stride_k)
            method = _norm_method(r.get(method_col, ""))

            k = _to_int_or_none(r.get(k_col, "")) if k_col else None
            alpha = _to_float_or_none(r.get(alpha_col, "")) if alpha_col else None
            power = _to_float_or_none(r.get(power_col, "")) if power_col else None

            ppl = _to_float_or_none(r.get(ppl_col, ""))
            speedup = _parse_speedup(r.get(speedup_col, ""))

            # enforce FA2 formatting
            if method == "FA2":
                k = None
                alpha = None
                power = None
                if speedup is None:
                    speedup = 1.0

            rows.append(
                {
                    "dataset": dataset,
                    "stride_k": stride_k,
                    "stride": stride,
                    "method": method,
                    "k": k,
                    "alpha": alpha,
                    "power": power,
                    "ppl": ppl,
                    "speedup": speedup,
                }
            )

    dataset_order = {"WikiText": 0, "PG19": 1}
    stride_order = {32: 0, 64: 1, 128: 2}
    method_order = {"FA2": 0, "HiP": 1, "AdaHiP": 2}

    def sort_key(x):
        ds = dataset_order.get(x["dataset"], 999)
        st = stride_order.get(x["stride_k"], 999)
        me = method_order.get(x["method"], 999)
        kk = x["k"] if x["k"] is not None else 10**9
        aa = x["alpha"] if x["alpha"] is not None else 10**9
        pp = x["power"] if x["power"] is not None else 10**9
        return (ds, st, me, kk, aa, pp)

    rows.sort(key=sort_key)

    cur_group = None
    first_group = True

    for x in rows:
        group = (x["dataset"], x["stride"])
        if group != cur_group:
            if not first_group:
                print(r"\midrule")
                print()
            first_group = False
            cur_group = group
            print(f"% {group[0]} {group[1]}")

        dataset = x["dataset"]
        stride = x["stride"]
        method = x["method"]

        # display rules
        if method == "FA2":
            k_s, alpha_s, power_s = "-", "-", "-"
        elif method == "HiP":
            k_s = "-" if x["k"] is None else str(x["k"])
            alpha_s, power_s = "-", "-"
        else:  # AdaHiP
            k_s = "-" if x["k"] is None else str(x["k"])
            # keep 0.75 / 1.0 / 1.25 formatting
            alpha_s = "-" if x["alpha"] is None else (f"{x['alpha']:.2f}".rstrip("0").rstrip(".") if (x["alpha"] % 1) else f"{x['alpha']:.1f}")
            power_s = "-" if x["power"] is None else (f"{x['power']:.2f}".rstrip("0").rstrip(".") if (x["power"] % 1) else f"{x['power']:.1f}")

        ppl = x["ppl"]
        speedup = x["speedup"]

        # IMPORTANT: rounding is to 2 decimals (3rd decimal rounds)
        ppl_s = "-" if ppl is None else f"{ppl:.2f}"
        sp_s = "-" if speedup is None else f"{speedup:.2f}x"

        print(f"{dataset} & {stride} & {method} & {k_s} & {alpha_s} & {power_s} & {ppl_s} & {sp_s} \\\\")


if __name__ == "__main__":
    main()
