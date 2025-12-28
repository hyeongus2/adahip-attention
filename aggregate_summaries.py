#!/usr/bin/env python3
# English comments only.

from __future__ import annotations

import argparse
import csv
import math
import re
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Optional


FILENAME_RE = re.compile(
    r"^ppl_"
    r"(?P<dataset>[^_]+)_"
    r"(?P<method>[^_]+)_"
    r"(?P<model>.+?)_"
    r"s(?P<stride>\d+)"
    r"(?:_k(?P<k>\d+))?"
    r"(?:_a(?P<alpha>[^_]+))?"
    r"(?:_p(?P<power>[^_]+))?"
    r"_(?P<date>\d{8})_(?P<time>\d{6})"
    r"\.summary\.txt$"
)

STEP_RE = re.compile(
    r"^step\s+(?P<step>\d+)\s+PPL:\s+(?P<ppl>[-+]?(\d+(\.\d*)?|\.\d+|nan))"
    r".*,\s+(?P<sec>[-+]?(\d+(\.\d*)?|\.\d+))\s+sec$"
)


def _to_float(s: Optional[str]) -> Optional[float]:
    if s is None:
        return None
    s = s.strip()
    if s.lower() == "nan":
        return math.nan
    try:
        return float(s)
    except ValueError:
        return None


def _ap_to_float(s: Optional[str]) -> Optional[float]:
    """
    Convert strings like '1p25' -> 1.25, '0p75' -> 0.75, '2p0' -> 2.0.
    """
    if s is None:
        return None
    s = s.strip()
    if "p" in s:
        s = s.replace("p", ".", 1)
    return _to_float(s)


@dataclass
class RunRow:
    dataset: str
    method: str
    stride: int
    k: Optional[int]
    alpha: Optional[float]
    power: Optional[float]
    ppl_mean: float
    latency_mean_sec: float
    speedup: Optional[float] = None


def parse_summary_file(path: Path) -> Optional[RunRow]:
    m = FILENAME_RE.match(path.name)
    if not m:
        return None

    dataset = m.group("dataset")
    method = m.group("method")
    stride = int(m.group("stride"))

    k_str = m.group("k")
    alpha_str = m.group("alpha")
    power_str = m.group("power")

    k = int(k_str) if k_str is not None else None
    alpha = _ap_to_float(alpha_str) if alpha_str is not None else None
    power = _ap_to_float(power_str) if power_str is not None else None

    ppl_list: list[float] = []
    sec_list: list[float] = []

    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        sm = STEP_RE.match(line)
        if not sm:
            continue

        ppl = _to_float(sm.group("ppl"))
        sec = _to_float(sm.group("sec"))
        if ppl is None or sec is None:
            continue
        if isinstance(ppl, float) and math.isnan(ppl):
            continue

        ppl_list.append(float(ppl))
        sec_list.append(float(sec))

    if not ppl_list or not sec_list or len(ppl_list) != len(sec_list):
        return None

    return RunRow(
        dataset=dataset,
        method=method,
        stride=stride,
        k=k,
        alpha=alpha,
        power=power,
        ppl_mean=mean(ppl_list),
        latency_mean_sec=mean(sec_list),
        speedup=None,
    )


def _dataset_rank(dataset: str) -> int:
    order = {"wikitext": 0, "pg19": 1}
    return order.get(dataset, 999)


def _method_rank(method: str) -> int:
    order = {"fa2": 0, "hip": 1, "adahip": 2}
    return order.get(method, 999)


def _num_or_big(x: Optional[float], big: float = 1e18) -> float:
    return big if x is None else float(x)


def _int_or_big(x: Optional[int], big: int = 10**9) -> int:
    return big if x is None else int(x)


def sort_key(r: RunRow) -> tuple:
    return (
        _dataset_rank(r.dataset),
        r.stride,
        _method_rank(r.method),
        _int_or_big(r.k),
        _num_or_big(r.alpha),
        _num_or_big(r.power),
    )


def compute_speedup(rows: list[RunRow]) -> None:
    """
    For each (dataset, stride), define baseline latency as the mean latency_mean_sec of fa2 runs.
    speedup = baseline / method_latency
    fa2 speedup becomes 1.0 (since baseline == its own latency, assuming only one fa2 run).
    """
    # Collect baseline latencies per (dataset, stride)
    baselines: dict[tuple[str, int], float] = {}
    fa2_lat_list: dict[tuple[str, int], list[float]] = {}

    for r in rows:
        if r.method != "fa2":
            continue
        key = (r.dataset, r.stride)
        fa2_lat_list.setdefault(key, []).append(r.latency_mean_sec)

    for key, vals in fa2_lat_list.items():
        if vals:
            baselines[key] = mean(vals)

    # Assign speedup
    for r in rows:
        key = (r.dataset, r.stride)
        base = baselines.get(key)
        if base is None or r.latency_mean_sec <= 0:
            r.speedup = None
            continue
        r.speedup = base / r.latency_mean_sec


def write_csv(path: Path, rows: list[RunRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "dataset",
        "stride",
        "method",
        "k",
        "alpha",
        "power",
        "ppl_mean",
        "latency_mean_sec",
        "speedup",
    ]

    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(
                {
                    "dataset": r.dataset,
                    "stride": r.stride,
                    "method": r.method,
                    "k": "" if r.k is None else r.k,
                    "alpha": "" if r.alpha is None else f"{r.alpha:g}",
                    "power": "" if r.power is None else f"{r.power:g}",
                    "ppl_mean": f"{r.ppl_mean:.6f}",
                    "latency_mean_sec": f"{r.latency_mean_sec:.6f}",
                    "speedup": "" if r.speedup is None else f"{r.speedup:.6f}",
                }
            )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", type=str, default="results/summaries")
    ap.add_argument("--out_dir", type=str, default="results/aggregate")
    args = ap.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)

    files = sorted(in_dir.glob("*.summary.txt"))
    if not files:
        raise SystemExit(f"No .summary.txt files found in: {in_dir}")

    all_rows: list[RunRow] = []
    skipped: list[str] = []

    for fp in files:
        row = parse_summary_file(fp)
        if row is None:
            skipped.append(fp.name)
            continue
        all_rows.append(row)

    # Compute speedup before sorting/writing
    compute_speedup(all_rows)

    # Sort into dataset/stride blocks
    all_rows.sort(key=sort_key)

    results_csv = out_dir / "results.csv"
    write_csv(results_csv, all_rows)

    ablation_rows = [r for r in all_rows if r.method == "adahip"]
    ablation_csv = out_dir / "ablation.csv"
    write_csv(ablation_csv, ablation_rows)

    print(f"[OK] Parsed runs: {len(all_rows)}")
    print(f"[OK] Wrote: {results_csv}  (all methods)")
    print(f"[OK] Wrote: {ablation_csv}  (adahip only: {len(ablation_rows)})")
    if skipped:
        print(f"[WARN] Skipped files (failed parse): {len(skipped)}")
        for s in skipped[:20]:
            print("  -", s)
        if len(skipped) > 20:
            print("  ...")


if __name__ == "__main__":
    main()
