# compare_binary_searches_no_obsqs_avg.py
from __future__ import annotations
from typing import List, Sequence, Callable, Optional, Tuple, Dict, Any
import csv, sys, time, random, math
from pathlib import Path
from statistics import mean, stdev

def binary_search_normal(arr: Sequence[int], key: int) -> int:
    lo, hi = 0, len(arr) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        v = arr[mid]
        if v < key: lo = mid + 1
        elif v > key: hi = mid - 1
        else: return mid
    return -1

def bbs_lower_bound(arr: Sequence[int], key: int) -> int:
    n = len(arr)
    if n == 0: return 0
    step = 1 << (n.bit_length() - 1)
    idx = -1
    while step:
        probe = idx + step
        if probe < n and arr[probe] < key: idx = probe
        step >>= 1
    return idx + 1

def bbs_search(arr: Sequence[int], key: int) -> int:
    i = bbs_lower_bound(arr, key)
    return i if i < len(arr) and arr[i] == key else -1

def bitflip_lower_bound(arr: Sequence[int], key: int) -> int:
    n = len(arr)
    if n == 0: return 0
    msb = n.bit_length() - 1
    m = 0
    for b in range(msb, -1, -1):
        cand = m | (1 << b)
        if cand <= n and arr[cand - 1] < key: m = cand
    return m

def bitflip_search(arr: Sequence[int], key: int) -> int:
    i = bitflip_lower_bound(arr, key)
    return i if i < len(arr) and arr[i] == key else -1

def bitflip_upper_bound(arr: Sequence[int], key: int) -> int:
    n = len(arr)
    if n == 0: return 0
    msb = n.bit_length() - 1
    m = 0
    for b in range(msb, -1, -1):
        cand = m | (1 << b)
        if cand <= n and arr[cand - 1] <= key: m = cand
    return m

def load_dataset_csv(path: Path) -> list[int]:
    if not path.exists(): print(f"[SKIP] {path}", file=sys.stderr); return []
    data: list[int] = []
    with path.open(newline="") as f:
        rd = csv.DictReader(f)
        if "value" not in (rd.fieldnames or []):
            print(f"[ERROR] {path} missing header 'value'", file=sys.stderr); return []
        for row in rd:
            try: data.append(int(row["value"]))
            except: pass
    if not data: print(f"[SKIP] No ints: {path}", file=sys.stderr)
    return data

def prep_queries(sorted_arr: Sequence[int], n_queries: int) -> list[int]:
    n = len(sorted_arr)
    if n == 0: return []
    rng = random.Random(42)
    qs: list[int] = []
    for _ in range(n_queries // 2): qs.append(sorted_arr[rng.randrange(n)])
    lo, hi = sorted_arr[0], sorted_arr[-1]
    for _ in range(n_queries // 4): qs.append(lo - rng.randint(1, 3))
    for _ in range(n_queries // 4): qs.append(hi + rng.randint(1, 3))
    while len(qs) < n_queries:
        i = rng.randrange(n); qs.append(sorted_arr[i] + (1 if rng.random() < 0.5 else -1))
    rng.shuffle(qs); return qs

def bench_batch(search_fn: Callable[[Sequence[int], int], int],
                arr: Sequence[int],
                queries: Sequence[int],
                repeats: int = 1) -> Tuple[float, int]:
    total = 0
    s = time.perf_counter()
    for _ in range(repeats):
        for q in queries:
            total += (search_fn(arr, q) >= 0)
    e = time.perf_counter()
    if total == -1: print("")
    return e - s, len(queries) * repeats

def print_one_result(label: str, seconds: float, ops: int, baseline: Optional[float]) -> None:
    qps = ops / seconds if seconds > 0 else float('inf')
    if baseline is None:
        print(f"[{label:<20}] {seconds:10.6f} s   {qps:12.0f} qps   (baseline)")
    else:
        speedup = baseline / seconds if seconds > 0 else float('inf')
        print(f"[{label:<20}] {seconds:10.6f} s   {qps:12.0f} qps   ×{speedup:5.2f} vs normal")

def run_one_file(path: Path,
                 n_queries: Optional[int] = None,
                 repeats: int = 2,
                 verify: bool = True) -> Optional[Dict[str, float]]:
    data = load_dataset_csv(path)
    if not data: return None
    sorted_data = sorted(data)
    n = len(sorted_data)
    if n_queries is None: n_queries = min(2 * n, 200_000)
    queries = prep_queries(sorted_data, n_queries)
    print(f"\n=== {path.name}  (n={n}, queries={len(queries)}, repeats={repeats}) ===")
    if verify and n > 0:
        for q in [sorted_data[0], sorted_data[-1], sorted_data[n//2]]:
            assert binary_search_normal(sorted_data, q) != -1
            assert bbs_search(sorted_data, q) != -1
            assert bitflip_search(sorted_data, q) != -1
    t_norm, ops = bench_batch(binary_search_normal, sorted_data, queries, repeats=repeats)
    print_one_result("Normal Binary Search", t_norm, ops, baseline=None)
    t_bbs, _ = bench_batch(bbs_search, sorted_data, queries, repeats=repeats)
    print_one_result("Bitwise Binary Search", t_bbs, ops, baseline=t_norm)
    t_bitflip, _ = bench_batch(bitflip_search, sorted_data, queries, repeats=repeats)
    print_one_result("Local Bit Flip Search", t_bitflip, ops, baseline=t_norm)
    return {"file": path.name, "n": n, "queries": len(queries) * repeats,
            "t_normal": t_norm, "t_bbs": t_bbs, "t_bitflip": t_bitflip}

# Print
def _finite(vals): return [v for v in vals if v is not None and math.isfinite(v)]

def print_perfile(rows: list[Dict[str, float]]) -> None:
    if not rows: return
    print("\n================ Per-file Results ================")
    hdr = f"{'File':18} {'n':>9} {'Queries':>10}  {'Normal(s)':>10} {'BBS(s)':>10} {'BitFlip(s)':>11}"
    print(hdr)
    for r in rows:
        print(f"{r['file']:18} {int(r['n']):9d} {int(r['queries']):10d}  "
              f"{r['t_normal']:10.4f} {r['t_bbs']:10.4f} {r['t_bitflip']:11.4f}")

def print_group_stats(rows: list[Dict[str, float]], label: str) -> None:
    if not rows: return
    print(f"\n===== Size {label} — Averages (± std) over {len(rows)} runs =====")
    print(f"{'Metric':16} {'Mean(s)':>12} {'Std(s)':>12}")
    for k in ["t_normal", "t_bbs", "t_bitflip"]:
        vals = _finite([r[k] for r in rows])
        m = mean(vals) if vals else float('nan')
        sd = stdev(vals) if len(vals) >= 2 else 0.0 if vals else float('nan')
        print(f"{k:16} {m:12.4f} {sd:12.4f}")

# Main
if __name__ == "__main__":
    try:
        base_dir = Path(__file__).resolve().parent
    except NameError:
        base_dir = Path.cwd()
    print(f"Looking for datasets in: {base_dir}")

    sizes = [100, 10_000, 1_000_000]
    variants = [1,2,3,4,5]
    REPEATS = {100:5, 10_000:3, 1_000_000:2}
    VERIFY_RESULTS = True

    for n in sizes:
        group_rows: list[Dict[str, float]] = []
        for i in variants:
            path = base_dir / f"{n}_dataset_{i}.csv"
            r = run_one_file(path, n_queries=None, repeats=REPEATS[n], verify=VERIFY_RESULTS)
            if r: group_rows.append(r)
        print_perfile(group_rows)
        print_group_stats(group_rows, str(n))