# compare_classic_vs_obsqs_avg.py
from __future__ import annotations
from typing import List, Sequence, Callable, Optional, Tuple, Dict, Any
import csv, sys, time, random, math
from pathlib import Path
from statistics import mean, stdev

def quick_sort_inplace(a: List[int]) -> None:
    def _partition(lo: int, hi: int) -> int:
        pivot = a[(lo + hi) >> 1]
        i, j = lo - 1, hi + 1
        while True:
            i += 1
            while a[i] < pivot: i += 1
            j -= 1
            while a[j] > pivot: j -= 1
            if i >= j: return j
            a[i], a[j] = a[j], a[i]
    def _q(lo: int, hi: int) -> None:
        while lo < hi:
            p = _partition(lo, hi)
            if (p - lo) < (hi - (p + 1)):
                _q(lo, p); lo = p + 1
            else:
                _q(p + 1, hi); hi = p
    if len(a) > 1: _q(0, len(a) - 1)

def binary_search_classic(arr: Sequence[int], key: int) -> int:
    lo, hi = 0, len(arr) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        v = arr[mid]
        if v < key: lo = mid + 1
        elif v > key: hi = mid - 1
        else: return mid
    return -1

def obsqs_search(arr: Sequence[int], key: int) -> int:
    lo, hi = 0, len(arr) - 1
    if hi < 0: return -1
    if hi == 0: return 0 if arr[0] == key else -1
    while lo <= hi:
        mid = (lo + hi) >> 1
        v = arr[mid]
        lt = -int(key < v); gt = -int(key > v); eq = ~(lt | gt)
        lo = (lo & ~gt) | ((mid + 1) & gt)
        hi = (hi & ~lt) | ((mid - 1) & lt)
        if eq == -1: return mid
    return -1

def load_dataset_csv(path: Path) -> list[int]:
    if not path.exists(): print(f"[SKIP] {path}", file=sys.stderr); return []
    data: list[int] = []
    with path.open(newline="") as f:
        rd = csv.DictReader(f)
        if "value" not in (rd.fieldnames or []):
            print(f"[ERROR] {path} missing 'value' header", file=sys.stderr); return []
        for row in rd:
            try: data.append(int(row["value"]))
            except: pass
    if not data: print(f"[SKIP] No ints: {path}", file=sys.stderr)
    return data

def prep_queries(sorted_arr: Sequence[int], n_queries: int) -> list[int]:
    n = len(sorted_arr)
    if n == 0: return []
    rng = random.Random(42); qs: list[int] = []
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
    if total == -1: print("")  # keep var used
    return e - s, len(queries) * repeats

def print_line(label: str, seconds: float, ops: int, baseline: Optional[float]) -> None:
    qps = ops / seconds if seconds > 0 else float('inf')
    if baseline is None:
        print(f"[{label:<26}] {seconds:10.6f} s   {qps:12.0f} qps   (baseline)")
    else:
        speed = baseline / seconds if seconds > 0 else float('inf')
        print(f"[{label:<26}] {seconds:10.6f} s   {qps:12.0f} qps   ×{speed:5.2f} vs classic")

def run_one_file(path: Path,
                 n_queries: Optional[int] = None,
                 repeats: int = 2,
                 verify: bool = True) -> Optional[Dict[str, float]]:
    data = load_dataset_csv(path)
    if not data: return None
    baseline_sorted = sorted(data); n = len(baseline_sorted)
    if n_queries is None: n_queries = min(2 * n, 200_000)
    queries = prep_queries(baseline_sorted, n_queries)
    print(f"\n=== {path.name}  (n={n}, queries={len(queries)}, repeats={repeats}) ===")
    if verify and n > 0:
        for q in [baseline_sorted[0], baseline_sorted[-1], baseline_sorted[n//2]]:
            assert binary_search_classic(baseline_sorted, q) != -1
            assert obsqs_search(baseline_sorted, q) != -1
    t_classic_search, ops = bench_batch(binary_search_classic, baseline_sorted, queries, repeats=repeats)
    print_line("Classic: search only", t_classic_search, ops, baseline=None)
    t_obsqs_search, _ = bench_batch(obsqs_search, baseline_sorted, queries, repeats=repeats)
    print_line("OBSQS:   search only", t_obsqs_search, ops, baseline=t_classic_search)
    classic_arr = data[:]; t0 = time.perf_counter(); quick_sort_inplace(classic_arr); t_sort_classic = time.perf_counter() - t0
    t_search_classic, _ = bench_batch(binary_search_classic, classic_arr, queries, repeats=repeats)
    t_classic_total = t_sort_classic + t_search_classic
    print_line("Classic: sort + search", t_classic_total, ops, baseline=None)
    obsqs_arr = data[:]; t1 = time.perf_counter(); quick_sort_inplace(obsqs_arr); t_sort_obsqs = time.perf_counter() - t1
    t_search_obsqs, _ = bench_batch(obsqs_search, obsqs_arr, queries, repeats=repeats)
    t_obsqs_total = t_sort_obsqs + t_search_obsqs
    print_line("OBSQS:   sort + search", t_obsqs_total, ops, baseline=t_classic_total)
    return {
        "file": path.name, "n": n, "queries": len(queries) * repeats,
        "classic_search": t_classic_search, "obsqs_search": t_obsqs_search,
        "classic_total": t_classic_total, "obsqs_total": t_obsqs_total
    }

# Print
def print_perfile(rows: list[Dict[str, float]]) -> None:
    if not rows: return
    print("\n================ Per-file Results ================")
    hdr = (f"{'File':18} {'n':>9} {'Queries':>10}  "
           f"{'Classic(srch)':>13} {'OBSQS(srch)':>13}  "
           f"{'Classic(total)':>15} {'OBSQS(total)':>13}  "
           f"{'Speedup(obsqs/cls)':>18}")
    print(hdr)
    for r in rows:
        sp = r["classic_total"]/r["obsqs_total"] if r["obsqs_total"]>0 else float('inf')
        print(f"{r['file']:18} {int(r['n']):9d} {int(r['queries']):10d}  "
              f"{r['classic_search']:13.4f} {r['obsqs_search']:13.4f}  "
              f"{r['classic_total']:15.4f} {r['obsqs_total']:13.4f}  "
              f"{sp:18.2f}")

def _vals(rows, key): return [r[key] for r in rows if key in r and math.isfinite(r[key])]
def print_group_stats(rows: list[Dict[str, float]], label: str) -> None:
    if not rows: return
    print(f"\n===== Size {label} — Averages (± std) over {len(rows)} runs =====")
    print(f"{'Metric':18} {'Mean(s)':>12} {'Std(s)':>12}")
    for k in ["classic_search","obsqs_search","classic_total","obsqs_total"]:
        vs = _vals(rows, k)
        m = mean(vs) if vs else float('nan')
        sd = stdev(vs) if len(vs)>=2 else 0.0 if vs else float('nan')
        print(f"{k:18} {m:12.4f} {sd:12.4f}")
    ct = _vals(rows,"classic_total"); ot = _vals(rows,"obsqs_total")
    if ct and ot:
        ratio = (mean(ct)/mean(ot)) if mean(ot)>0 else float('inf')
        print(f"{'Speedup cls/obsqs':18} {ratio:12.4f} {'(ratio of means)':>12}")

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
    VERIFY = True

    for n in sizes:
        group_rows: list[Dict[str, float]] = []
        for i in variants:
            p = base_dir / f"{n}_dataset_{i}.csv"
            r = run_one_file(p, n_queries=None, repeats=REPEATS[n], verify=VERIFY)
            if r: group_rows.append(r)
        print_perfile(group_rows)
        print_group_stats(group_rows, str(n))