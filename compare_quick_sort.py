# compare_quicksorts_avg.py
from typing import Iterable, List, Tuple, Optional, Dict, Any
import csv, os, sys, time, random, math
from statistics import mean, stdev

def _median_of_three(a: List[int], lo: int, hi: int) -> int:
    mid = (lo + hi) // 2
    x, y, z = a[lo], a[mid], a[hi]
    if x <= y <= z or z <= y <= x: return mid
    elif y <= x <= z or z <= x <= y: return lo
    else: return hi

def quicksort_compare_inplace(a: List[int]) -> None:
    n = len(a)
    if n <= 1: return
    stack: List[Tuple[int, int]] = [(0, n - 1)]
    while stack:
        lo, hi = stack.pop()
        while lo < hi:
            p_idx = _median_of_three(a, lo, hi)
            pivot = a[p_idx]
            i, j = lo - 1, hi + 1
            while True:
                i += 1
                while a[i] < pivot: i += 1
                j -= 1
                while a[j] > pivot: j -= 1
                if i >= j: break
                a[i], a[j] = a[j], a[i]
            left_lo, left_hi = lo, j
            right_lo, right_hi = j + 1, hi
            if (left_hi - left_lo) < (right_hi - right_lo):
                if right_lo < right_hi: stack.append((right_lo, right_hi))
                lo, hi = left_lo, left_hi
            else:
                if left_lo < left_hi: stack.append((left_lo, left_hi))
                lo, hi = right_lo, right_hi

def quicksort_compare(data: Iterable[int]) -> List[int]:
    a = list(data); quicksort_compare_inplace(a); return a

def oqs_bitwise_quicksort_inplace(a: List[int]) -> None:
    n = len(a)
    if n <= 1: return
    mn = min(a); bias = -mn if mn < 0 else 0
    if bias:
        for i in range(n): a[i] += bias
    mx = 0
    for x in a:
        if x > mx: mx = x
    if mx == 0:
        if bias:
            for i in range(n): a[i] -= bias
        return
    mask = 1 << (mx.bit_length() - 1)
    stack: List[Tuple[int, int, int]] = [(0, n - 1, mask)]
    while stack:
        lo, hi, m = stack.pop()
        if lo >= hi or m == 0: continue
        i, j = lo, hi
        while i <= j:
            while i <= j and (a[i] & m) == 0: i += 1
            while i <= j and (a[j] & m) != 0: j -= 1
            if i < j:
                a[i], a[j] = a[j], a[i]; i += 1; j -= 1
        next_m = m >> 1
        left_size = max(0, j - lo + 1)
        right_size = max(0, hi - i + 1)
        if left_size < right_size:
            if i < hi: stack.append((i, hi, next_m))
            if lo < j: stack.append((lo, j, next_m))
        else:
            if lo < j: stack.append((lo, j, next_m))
            if i < hi: stack.append((i, hi, next_m))
    if bias:
        for i in range(n): a[i] -= bias

def oqs_bitwise_quicksort(data: Iterable[int]) -> List[int]:
    a = list(data); oqs_bitwise_quicksort_inplace(a); return a

def load_dataset_csv(path: str) -> list[int]:
    if not os.path.exists(path):
        print(f"[SKIP] File not found: {path}", file=sys.stderr); return []
    data: list[int] = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        if "value" not in (reader.fieldnames or []):
            print(f"[ERROR] {path} must have a 'value' column header.", file=sys.stderr); return []
        for row in reader:
            try: data.append(int(row["value"]))
            except: pass
    if not data: print(f"[SKIP] No valid integers in: {path}", file=sys.stderr)
    return data

def bench(func, arr, **kwargs):
    s = time.perf_counter(); res = func(arr, **kwargs); e = time.perf_counter()
    return res, e - s

def run_one(path: str, *, check=True, shuffle_copy=False) -> Optional[Dict[str, Any]]:
    data = load_dataset_csv(path)
    if not data: return None
    src = data[:]
    if shuffle_copy: random.Random(0xC0FFEE).shuffle(src)
    n = len(src)
    py_sorted, t_builtin = bench(sorted, src)
    cmp_sorted, t_cmp = bench(quicksort_compare, src)
    oqs_sorted, t_oqs = bench(oqs_bitwise_quicksort, src)
    if check:
        assert cmp_sorted == py_sorted, f"Mismatch (comparison quicksort) on {path}"
        assert oqs_sorted == py_sorted, f"Mismatch (OQS bitwise) on {path}"
    return {"file": os.path.basename(path), "n": n,
            "builtin_s": t_builtin, "compare_qs_s": t_cmp, "oqs_s": t_oqs}

def _finite(vals): return [v for v in vals if v is not None and math.isfinite(v)]

# Print
def print_perfile(rows):
    if not rows: return
    print("\n================ Per-file Results ================")
    print(f"{'File':18} {'n':>10} {'Builtin(s)':>12} {'CompareQS(s)':>14} {'OQS(s)':>10} {'Cmp/OQS':>9} {'Built/OQS':>10}")
    for r in rows:
        cmp_oqs = (r['compare_qs_s']/r['oqs_s']) if r['oqs_s']>0 else float('inf')
        built_oqs = (r['builtin_s']/r['oqs_s']) if r['oqs_s']>0 else float('inf')
        print(f"{r['file']:18} {r['n']:10d} {r['builtin_s']:12.4f} {r['compare_qs_s']:14.4f} {r['oqs_s']:10.4f} {cmp_oqs:9.2f} {built_oqs:10.2f}")

def print_group_stats(rows, label: str):
    if not rows: return
    print(f"\n===== Size {label} — Averages (± std) over {len(rows)} runs =====")
    print(f"{'Metric':18} {'Mean(s)':>12} {'Std(s)':>12}")
    for k in ["builtin_s","compare_qs_s","oqs_s"]:
        vals = _finite([r[k] for r in rows])
        m = mean(vals) if vals else float('nan')
        sd = stdev(vals) if len(vals)>=2 else 0.0 if vals else float('nan')
        print(f"{k:18} {m:12.4f} {sd:12.4f}")
    cmp_vals = _finite([r["compare_qs_s"] for r in rows]); oqs_vals = _finite([r["oqs_s"] for r in rows])
    if cmp_vals and oqs_vals and mean(oqs_vals)>0:
        print(f"{'Speedup Cmp/OQS':18} {mean(cmp_vals)/mean(oqs_vals):12.4f} {'(ratio of means)':>12}")

# Main
if __name__ == "__main__":
    BASE_DIR = os.path.dirname(__file__) if "__file__" in globals() else os.getcwd()
    print(f"Looking for datasets in: {BASE_DIR}")

    sizes = [100, 10_000, 1_000_000]
    variants = [1,2,3,4,5]
    CHECK = {100: True, 10_000: True, 1_000_000: False}
    SHUFFLE_FOR_TIMING = False

    for n in sizes:
        group = []
        for i in variants:
            path = os.path.join(BASE_DIR, f"{n}_dataset_{i}.csv")
            r = run_one(path, check=CHECK[n], shuffle_copy=SHUFFLE_FOR_TIMING)
            if r: group.append(r)
        print_perfile(group)
        print_group_stats(group, str(n))