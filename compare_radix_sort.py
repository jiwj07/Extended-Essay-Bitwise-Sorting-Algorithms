# compare_radix_sorts_avg.py
from typing import Iterable, List, Dict, Any, Optional
import csv, sys, time, random, math
from pathlib import Path
from statistics import mean, stdev

def radix_sort_divmod(data: Iterable[int], *, bits: int = 32, group_bits: int = 8) -> List[int]:
    a = list(data); n = len(a)
    if n <= 1: return a.copy()
    passes = (bits + group_bits - 1) // group_bits
    base = 1 << group_bits
    out = [0]*n
    base_power = 1
    for _ in range(passes):
        count = [0]*base
        for x in a:
            d = (x // base_power) % base
            count[d] += 1
        total = 0
        for i in range(base):
            c = count[i]; count[i] = total; total += c
        for x in a:
            d = (x // base_power) % base
            out[count[d]] = x; count[d] += 1
        a, out = out, a
        base_power *= base
    return a

def frs_sort_bitwise(data: Iterable[int], *, bits: int = 32, group_bits: int = 8) -> List[int]:
    a = list(data); n = len(a)
    if n <= 1: return a.copy()
    passes = (bits + group_bits - 1) // group_bits
    radix = 1 << group_bits
    digit_mask = radix - 1
    out = [0]*n
    for p in range(passes):
        shift = p * group_bits
        count = [0]*radix
        for x in a:
            count[(x >> shift) & digit_mask] += 1
        total = 0
        for i in range(radix):
            c = count[i]; count[i] = total; total += c
        for x in a:
            d = (x >> shift) & digit_mask
            out[count[d]] = x; count[d] += 1
        a, out = out, a
    return a

def load_dataset_csv(path: Path) -> list[int]:
    if not path.exists(): print(f"[SKIP] {path}", file=sys.stderr); return []
    data: list[int] = []
    with path.open(newline="") as f:
        rd = csv.DictReader(f)
        if "value" not in (rd.fieldnames or []): print(f"[ERROR] {path} missing 'value' header", file=sys.stderr); return []
        for row in rd:
            try: data.append(int(row["value"]))
            except: pass
    if not data: print(f"[SKIP] No ints: {path}", file=sys.stderr)
    return data

def bench(func, arr, **kwargs):
    s = time.perf_counter(); res = func(arr, **kwargs); e = time.perf_counter()
    return res, e - s

def run_one(path: Path, *, bits=32, group_bits=8, check=True, shuffle_copy=False) -> Optional[Dict[str, Any]]:
    data = load_dataset_csv(path)
    if not data: return None
    work = data[:]
    if shuffle_copy: random.shuffle(work)
    n = len(work)
    py_sorted, t_builtin = bench(sorted, work)
    div_sorted, t_div = bench(radix_sort_divmod, work, bits=bits, group_bits=group_bits)
    bit_sorted, t_bit = bench(frs_sort_bitwise, work, bits=bits, group_bits=group_bits)
    if check:
        assert div_sorted == py_sorted, f"Mismatch (div/mod) on {path.name}"
        assert bit_sorted == py_sorted, f"Mismatch (fastbit) on {path.name}"
    return {"file": path.name, "n": n, "builtin_s": t_builtin, "divmod_s": t_div, "fastbit_s": t_bit}

def _finite(vals): return [v for v in vals if v is not None and math.isfinite(v)]

# Print
def print_perfile(rows):
    if not rows: return
    print("\n================ Per-file Results ================")
    print(f"{'File':18} {'n':>10} {'Builtin(s)':>12} {'Div/Mod(s)':>12} {'Fastbit(s)':>12} {'Div/Fast':>9} {'Built/Fast':>10}")
    for r in rows:
        div_fast = (r['divmod_s']/r['fastbit_s']) if r['fastbit_s']>0 else float('inf')
        built_fast = (r['builtin_s']/r['fastbit_s']) if r['fastbit_s']>0 else float('inf')
        print(f"{r['file']:18} {r['n']:10d} {r['builtin_s']:12.4f} {r['divmod_s']:12.4f} {r['fastbit_s']:12.4f} {div_fast:9.2f} {built_fast:10.2f}")

def print_group_stats(rows, label: str):
    if not rows: return
    print(f"\n===== Size {label} — Averages (± std) over {len(rows)} runs =====")
    print(f"{'Metric':18} {'Mean(s)':>12} {'Std(s)':>12}")
    for k in ["builtin_s","divmod_s","fastbit_s"]:
        vals = _finite([r[k] for r in rows])
        m = mean(vals) if vals else float('nan')
        sd = stdev(vals) if len(vals)>=2 else 0.0 if vals else float('nan')
        print(f"{k:18} {m:12.4f} {sd:12.4f}")
    div_vals = _finite([r["divmod_s"] for r in rows]); fast_vals = _finite([r["fastbit_s"] for r in rows])
    built_vals = _finite([r["builtin_s"] for r in rows])
    if fast_vals:
        if div_vals: print(f"{'Speedup Div/Fast':18} {mean(div_vals)/mean(fast_vals):12.4f} {'(ratio of means)':>12}")
        if built_vals: print(f"{'Speedup Built/Fast':18} {mean(built_vals)/mean(fast_vals):12.4f} {'(ratio of means)':>12}")

# Main
if __name__ == "__main__":
    try:
        base_dir = Path(__file__).resolve().parent
    except NameError:
        base_dir = Path.cwd()
    print(f"Looking for datasets in: {base_dir}")

    sizes = [100, 10_000, 1_000_000]
    variants = [1,2,3,4,5]
    BITS, GROUP_BITS = 32, 8
    CHECK = {100: True, 10_000: True, 1_000_000: False}
    SHUFFLE_FOR_TIMING = False

    for n in sizes:
        group = []
        for i in variants:
            path = base_dir / f"{n}_dataset_{i}.csv"
            r = run_one(path, bits=BITS, group_bits=GROUP_BITS, check=CHECK[n], shuffle_copy=SHUFFLE_FOR_TIMING)
            if r: group.append(r)
        print_perfile(group)
        print_group_stats(group, str(n))