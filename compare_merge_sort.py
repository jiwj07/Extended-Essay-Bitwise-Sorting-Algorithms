# compare_merge_sorts_parallel_avg.py
from typing import Iterable, List, Optional, Dict, Any
import csv, sys, time, random, math, multiprocessing
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import shared_memory
from statistics import mean, stdev

try:
    import numpy as np
except ImportError:
    print("[ERROR] NumPy is required.", file=sys.stderr); raise

def merge_sort(data: Iterable[int]) -> List[int]:
    a = list(data); n = len(a)
    if n <= 1: return a
    buf = [0]*n
    def _merge(lo, mid, hi):
        i, j, k = lo, mid, lo
        while i < mid and j < hi:
            if a[i] <= a[j]: buf[k] = a[i]; i += 1
            else: buf[k] = a[j]; j += 1
            k += 1
        while i < mid: buf[k] = a[i]; i += 1; k += 1
        while j < hi: buf[k] = a[j]; j += 1; k += 1
        a[lo:hi] = buf[lo:hi]
    def _sort(lo, hi):
        if hi - lo <= 1: return
        mid = lo + ((hi - lo) >> 1)
        _sort(lo, mid); _sort(mid, hi); _merge(lo, mid, hi)
    _sort(0, n); return a

def _is_power_of_two(x: int) -> bool: return x > 0 and (x & (x - 1)) == 0
def _next_power_of_two(x: int) -> int:
    if x <= 1: return 1
    x -= 1; x |= x>>1; x |= x>>2; x |= x>>4; x |= x>>8; x |= x>>16; x |= x>>32
    return x + 1

def bitonic_sort_inplace(a: List[int]) -> None:
    n = len(a)
    if not _is_power_of_two(n): raise ValueError("len must be power of two")
    k = 2
    while k <= n:
        j = k >> 1
        while j > 0:
            for i in range(n):
                ixj = i ^ j
                if ixj > i:
                    asc = (i & k) == 0
                    if (asc and a[i] > a[ixj]) or ((not asc) and a[i] < a[ixj]):
                        a[i], a[ixj] = a[ixj], a[i]
            j >>= 1
        k <<= 1

def bitonic_sorted_serial(data: Iterable[int]) -> List[int]:
    a = list(data); n = len(a)
    if n <= 1: return a
    target = _next_power_of_two(n)
    if target != n:
        sentinel = max(a) + 1
        a.extend([sentinel]*(target-n))
        bitonic_sort_inplace(a)
        return a[:n]
    bitonic_sort_inplace(a); return a

def _require_int64_range(a: List[int]) -> None:
    if not a: return
    mn, mx = min(a), max(a)
    if mn < np.iinfo(np.int64).min or mx > np.iinfo(np.int64).max - 1:
        raise ValueError("int64 required")

def _bitonic_compare_exchange_range(shm_name: str, n: int, k: int, j: int, start: int, end: int):
    shm = shared_memory.SharedMemory(name=shm_name)
    try:
        arr = np.ndarray((n,), dtype=np.int64, buffer=shm.buf)
        for i in range(start, end):
            ixj = i ^ j
            if ixj > i:
                asc = (i & k) == 0
                ai = arr[i]; aj = arr[ixj]
                if (asc and ai > aj) or ((not asc) and ai < aj):
                    arr[i], arr[ixj] = aj, ai
    finally:
        shm.close()

def bitonic_sorted_parallel(data: Iterable[int], *, max_workers: Optional[int] = None, chunk_size: Optional[int] = None) -> List[int]:
    a = list(data); n = len(a)
    if n <= 1: return a
    _require_int64_range(a)
    target = _next_power_of_two(n); need_trim = target != n
    if need_trim:
        sentinel = np.int64(max(a) + 1)
        a.extend([int(sentinel)]*(target-n))
    arr_np = np.array(a, dtype=np.int64, copy=True)
    shm = shared_memory.SharedMemory(create=True, size=arr_np.nbytes)
    try:
        shm_arr = np.ndarray(arr_np.shape, dtype=arr_np.dtype, buffer=shm.buf)
        shm_arr[:] = arr_np
        size = shm_arr.shape[0]
        if max_workers is None: max_workers = max(1, multiprocessing.cpu_count()-1)
        if chunk_size is None: chunk_size = 32_768
        k = 2
        while k <= size:
            j = k >> 1
            while j > 0:
                with ProcessPoolExecutor(max_workers=max_workers) as ex:
                    futures = []
                    for start in range(0, size, chunk_size):
                        end = min(start + chunk_size, size)
                        futures.append(ex.submit(_bitonic_compare_exchange_range, shm.name, size, k, j, start, end))
                    for f in as_completed(futures): _ = f.result()
                j >>= 1
            k <<= 1
        result = np.ndarray(arr_np.shape, dtype=arr_np.dtype, buffer=shm.buf).copy().tolist()
    finally:
        shm.close(); shm.unlink()
    return result[:n] if need_trim else result

def load_dataset_csv(path: Path) -> list[int]:
    if not path.exists(): print(f"[SKIP] {path}", file=sys.stderr); return []
    data: list[int] = []
    with path.open(newline="") as f:
        rd = csv.DictReader(f)
        if "value" not in (rd.fieldnames or []): print(f"[ERROR] {path} missing header 'value'", file=sys.stderr); return []
        for row in rd:
            try: data.append(int(row["value"]))
            except: pass
    if not data: print(f"[SKIP] No ints: {path}", file=sys.stderr)
    return data

def bench(func, arr, **kwargs):
    s = time.perf_counter(); res = func(arr, **kwargs); e = time.perf_counter()
    return res, e - s

def run_one(path: Path, *, check=True, shuffle_copy=False, parallel_workers: Optional[int] = None) -> Optional[Dict[str, Any]]:
    data = load_dataset_csv(path)
    if not data: return None
    work = data[:]
    if shuffle_copy: random.shuffle(work)
    n = len(work)
    py_sorted, t_builtin = bench(sorted, work)
    ms_sorted, t_merge = bench(merge_sort, work)
    bt_serial, t_bit_serial = bench(bitonic_sorted_serial, work)
    try:
        bt_parallel, t_bit_par = bench(bitonic_sorted_parallel, work, max_workers=parallel_workers)
    except Exception:
        bt_parallel, t_bit_par = None, float('nan')
    if check:
        assert ms_sorted == py_sorted
        assert bt_serial == py_sorted
        if bt_parallel is not None: assert bt_parallel == py_sorted
    return {"file": path.name, "n": n, "builtin_s": t_builtin, "merge_s": t_merge,
            "bitonic_serial_s": t_bit_serial, "bitonic_parallel_s": t_bit_par}

def _finite(vals): return [v for v in vals if v is not None and math.isfinite(v)]

# Print
def print_perfile(rows):
    if not rows: return
    print("\n================ Per-file Results ================")
    print(f"{'File':18} {'n':>10} {'Builtin(s)':>12} {'Merge(s)':>12} {'Bitonic-Ser(s)':>14} {'Bitonic-Par(s)':>14}")
    for r in rows:
        if not r: continue
        print(f"{r['file']:18} {r['n']:10d} {r['builtin_s']:12.4f} {r['merge_s']:12.4f} {r['bitonic_serial_s']:14.4f} {r['bitonic_parallel_s']:14.4f}")

def summarize_group(rows, label: str):
    if not rows: return
    keys = ["builtin_s","merge_s","bitonic_serial_s","bitonic_parallel_s"]
    print(f"\n===== Size {label} — Averages (± std) over {len(rows)} runs =====")
    print(f"{'Metric':22} {'Mean(s)':>12} {'Std(s)':>12}")
    for k in keys:
        vals = _finite([r[k] for r in rows])
        m = mean(vals) if vals else float('nan')
        sd = stdev(vals) if len(vals) >= 2 else 0.0 if vals else float('nan')
        print(f"{k:22} {m:12.4f} {sd:12.4f}")

# Main
if __name__ == "__main__":
    try:
        base_dir = Path(__file__).resolve().parent
    except NameError:
        base_dir = Path.cwd()
    print(f"Looking for datasets in: {base_dir}")

    sizes = [100, 10_000, 1_000_000]
    variants = [1,2,3,4,5]
    CHECK = {100: True, 10_000: True, 1_000_000: False}
    SHUFFLE_FOR_TIMING = False
    PARALLEL_WORKERS = None

    for n in sizes:
        group_rows = []
        for i in variants:
            path = base_dir / f"{n}_dataset_{i}.csv"
            r = run_one(path, check=CHECK[n], shuffle_copy=SHUFFLE_FOR_TIMING, parallel_workers=PARALLEL_WORKERS)
            if r: group_rows.append(r)
        print_perfile(group_rows)
        summarize_group(group_rows, str(n))