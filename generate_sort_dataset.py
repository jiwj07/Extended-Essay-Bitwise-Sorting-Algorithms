import csv
import random
import os

def make_and_save_dataset(n: int, bits: int, seed: int, filename: str):
    """
    Generate a dataset of n random integers and save to a CSV file.

    Parameters
    ----------
    n : int
        Number of integers to generate.
    bits : int
        Bit width (values will be in [0, 2**bits - 1]).
    seed : int
        Seed for reproducibility.
    filename : str
        Path to save the CSV file.
    """
    random.seed(seed)
    data = [random.getrandbits(bits) for _ in range(n)]

    os.makedirs(os.path.dirname(filename) or ".", exist_ok=True)

    with open(filename, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["value"])
        for val in data:
            writer.writerow([val])

    print(f"Generated {n} random {bits}-bit integers -> {filename}")


if __name__ == "__main__":
    BITS = 32
    BASE_SEED = 42

    SIZES = [100, 10_000, 1_000_000]
    NUM_DATASETS = 5

    for n in SIZES:
        for i in range(1, NUM_DATASETS + 1):
            seed = BASE_SEED + i
            filename = f"{n}_dataset_{i}.csv"
            make_and_save_dataset(n, BITS, seed, filename)