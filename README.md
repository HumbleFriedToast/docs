Primality Test Benchmarks

This folder contains `main.py`, which implements four primality-test algorithms and benchmarks them across increasing input sizes. It produces two plots:

- `big_o_vs_measured.png`: experimental timing curves for each algorithm (log-log) and scaled theoretical curves overlayed.

Requirements

Install required packages (using your fish shell):

```fish
python -m pip install -r requirements.txt
```

Usage

Run the benchmark script:

```fish
python main.py
```

The script will print progress and save images into the same folder.

Notes

- The script chooses worst-case inputs (prime numbers at/above each target size) so each algorithm performs its full loop.
- Theoretical curves are scaled to match measured data at the largest n so shapes align for comparison.
