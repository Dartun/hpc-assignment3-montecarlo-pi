import sys
import time
import math
import numpy as np
from numba import njit, prange


@njit
def calc_pi_numba_serial(n: int) -> float:
    """Numba JIT-compiled serial Monte Carlo pi estimator."""
    hits = 0
    for _ in range(n):
        x = np.random.random()
        y = np.random.random()
        if x * x + y * y <= 1.0:
            hits += 1
    return 4.0 * hits / n


@njit(parallel=True)
def calc_pi_numba_parallel(n: int) -> float:
    """Numba JIT-compiled parallel Monte Carlo pi estimator."""
    hits = 0
    for _ in prange(n):
        x = np.random.random()
        y = np.random.random()
        if x * x + y * y <= 1.0:
            hits += 1
    return 4.0 * hits / n


def run_and_time(func, n: int):
    t0 = time.perf_counter()
    pi_est = func(n)
    t1 = time.perf_counter()
    elapsed = t1 - t0
    return pi_est, elapsed


def print_stats(label: str, n: int, pi_est: float, elapsed: float):
    samples_per_sec = n / elapsed if elapsed > 0 else float("inf")
    time_per_sample = elapsed / n
    print(f"[{label}]")
    print(f"n={n}")
    print(f"pi_est={pi_est:.10f}")
    print(f"abs_error={abs(pi_est - math.pi):.10f}")
    print(f"time_seconds={elapsed:.6f}")
    print(f"samples_per_second={samples_per_sec:.2f}")
    print(f"time_per_sample={time_per_sample:.12e}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python src/pi_numba.py <n>")
        sys.exit(1)

    n = int(sys.argv[1])
    if n <= 0:
        raise ValueError("n must be a positive integer")

    # ---- First-call timings (include JIT compilation overhead) ----
    pi_serial_1, t_serial_1 = run_and_time(calc_pi_numba_serial, n)
    pi_parallel_1, t_parallel_1 = run_and_time(calc_pi_numba_parallel, n)

    print("=== FIRST RUN (includes JIT compile overhead) ===")
    print_stats("numba_serial_first", n, pi_serial_1, t_serial_1)
    print()
    print_stats("numba_parallel_first", n, pi_parallel_1, t_parallel_1)
    print()

    # ---- Second-call timings (steady-state execution) ----
    pi_serial_2, t_serial_2 = run_and_time(calc_pi_numba_serial, n)
    pi_parallel_2, t_parallel_2 = run_and_time(calc_pi_numba_parallel, n)

    print("=== SECOND RUN (compiled; steady-state) ===")
    print_stats("numba_serial_second", n, pi_serial_2, t_serial_2)
    print()
    print_stats("numba_parallel_second", n, pi_parallel_2, t_parallel_2)
    print()

    # ---- Speedup based on steady-state times ----
    speedup = t_serial_2 / t_parallel_2 if t_parallel_2 > 0 else float("inf")
    print("=== PARALLEL SPEEDUP (steady-state) ===")
    print(f"speedup_serial_over_parallel={speedup:.4f}")


if __name__ == "__main__":
    main()
