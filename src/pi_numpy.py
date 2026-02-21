import sys
import time
import numpy as np


def calc_pi_numpy(n: int) -> float:
    """Estimate pi using NumPy vectorized arrays."""
    if n <= 0:
        raise ValueError("n must be a positive integer")

    x = np.random.rand(n)
    y = np.random.rand(n)
    hits = np.sum(x * x + y * y <= 1.0)

    return 4.0 * float(hits) / float(n)


def main():
    if len(sys.argv) < 2:
        print("Usage: python src/pi_numpy.py <n>")
        sys.exit(1)

    n = int(sys.argv[1])

    t0 = time.perf_counter()
    pi_est = calc_pi_numpy(n)
    t1 = time.perf_counter()

    elapsed = t1 - t0
    samples_per_sec = n / elapsed if elapsed > 0 else float("inf")
    time_per_sample = elapsed / n

    print(f"n={n}")
    print(f"pi_est={pi_est:.10f}")
    print(f"abs_error={abs(pi_est - 3.141592653589793):.10f}")
    print(f"time_seconds={elapsed:.6f}")
    print(f"samples_per_second={samples_per_sec:.2f}")
    print(f"time_per_sample={time_per_sample:.12e}")


if __name__ == "__main__":
    main()
