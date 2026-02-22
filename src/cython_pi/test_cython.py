import sys
import math
import time
from calc_pi import calc_pi_cython


def main():
    if len(sys.argv) < 2:
        print("Usage: python test_cython.py <n>")
        sys.exit(1)

    n = int(sys.argv[1])

    t0 = time.perf_counter()
    pi_est = calc_pi_cython(n)
    t1 = time.perf_counter()

    elapsed = t1 - t0
    samples_per_sec = n / elapsed if elapsed > 0 else float("inf")
    time_per_sample = elapsed / n

    print(f"n={n}")
    print(f"pi_est={pi_est:.10f}")
    print(f"abs_error={abs(pi_est - math.pi):.10f}")
    print(f"time_seconds={elapsed:.6f}")
    print(f"samples_per_second={samples_per_sec:.2f}")
    print(f"time_per_sample={time_per_sample:.12e}")


if __name__ == "__main__":
    main()
