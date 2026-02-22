import math
import os
import csv
import time
import statistics
import sys

# Make sure we can import from src/
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.join(REPO_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

# Import your actual Numba functions with aliases that stay consistent in this script
import pi_numba as pn

# Resolve function names robustly (works with either naming style)
if hasattr(pn, "calc_pi_numba_parallel"):
    calc_pi_parallel = pn.calc_pi_numba_parallel
elif hasattr(pn, "calc_pi_parallel"):
    calc_pi_parallel = pn.calc_pi_parallel
else:
    raise ImportError("Could not find parallel Numba function in src/pi_numba.py")

if hasattr(pn, "calc_pi_numba_serial"):
    calc_pi_serial = pn.calc_pi_numba_serial
elif hasattr(pn, "calc_pi_numba"):
    calc_pi_serial = pn.calc_pi_numba
elif hasattr(pn, "calc_pi_serial"):
    calc_pi_serial = pn.calc_pi_serial
else:
    raise ImportError("Could not find serial Numba function in src/pi_numba.py")


PI_TRUE = math.pi

# Full assignment run (can reduce temporarily for quick testing)
N_VALUES = [10**3, 10**4, 10**5, 10**6, 10**7, 10**8, 10**9]
N_RUNS = 10

RAW_CSV = os.path.join(REPO_ROOT, "results", "section6", "scaling_raw_numba_parallel.csv")
SUMMARY_CSV = os.path.join(REPO_ROOT, "results", "section6", "scaling_summary_numba_parallel.csv")


def timed_run(func, n):
    t0 = time.perf_counter()
    pi_est = func(n)
    t1 = time.perf_counter()
    dt = t1 - t0
    abs_err = abs(pi_est - PI_TRUE)
    samples_per_sec = n / dt if dt > 0 else float("inf")
    time_per_sample = dt / n
    return pi_est, abs_err, dt, time_per_sample, samples_per_sec


def main():
    os.makedirs(os.path.dirname(RAW_CSV), exist_ok=True)

    print("Warming up Numba JIT (serial + parallel)...")
    _ = calc_pi_serial(1000)
    _ = calc_pi_parallel(1000)
    print("Warm-up complete.\n")

    raw_rows = []
    summary_rows = []

    for n in N_VALUES:
        print(f"Collecting n={n} ...")

        pi_vals = []
        err_vals = []
        runtimes = []
        tps_vals = []
        sps_vals = []

        for run_id in range(1, N_RUNS + 1):
            pi_est, abs_err, runtime, time_per_sample, samples_per_sec = timed_run(calc_pi_parallel, n)

            raw_rows.append({
                "method": "numba_parallel_steady",
                "n": n,
                "run_id": run_id,
                "pi_estimate": pi_est,
                "abs_error": abs_err,
                "runtime_sec": runtime,
                "time_per_sample_sec": time_per_sample,
                "samples_per_sec": samples_per_sec,
            })

            pi_vals.append(pi_est)
            err_vals.append(abs_err)
            runtimes.append(runtime)
            tps_vals.append(time_per_sample)
            sps_vals.append(samples_per_sec)

            print(
                f"  run {run_id:02d}: pi={pi_est:.12f}, "
                f"err={abs_err:.3e}, t={runtime:.4f}s, sps={samples_per_sec:.3e}"
            )

        pi_mean = statistics.mean(pi_vals)
        pi_std = statistics.stdev(pi_vals) if len(pi_vals) > 1 else 0.0
        abs_err_mean = abs(pi_mean - PI_TRUE)
        abs_err_run_mean = statistics.mean(err_vals)
        runtime_mean = statistics.mean(runtimes)
        runtime_std = statistics.stdev(runtimes) if len(runtimes) > 1 else 0.0
        time_per_sample_mean = statistics.mean(tps_vals)
        samples_per_sec_mean = statistics.mean(sps_vals)

        summary_rows.append({
            "method": "numba_parallel_steady",
            "n": n,
            "num_runs": N_RUNS,
            "pi_mean": pi_mean,
            "pi_std": pi_std,
            "abs_error_of_mean": abs_err_mean,
            "mean_abs_error_across_runs": abs_err_run_mean,
            "runtime_mean_sec": runtime_mean,
            "runtime_std_sec": runtime_std,
            "time_per_sample_mean_sec": time_per_sample_mean,
            "samples_per_sec_mean": samples_per_sec_mean,
        })

        print(
            f"Summary n={n}: mean(pi)={pi_mean:.12f}, std={pi_std:.3e}, "
            f"|mean-pi|={abs_err_mean:.3e}, runtime_mean={runtime_mean:.4f}s\n"
        )

    # Write raw CSV
    with open(RAW_CSV, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "method",
                "n",
                "run_id",
                "pi_estimate",
                "abs_error",
                "runtime_sec",
                "time_per_sample_sec",
                "samples_per_sec",
            ],
        )
        writer.writeheader()
        writer.writerows(raw_rows)

    # Write summary CSV
    with open(SUMMARY_CSV, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "method",
                "n",
                "num_runs",
                "pi_mean",
                "pi_std",
                "abs_error_of_mean",
                "mean_abs_error_across_runs",
                "runtime_mean_sec",
                "runtime_std_sec",
                "time_per_sample_mean_sec",
                "samples_per_sec_mean",
            ],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    print("Done.")
    print(f"Raw data saved to: {RAW_CSV}")
    print(f"Summary data saved to: {SUMMARY_CSV}")


if __name__ == "__main__":
    main()
