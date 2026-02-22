import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PI_TRUE = math.pi


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _pick_column(df: pd.DataFrame, candidates, required=True):
    """Return the first matching column name from candidates."""
    for c in candidates:
        if c in df.columns:
            return c
    if required:
        raise ValueError(
            f"Missing expected column. Tried {candidates}. "
            f"Found columns: {list(df.columns)}"
        )
    return None


def fit_power_law(n_vals: np.ndarray, y_vals: np.ndarray):
    """
    Fit y = A * n^(-alpha) using log-log linear regression:
      log(y) = log(A) - alpha * log(n)
    Returns a dict with fit parameters and uncertainties.
    """
    # Keep only positive values
    mask = (n_vals > 0) & (y_vals > 0) & np.isfinite(n_vals) & np.isfinite(y_vals)
    x = np.log(n_vals[mask].astype(float))
    y = np.log(y_vals[mask].astype(float))

    if len(x) < 2:
        raise ValueError("Need at least 2 valid positive points for power-law fit.")

    # Linear fit: y = m x + b
    m, b = np.polyfit(x, y, 1)

    # Predictions and residuals
    yhat = m * x + b
    resid = y - yhat
    npts = len(x)

    # Standard errors (simple OLS formulas)
    if npts > 2:
        s2 = np.sum(resid**2) / (npts - 2)
        xbar = np.mean(x)
        sxx = np.sum((x - xbar) ** 2)

        se_m = np.sqrt(s2 / sxx) if sxx > 0 else np.nan
        se_b = np.sqrt(s2 * (1.0 / npts + xbar**2 / sxx)) if sxx > 0 else np.nan
    else:
        se_m, se_b = np.nan, np.nan

    alpha = -m
    alpha_se = se_m if np.isfinite(se_m) else np.nan
    A = np.exp(b)
    A_se = A * se_b if np.isfinite(se_b) else np.nan  # delta method

    # R^2 in log-space
    ss_res = np.sum((y - yhat) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

    return {
        "alpha": alpha,
        "alpha_se": alpha_se,
        "A": A,
        "A_se": A_se,
        "slope_m": m,
        "intercept_b": b,
        "r2_logspace": r2,
        "x_log": x,
        "y_log": y,
        "yhat_log": yhat,
        "mask": mask,
    }


def main():
    # ---- Paths ----
    raw_csv = os.path.join("results", "section6", "scaling_raw_numba_parallel.csv")
    summary_csv = os.path.join("results", "section6", "scaling_summary_numba_parallel.csv")
    out_dir = os.path.join("results", "section6")
    _ensure_dir(out_dir)

    # ---- Load data ----
    if not os.path.exists(raw_csv):
        raise FileNotFoundError(f"Raw CSV not found: {raw_csv}")
    if not os.path.exists(summary_csv):
        raise FileNotFoundError(f"Summary CSV not found: {summary_csv}")

    raw = pd.read_csv(raw_csv)
    summary = pd.read_csv(summary_csv)

    print("Loaded:")
    print(f"  raw rows    = {len(raw)}")
    print(f"  summary rows= {len(summary)}")
    print()
    print("Summary columns:", list(summary.columns))
    print("\nSummary preview:")
    print(summary.head())

    # ---- Column mapping (supports your naming style) ----
    col_method = _pick_column(summary, ["method"], required=False)
    col_n = _pick_column(summary, ["n"])

    col_pi_mean = _pick_column(summary, ["pi_mean", "mean_pi"], required=False)
    col_pi_std = _pick_column(summary, ["pi_std", "std"])

    # For "Mean Error vs n" use error of the mean if present; fallback to mean abs error across runs
    col_mean_err = _pick_column(
        summary,
        ["abs_error_of_mean", "mean_abs_error", "abs_error_mean", "mean_abs_error_across_runs"],
        required=False,
    )
    if col_mean_err is None:
        raise ValueError(
            "Could not find a mean-error column. Need one of: "
            "['abs_error_of_mean','mean_abs_error','abs_error_mean','mean_abs_error_across_runs']"
        )

    col_runtime_mean = _pick_column(summary, ["runtime_mean_sec", "runtime_mean", "time_seconds_mean"])
    col_runtime_std = _pick_column(summary, ["runtime_std_sec", "runtime_std"], required=False)
    col_time_per_sample = _pick_column(
        summary, ["time_per_sample_mean_sec", "time_per_sample_mean"], required=False
    )
    col_samples_per_sec = _pick_column(
        summary, ["samples_per_sec_mean", "samples_per_sec"], required=False
    )
    col_num_runs = _pick_column(summary, ["num_runs"], required=False)

    # Sort by n
    summary = summary.sort_values(by=col_n).reset_index(drop=True)

    n = summary[col_n].to_numpy(dtype=float)
    mean_err = summary[col_mean_err].to_numpy(dtype=float)
    std_pi = summary[col_pi_std].to_numpy(dtype=float)
    runtime_mean = summary[col_runtime_mean].to_numpy(dtype=float)

    # ---- Plots required by Section 6 ----

    # 1) Mean Error vs n (log-log)
    plt.figure(figsize=(7, 5))
    plt.loglog(n, mean_err, marker="o", label="Mean absolute error")
    plt.xlabel("n (samples)")
    plt.ylabel(r"Mean error $|\bar{\pi}_{est}-\pi|$")
    plt.title("Mean Error vs n (Numba parallel steady-state)")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    mean_err_plot = os.path.join(out_dir, "mean_error_vs_n.png")
    plt.tight_layout()
    plt.savefig(mean_err_plot, dpi=150)
    plt.close()

    # 2) Standard Deviation vs n (log-log)
    plt.figure(figsize=(7, 5))
    plt.loglog(n, std_pi, marker="o", label="Std dev of pi estimates")
    plt.xlabel("n (samples)")
    plt.ylabel(r"Standard deviation $\sigma(\pi_{est})$")
    plt.title("Standard Deviation vs n (Numba parallel steady-state)")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    std_plot = os.path.join(out_dir, "std_vs_n.png")
    plt.tight_layout()
    plt.savefig(std_plot, dpi=150)
    plt.close()

    # 3) Power-law fit on std dev: sigma = A n^{-alpha}
    fit = fit_power_law(n, std_pi)
    alpha = fit["alpha"]
    alpha_se = fit["alpha_se"]
    A = fit["A"]
    A_se = fit["A_se"]
    r2 = fit["r2_logspace"]

    # Fitted curve (in original space)
    n_fit = np.logspace(np.log10(np.min(n)), np.log10(np.max(n)), 200)
    std_fit = A * n_fit ** (-alpha)

    plt.figure(figsize=(7, 5))
    plt.loglog(n, std_pi, marker="o", linestyle="None", label="Observed std dev")
    plt.loglog(
        n_fit,
        std_fit,
        linestyle="-",
        label=rf"Fit: $\sigma = A n^{{-\alpha}}$, $\alpha={alpha:.4f}$"
    )
    plt.xlabel("n (samples)")
    plt.ylabel(r"Standard deviation $\sigma(\pi_{est})$")
    plt.title("Power-law Fit to Standard Deviation Scaling")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    fit_plot = os.path.join(out_dir, "std_fit_powerlaw.png")
    plt.tight_layout()
    plt.savefig(fit_plot, dpi=150)
    plt.close()

    # ---- Quantitative analysis requested in Section 6 ----

    # Scaling exponent comparison to theoretical alpha = 0.5
    alpha_theory = 0.5
    rel_diff_alpha = abs(alpha - alpha_theory) / alpha_theory if alpha_theory != 0 else np.nan

    # Accuracy requirement: 12 digits for pi means absolute error ~ 1e-12
    # Use fitted std ~ A n^{-alpha}, set std ≈ 1e-12  => n ≈ (A/1e-12)^(1/alpha)
    # (This is a rough extrapolation using std as a proxy for error scale)
    target_err = 1e-12
    if alpha > 0:
        n_for_12_digits = (A / target_err) ** (1.0 / alpha)
    else:
        n_for_12_digits = np.nan

    # Memory feasibility for NumPy storing x and y arrays of float64:
    # 2 arrays * n * 8 bytes = 16 n bytes
    mem_bytes = 16.0 * n_for_12_digits if np.isfinite(n_for_12_digits) else np.nan
    mem_gib = mem_bytes / (1024**3) if np.isfinite(mem_bytes) else np.nan
    mem_tib = mem_bytes / (1024**4) if np.isfinite(mem_bytes) else np.nan

    # Single-core time estimate using fastest observed samples/sec from summary
    if col_samples_per_sec is not None:
        sps = summary[col_samples_per_sec].to_numpy(dtype=float)
        sps_fastest = np.nanmax(sps)
        time_sec_1core = n_for_12_digits / sps_fastest if np.isfinite(n_for_12_digits) and sps_fastest > 0 else np.nan
    else:
        sps_fastest = np.nan
        time_sec_1core = np.nan

    # Parallel scaling thought experiment: cores needed to finish in one year
    one_year_sec = 365 * 24 * 3600
    cores_for_one_year = time_sec_1core / one_year_sec if np.isfinite(time_sec_1core) else np.nan

    # Helper time formatting
    def fmt_time(seconds):
        if not np.isfinite(seconds):
            return "nan"
        seconds = float(seconds)
        if seconds < 60:
            return f"{seconds:.3f} s"
        minutes = seconds / 60
        if minutes < 60:
            return f"{minutes:.3f} min"
        hours = minutes / 60
        if hours < 24:
            return f"{hours:.3f} hr"
        days = hours / 24
        if days < 365:
            return f"{days:.3f} days"
        years = days / 365
        return f"{years:.3f} years"

    # ---- Save summary text ----
    summary_txt = os.path.join(out_dir, "section6_analysis_summary.txt")
    with open(summary_txt, "w", encoding="utf-8") as f:
        f.write("SECTION 6 ANALYSIS SUMMARY (Numba parallel steady-state)\n")
        f.write("=" * 60 + "\n\n")

        f.write("Input files\n")
        f.write(f"- Raw CSV:     {raw_csv}\n")
        f.write(f"- Summary CSV: {summary_csv}\n\n")

        f.write("Dataset overview\n")
        f.write(f"- Raw rows: {len(raw)}\n")
        f.write(f"- Summary rows (unique n values): {len(summary)}\n")
        if col_num_runs and col_num_runs in summary.columns:
            f.write(f"- Runs per n (from summary): {list(summary[col_num_runs].astype(int))}\n")
        f.write("\n")

        f.write("Power-law fit to standard deviation\n")
        f.write("Model: sigma(pi_est) = A * n^(-alpha)\n")
        f.write(f"- A = {A:.6e}")
        if np.isfinite(A_se):
            f.write(f" ± {A_se:.6e}")
        f.write("\n")
        f.write(f"- alpha = {alpha:.6f}")
        if np.isfinite(alpha_se):
            f.write(f" ± {alpha_se:.6f}")
        f.write("\n")
        f.write(f"- Theoretical alpha = {alpha_theory:.6f}\n")
        f.write(f"- Relative difference from theory = {rel_diff_alpha:.6%}\n")
        f.write(f"- Log-space R^2 = {r2:.6f}\n\n")

        f.write("Accuracy extrapolation (rough)\n")
        f.write(f"- Target absolute error scale: {target_err:.1e}\n")
        f.write(f"- Estimated n needed for ~12 digits: {n_for_12_digits:.6e}\n\n")

        f.write("Memory feasibility (NumPy x,y float64 arrays)\n")
        f.write("- Assumption: 2 arrays (x and y), float64 => 16 bytes/sample total\n")
        f.write(f"- Estimated memory: {mem_bytes:.6e} bytes\n")
        f.write(f"- Estimated memory: {mem_gib:.6e} GiB\n")
        f.write(f"- Estimated memory: {mem_tib:.6e} TiB\n\n")

        f.write("Runtime extrapolation using fastest observed samples/sec\n")
        f.write(f"- Fastest observed samples/sec: {sps_fastest:.6e}\n")
        f.write(f"- Single-core estimated runtime: {time_sec_1core:.6e} s ({fmt_time(time_sec_1core)})\n")
        f.write(f"- Cores needed to finish in 1 year (ideal scaling): {cores_for_one_year:.6e}\n\n")

        f.write("Generated plots\n")
        f.write(f"- {mean_err_plot}\n")
        f.write(f"- {std_plot}\n")
        f.write(f"- {fit_plot}\n")

    # ---- Console summary ----
    print("\nDone.")
    print(f"Saved plots:")
    print(f"  {mean_err_plot}")
    print(f"  {std_plot}")
    print(f"  {fit_plot}")
    print(f"Saved text summary:")
    print(f"  {summary_txt}")

    print("\nKey fit results:")
    print(f"  alpha = {alpha:.6f}" + (f" ± {alpha_se:.6f}" if np.isfinite(alpha_se) else ""))
    print(f"  A     = {A:.6e}" + (f" ± {A_se:.6e}" if np.isfinite(A_se) else ""))
    print(f"  R^2 (log-space) = {r2:.6f}")


if __name__ == "__main__":
    main()