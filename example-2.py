from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def find_data_path():
    candidates = [
        Path("mers.txt"),
        Path("/Users/js/Downloads/mers.txt"),
    ]

    for path in candidates:
        if path.exists():
            return path

    raise FileNotFoundError("Could not find mers.txt")


def load_mers_data(path):
    data = np.loadtxt(path)
    t = data[:, 0]
    n_real = data[:, 1]
    return t, n_real


def infection_rate(n, r, k):
    return r * n * (1.0 - n / k)


def runge_kutta_2(t, n0, r, k):
    """Second-order Runge-Kutta method using the midpoint form."""
    n = np.zeros_like(t, dtype=float)
    n[0] = n0

    for i in range(len(t) - 1):
        dt = t[i + 1] - t[i]
        k1 = infection_rate(n[i], r, k)
        k2 = infection_rate(n[i] + 0.5 * dt * k1, r, k)
        n[i + 1] = n[i] + dt * k2

    return n


def main():
    data_path = find_data_path()
    t, n_real = load_mers_data(data_path)

    r = 0.25
    k = 186.0
    t0 = t[0]
    n0 = n_real[0]

    n_pred = runge_kutta_2(t, n0, r, k)
    error = n_real - n_pred
    mae = np.mean(np.abs(error))
    rmse = np.sqrt(np.mean(error**2))

    print("MERS confirmed cases: real data vs RK2 logistic prediction")
    print(f"data path = {data_path}")
    print(f"r = {r}, K = {k:.0f}, initial N({t0:.0f}) = {n0:.0f}")
    print("-" * 72)
    print(f"{'day':>6} {'real N':>10} {'predicted N':>14} {'real - pred':>14}")
    print("-" * 72)

    for day, real, pred, diff in zip(t, n_real, n_pred, error):
        print(f"{day:6.0f} {real:10.0f} {pred:14.3f} {diff:14.3f}")

    print("-" * 72)
    print(f"MAE  = {mae:.3f}")
    print(f"RMSE = {rmse:.3f}")
    print(f"final real - predicted = {error[-1]:.3f}")

    fig, (ax_cases, ax_error) = plt.subplots(
        2,
        1,
        figsize=(10, 8),
        sharex=True,
        gridspec_kw={"height_ratios": [2, 1]},
    )

    ax_cases.plot(
        t,
        n_real,
        marker="o",
        markersize=4,
        linewidth=1.5,
        label="real data",
    )
    ax_cases.plot(
        t,
        n_pred,
        marker="s",
        markersize=3,
        linewidth=1.5,
        label=f"RK2 logistic prediction (r={r}, K={k:.0f})",
    )
    ax_cases.axhline(
        k,
        color="gray",
        linestyle="--",
        linewidth=1.0,
        label=f"K = {k:.0f}",
    )
    ax_cases.set_title("MERS Confirmed Cases: Real Data vs RK2 Logistic Prediction")
    ax_cases.set_ylabel("confirmed cases N")
    ax_cases.grid(True, alpha=0.3)
    ax_cases.legend()

    ax_error.axhline(0.0, color="gray", linestyle="--", linewidth=1.0)
    ax_error.plot(
        t,
        error,
        marker="o",
        markersize=3,
        linewidth=1.2,
        color="tab:red",
        label="real - predicted",
    )
    ax_error.set_xlabel("time t (day)")
    ax_error.set_ylabel("error")
    ax_error.grid(True, alpha=0.3)
    ax_error.legend()

    fig.tight_layout()
    fig.savefig("example-2.png", dpi=200)
    print("\nSaved graph to example-2.png")


if __name__ == "__main__":
    main()
