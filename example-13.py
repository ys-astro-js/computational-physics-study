import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


DIFFUSION_COEFFICIENT = 1.0
DRIFT = 0.0
T_MAX = 10.0
DT = 0.01
NUM_PATHS = 1000
NUM_PATHS_TO_PLOT = 8
SEED = 2026
OUTPUT_PATH = "example-13.png"


def euler_maruyama(drift_function, diffusion_function, x0, t_max, dt, num_paths, seed=None):
    rng = np.random.default_rng(seed)
    time = np.arange(0.0, t_max + dt, dt)
    num_steps = len(time) - 1

    x = np.zeros((num_paths, len(time)))
    x[:, 0] = x0
    noise = rng.normal(0.0, 1.0, size=(num_paths, num_steps))

    for step in range(num_steps):
        t = time[step]
        x_now = x[:, step]
        x[:, step + 1] = (
            x_now
            + drift_function(x_now, t) * dt
            + diffusion_function(x_now, t) * np.sqrt(dt) * noise[:, step]
        )

    return time, x


def brownian_drift(x, t):
    return DRIFT + 0.0 * x


def brownian_diffusion(x, t):
    return np.sqrt(2.0 * DIFFUSION_COEFFICIENT) + 0.0 * x


def plot_result(time, paths, drift, diffusion_coefficient):
    mean = np.mean(paths, axis=0)
    variance = np.var(paths, axis=0)
    theory_mean = drift * time
    theory_variance = 2.0 * diffusion_coefficient * time

    fig, (ax_paths, ax_moments) = plt.subplots(
        1, 2, figsize=(11, 4.5), constrained_layout=True
    )

    for path in paths[:NUM_PATHS_TO_PLOT]:
        ax_paths.plot(time, path, linewidth=1.0, alpha=0.85)

    ax_paths.set_title("Brownian Motion Sample Paths")
    ax_paths.set_xlabel("time")
    ax_paths.set_ylabel("x(t)")
    ax_paths.grid(alpha=0.25)

    ax_moments.plot(time, mean, label="sample mean", color="tab:blue")
    ax_moments.plot(
        time,
        theory_mean,
        label="theory mean",
        color="tab:blue",
        linestyle="--",
    )
    ax_moments.plot(time, variance, label="sample variance", color="tab:orange")
    ax_moments.plot(
        time,
        theory_variance,
        label="theory variance",
        color="tab:orange",
        linestyle="--",
    )
    ax_moments.set_title("Ensemble Statistics")
    ax_moments.set_xlabel("time")
    ax_moments.grid(alpha=0.25)
    ax_moments.legend()

    fig.savefig(OUTPUT_PATH, dpi=200)


def main():
    time, paths = euler_maruyama(
        drift_function=brownian_drift,
        diffusion_function=brownian_diffusion,
        x0=0.0,
        t_max=T_MAX,
        dt=DT,
        num_paths=NUM_PATHS,
        seed=SEED,
    )

    final_mean = np.mean(paths[:, -1])
    final_variance = np.var(paths[:, -1])
    theory_variance = 2.0 * DIFFUSION_COEFFICIENT * T_MAX

    print("Euler-Maruyama simulation of 1D Brownian motion")
    print("SDE: dX = drift * dt + sqrt(2D) * dW")
    print(f"dt = {DT}, paths = {NUM_PATHS}, T = {T_MAX}")
    print(f"final sample mean = {final_mean:.6f}")
    print(f"final sample variance = {final_variance:.6f}")
    print(f"theory final variance = {theory_variance:.6f}")

    plot_result(time, paths, DRIFT, DIFFUSION_COEFFICIENT)
    print(f"Saved graph to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
