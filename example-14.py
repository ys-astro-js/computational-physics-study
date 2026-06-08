import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


DT = 0.01
T_MAX = 1000.0
DIFFUSION_COEFFICIENT = 1.0
TEMPERATURE = 1.0
INITIAL_X = 0.0
INITIAL_V = 1.0
SEED = 2026
ENSEMBLE_PATHS = 300
SAMPLE_EVERY = 100
OUTPUT_PATH = "example-14.png"


def direct_brownian_path(dt, t_max, diffusion_coefficient, x0, seed=None):
    rng = np.random.default_rng(seed)
    time = np.arange(0.0, t_max + dt, dt)
    num_steps = len(time) - 1

    noise = rng.normal(0.0, 1.0, size=num_steps)
    increments = np.sqrt(2.0 * diffusion_coefficient * dt) * noise

    x = np.empty(len(time))
    x[0] = x0
    x[1:] = x0 + np.cumsum(increments)

    return time, x


def photo_style_langevin_path(dt, t_max, temperature, x0, v0, seed=None):
    rng = np.random.default_rng(seed)
    time = np.arange(0.0, t_max + dt, dt)
    num_steps = len(time) - 1

    x = np.empty(len(time))
    v = np.empty(len(time))
    x[0] = x0
    v[0] = v0

    noise_scale = np.sqrt(6.0 * temperature / dt)
    eta = noise_scale * (2.0 * rng.random(num_steps) - 1.0)

    for step in range(num_steps):
        x[step + 1] = x[step] + dt * v[step]
        v[step + 1] = v[step] + dt * (-v[step] + eta[step])

    return time, x, v


def ensemble_variance_comparison(
    dt,
    t_max,
    diffusion_coefficient,
    temperature,
    num_paths,
    sample_every,
    seed=None,
):
    rng = np.random.default_rng(seed)
    time = np.arange(0.0, t_max + dt, dt)
    num_steps = len(time) - 1
    sample_indices = np.arange(0, num_steps + 1, sample_every)
    sample_time = time[sample_indices]

    direct_x = np.zeros(num_paths)
    langevin_x = np.zeros(num_paths)
    langevin_v = np.full(num_paths, INITIAL_V)

    direct_variance = np.empty(len(sample_indices))
    langevin_variance = np.empty(len(sample_indices))
    direct_variance[0] = np.var(direct_x)
    langevin_variance[0] = np.var(langevin_x)

    noise_scale = np.sqrt(6.0 * temperature / dt)
    sample_position = 1

    for step in range(num_steps):
        direct_x += (
            np.sqrt(2.0 * diffusion_coefficient * dt)
            * rng.normal(0.0, 1.0, size=num_paths)
        )

        eta = noise_scale * (2.0 * rng.random(num_paths) - 1.0)
        langevin_x += dt * langevin_v
        langevin_v += dt * (-langevin_v + eta)

        if (step + 1) % sample_every == 0:
            direct_variance[sample_position] = np.var(direct_x)
            langevin_variance[sample_position] = np.var(langevin_x)
            sample_position += 1

    return sample_time, direct_variance, langevin_variance


def plot_comparison(
    time,
    direct_x,
    langevin_x,
    langevin_v,
    variance_time,
    direct_variance,
    langevin_variance,
):
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), constrained_layout=True)

    ax_direct = axes[0, 0]
    ax_direct.plot(time, direct_x, color="tab:blue", linewidth=1.0)
    ax_direct.set_title("Direct Brownian x(t)")
    ax_direct.set_xlabel("t")
    ax_direct.set_ylabel("x")
    ax_direct.grid(alpha=0.25)

    ax_langevin_x = axes[0, 1]
    ax_langevin_x.plot(time, langevin_x, color="tab:green", linewidth=1.0)
    ax_langevin_x.set_title("Photo-style Langevin x(t)")
    ax_langevin_x.set_xlabel("t")
    ax_langevin_x.set_ylabel("x")
    ax_langevin_x.grid(alpha=0.25)

    ax_langevin_v = axes[1, 0]
    ax_langevin_v.plot(time, langevin_v, color="tab:purple", linewidth=0.7)
    ax_langevin_v.set_title("Photo-style Langevin v(t)")
    ax_langevin_v.set_xlabel("t")
    ax_langevin_v.set_ylabel("v")
    ax_langevin_v.grid(alpha=0.25)

    ax_variance = axes[1, 1]
    ax_variance.plot(
        variance_time,
        direct_variance,
        label="direct Brownian",
        color="tab:blue",
    )
    ax_variance.plot(
        variance_time,
        langevin_variance,
        label="photo-style Langevin",
        color="tab:green",
    )
    ax_variance.plot(
        variance_time,
        2.0 * DIFFUSION_COEFFICIENT * variance_time,
        label="theory 2Dt",
        color="tab:orange",
        linestyle="--",
    )
    ax_variance.set_title("Position variance comparison")
    ax_variance.set_xlabel("t")
    ax_variance.set_ylabel("variance of x")
    ax_variance.grid(alpha=0.25)
    ax_variance.legend()

    fig.savefig(OUTPUT_PATH, dpi=200)


def main():
    time, direct_x = direct_brownian_path(
        dt=DT,
        t_max=T_MAX,
        diffusion_coefficient=DIFFUSION_COEFFICIENT,
        x0=INITIAL_X,
        seed=SEED,
    )
    _, langevin_x, langevin_v = photo_style_langevin_path(
        dt=DT,
        t_max=T_MAX,
        temperature=TEMPERATURE,
        x0=INITIAL_X,
        v0=INITIAL_V,
        seed=SEED,
    )
    variance_time, direct_variance, langevin_variance = ensemble_variance_comparison(
        dt=DT,
        t_max=T_MAX,
        diffusion_coefficient=DIFFUSION_COEFFICIENT,
        temperature=TEMPERATURE,
        num_paths=ENSEMBLE_PATHS,
        sample_every=SAMPLE_EVERY,
        seed=SEED + 1,
    )

    halfway = len(langevin_v) // 2
    average_v2 = np.average(langevin_v[halfway:] ** 2)

    print("Comparison: direct Brownian motion vs photo-style Langevin code")
    print(f"dt = {DT}, T_max = {T_MAX}")
    print(f"direct final x = {direct_x[-1]:.6f}")
    print(f"photo-style Langevin final x = {langevin_x[-1]:.6f}")
    print(f"photo-style average v^2 over second half = {average_v2:.6f}")
    print(f"expected equilibrium v^2 = {TEMPERATURE:.6f}")

    plot_comparison(
        time,
        direct_x,
        langevin_x,
        langevin_v,
        variance_time,
        direct_variance,
        langevin_variance,
    )
    print(f"Saved graph to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
