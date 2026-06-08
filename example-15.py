import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


DT_VALUES = np.arange(0.01, 0.201, 0.01)
T_MAX = 300.0
TARGET_TEMPERATURE = 1.0
DIFFUSION_COEFFICIENT = 1.0
INITIAL_V = 1.0
MICRO_SUBSTEPS = 10
MACRO_SUBSTEPS = 1
NUM_PATHS = 200
DIRECT_SAMPLES = 200_000
SEED = 2026
OUTPUT_PATH = "example-15.png"


def estimate_temperature_direct_brownian(dt, diffusion_coefficient, substeps, rng):
    internal_dt = dt / substeps

    noise = rng.normal(0.0, 1.0, size=DIRECT_SAMPLES)
    dx = np.sqrt(2.0 * diffusion_coefficient * internal_dt) * noise

    # Direct Brownian motion has no real velocity. This is the velocity one
    # would infer from the displacement across one resolved grid interval.
    inferred_velocity = dx / internal_dt
    return np.mean(inferred_velocity**2)


def estimate_temperature_photo_langevin(dt, t_max, temperature, substeps, num_paths, rng):
    internal_dt = dt / substeps
    num_steps = int(t_max / internal_dt)
    burn_in = num_steps // 2

    v = np.full(num_paths, INITIAL_V)
    sum_v2 = 0.0
    count = 0
    noise_scale = np.sqrt(6.0 * temperature / internal_dt)

    for step in range(num_steps):
        eta = noise_scale * (2.0 * rng.random(num_paths) - 1.0)
        v += internal_dt * (-v + eta)

        if step >= burn_in:
            sum_v2 += np.sum(v**2)
            count += num_paths

    return sum_v2 / count


def run_resolution_case(substeps, seed_offset):
    direct_temperatures = []
    langevin_temperatures = []

    for index, dt in enumerate(DT_VALUES):
        direct_rng = np.random.default_rng(SEED + seed_offset + 2 * index)
        langevin_rng = np.random.default_rng(SEED + seed_offset + 2 * index + 1)

        direct_temperatures.append(
            estimate_temperature_direct_brownian(
                dt=dt,
                diffusion_coefficient=DIFFUSION_COEFFICIENT,
                substeps=substeps,
                rng=direct_rng,
            )
        )
        langevin_temperatures.append(
            estimate_temperature_photo_langevin(
                dt=dt,
                t_max=T_MAX,
                temperature=TARGET_TEMPERATURE,
                substeps=substeps,
                num_paths=NUM_PATHS,
                rng=langevin_rng,
            )
        )

    return np.array(direct_temperatures), np.array(langevin_temperatures)


def plot_temperature_comparison(
    microscopic_direct,
    microscopic_langevin,
    macroscopic_direct,
    macroscopic_langevin,
):
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), constrained_layout=True)

    panels = [
        (
            axes[0, 0],
            microscopic_direct,
            f"Microscopic grid / direct Brownian ({MICRO_SUBSTEPS} substeps)",
        ),
        (
            axes[0, 1],
            microscopic_langevin,
            f"Microscopic grid / photo-style Langevin ({MICRO_SUBSTEPS} substeps)",
        ),
        (
            axes[1, 0],
            macroscopic_direct,
            f"Macroscopic grid / direct Brownian ({MACRO_SUBSTEPS} step)",
        ),
        (
            axes[1, 1],
            macroscopic_langevin,
            f"Macroscopic grid / photo-style Langevin ({MACRO_SUBSTEPS} step)",
        ),
    ]

    for ax, temperatures, title in panels:
        ax.plot(DT_VALUES, temperatures, marker="o", linewidth=1.5)
        ax.axhline(
            TARGET_TEMPERATURE,
            color="tab:red",
            linestyle="--",
            linewidth=1.2,
            label="target T = 1",
        )
        ax.set_title(title)
        ax.set_xlabel("outer grid interval dt")
        ax.set_ylabel("estimated T from <v^2>")
        ax.grid(alpha=0.25)
        ax.legend()

    fig.savefig(OUTPUT_PATH, dpi=200)


def print_summary(name, values):
    print(f"{name}")
    print(f"  T_est(dt=0.01) = {values[0]:.6f}")
    print(f"  T_est(dt=0.20) = {values[-1]:.6f}")
    print(f"  average over dt = {np.mean(values):.6f}")


def main():
    microscopic_direct, microscopic_langevin = run_resolution_case(
        substeps=MICRO_SUBSTEPS,
        seed_offset=0,
    )
    macroscopic_direct, macroscopic_langevin = run_resolution_case(
        substeps=MACRO_SUBSTEPS,
        seed_offset=1000,
    )

    plot_temperature_comparison(
        microscopic_direct,
        microscopic_langevin,
        macroscopic_direct,
        macroscopic_langevin,
    )

    print("Temperature estimation from time-averaged kinetic energy")
    print(f"outer dt range: {DT_VALUES[0]:.2f} to {DT_VALUES[-1]:.2f}")
    print(f"target temperature = {TARGET_TEMPERATURE:.1f}")
    print(f"microscopic grid: {MICRO_SUBSTEPS} internal steps per outer dt")
    print(f"macroscopic grid: {MACRO_SUBSTEPS} internal step per outer dt")
    print_summary("microscopic direct Brownian", microscopic_direct)
    print_summary("microscopic photo-style Langevin", microscopic_langevin)
    print_summary("macroscopic direct Brownian", macroscopic_direct)
    print_summary("macroscopic photo-style Langevin", macroscopic_langevin)
    print(f"Saved graph to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
