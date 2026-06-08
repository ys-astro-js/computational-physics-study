import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


DT = 0.01
T_MAX_TRAJECTORY = 300.0
DT_VALUES = np.arange(0.01, 0.201, 0.01)
T_MAX_TEMPERATURE = 300.0
TEMPERATURE = 1.0
INITIAL_X = 0.0
INITIAL_V = 1.0
NUM_PATHS = 600
SEED = 2026
OUTPUT_PATH = "example-16.png"


def random_impulse(rng, dt, size=None):
    return np.sqrt(6.0 * TEMPERATURE * dt) * (2.0 * rng.random(size) - 1.0)


def euler_step(x, v, dt, impulse):
    x_next = x + dt * v
    v_next = v + dt * (-v) + impulse
    return x_next, v_next


def rkhg_step(x, v, dt, impulse):
    x_predict = x + dt * v
    v_predict = v + dt * (-v) + impulse

    x_next = x + 0.5 * dt * (v + v_predict)
    v_next = v + 0.5 * dt * ((-v) + (-v_predict)) + impulse

    return x_next, v_next


def simulate_one_path(dt, t_max, seed=None):
    rng = np.random.default_rng(seed)
    time = np.arange(0.0, t_max + dt, dt)

    euler_x = np.empty(len(time))
    euler_v = np.empty(len(time))
    rkhg_x = np.empty(len(time))
    rkhg_v = np.empty(len(time))

    euler_x[0] = rkhg_x[0] = INITIAL_X
    euler_v[0] = rkhg_v[0] = INITIAL_V

    for step in range(len(time) - 1):
        impulse = random_impulse(rng, dt)
        euler_x[step + 1], euler_v[step + 1] = euler_step(
            euler_x[step],
            euler_v[step],
            dt,
            impulse,
        )
        rkhg_x[step + 1], rkhg_v[step + 1] = rkhg_step(
            rkhg_x[step],
            rkhg_v[step],
            dt,
            impulse,
        )

    return time, euler_x, euler_v, rkhg_x, rkhg_v


def estimate_temperature_vs_dt(dt_values, t_max, num_paths, seed=None):
    rng = np.random.default_rng(seed)
    euler_temperature = np.empty(len(dt_values))
    rkhg_temperature = np.empty(len(dt_values))

    for index, dt in enumerate(dt_values):
        num_steps = int(t_max / dt)
        burn_in = num_steps // 2

        euler_x = np.full(num_paths, INITIAL_X)
        euler_v = np.full(num_paths, INITIAL_V)
        rkhg_x = np.full(num_paths, INITIAL_X)
        rkhg_v = np.full(num_paths, INITIAL_V)

        euler_v2_sum = 0.0
        rkhg_v2_sum = 0.0
        count = 0

        for step in range(num_steps):
            impulse = random_impulse(rng, dt, size=num_paths)

            euler_x, euler_v = euler_step(euler_x, euler_v, dt, impulse)
            rkhg_x, rkhg_v = rkhg_step(rkhg_x, rkhg_v, dt, impulse)

            if step >= burn_in:
                euler_v2_sum += np.sum(euler_v**2)
                rkhg_v2_sum += np.sum(rkhg_v**2)
                count += num_paths

        euler_temperature[index] = euler_v2_sum / count
        rkhg_temperature[index] = rkhg_v2_sum / count

    return euler_temperature, rkhg_temperature


def euler_stationary_temperature(dt_values):
    return TEMPERATURE / (1.0 - 0.5 * dt_values)


def rkhg_stationary_temperature(dt_values):
    a = 1.0 - dt_values + 0.5 * dt_values**2
    b = 1.0 - 0.5 * dt_values
    return 2.0 * TEMPERATURE * dt_values * b**2 / (1.0 - a**2)


def continuous_theory_temperature(dt_values):
    return np.full_like(dt_values, TEMPERATURE)


def plot_comparison(
    time,
    euler_x,
    euler_v,
    rkhg_x,
    rkhg_v,
    euler_temperature,
    rkhg_temperature,
):
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), constrained_layout=True)
    continuous_theory = continuous_theory_temperature(DT_VALUES)
    euler_discrete_theory = euler_stationary_temperature(DT_VALUES)
    rkhg_discrete_theory = rkhg_stationary_temperature(DT_VALUES)

    ax_x = axes[0, 0]
    ax_x.plot(time, euler_x, label="Euler", linewidth=1.0)
    ax_x.plot(time, rkhg_x, label="RKHG", linewidth=1.0)
    ax_x.set_title("One trajectory: x(t)")
    ax_x.set_xlabel("t")
    ax_x.set_ylabel("x")
    ax_x.grid(alpha=0.25)
    ax_x.legend()

    ax_v = axes[0, 1]
    ax_v.plot(time, euler_v, label="Euler", linewidth=0.8)
    ax_v.plot(time, rkhg_v, label="RKHG", linewidth=0.8, alpha=0.85)
    ax_v.set_title("One trajectory: v(t)")
    ax_v.set_xlabel("t")
    ax_v.set_ylabel("v")
    ax_v.grid(alpha=0.25)
    ax_v.legend()

    ax_temp = axes[1, 0]
    ax_temp.plot(DT_VALUES, euler_temperature, "o-", label="Euler simulation")
    ax_temp.plot(DT_VALUES, rkhg_temperature, "o-", label="RKHG simulation")
    ax_temp.plot(
        DT_VALUES,
        continuous_theory,
        color="black",
        linestyle=":",
        linewidth=1.5,
        label="continuous theory T = 1",
    )
    ax_temp.plot(
        DT_VALUES,
        euler_discrete_theory,
        "--",
        label="Euler discrete theory",
    )
    ax_temp.plot(
        DT_VALUES,
        rkhg_discrete_theory,
        "--",
        label="RKHG discrete theory",
    )
    ax_temp.set_title("Estimated temperature vs theory")
    ax_temp.set_xlabel("dt")
    ax_temp.set_ylabel("estimated T")
    ax_temp.grid(alpha=0.25)
    ax_temp.legend(fontsize=8)

    ax_error = axes[1, 1]
    ax_error.axhline(0.0, color="black", linestyle=":", linewidth=1.2)
    ax_error.plot(
        DT_VALUES,
        euler_temperature - continuous_theory,
        "o-",
        label="Euler simulation - theory",
    )
    ax_error.plot(
        DT_VALUES,
        rkhg_temperature - continuous_theory,
        "o-",
        label="RKHG simulation - theory",
    )
    ax_error.plot(
        DT_VALUES,
        euler_discrete_theory - continuous_theory,
        "--",
        label="Euler discrete bias",
    )
    ax_error.plot(
        DT_VALUES,
        rkhg_discrete_theory - continuous_theory,
        "--",
        label="RKHG discrete bias",
    )
    ax_error.set_title("Difference from continuous theory")
    ax_error.set_xlabel("dt")
    ax_error.set_ylabel("estimated T - theoretical T")
    ax_error.grid(alpha=0.25)
    ax_error.legend(fontsize=8)

    fig.savefig(OUTPUT_PATH, dpi=200)


def main():
    time, euler_x, euler_v, rkhg_x, rkhg_v = simulate_one_path(
        dt=DT,
        t_max=T_MAX_TRAJECTORY,
        seed=SEED,
    )
    euler_temperature, rkhg_temperature = estimate_temperature_vs_dt(
        dt_values=DT_VALUES,
        t_max=T_MAX_TEMPERATURE,
        num_paths=NUM_PATHS,
        seed=SEED + 1,
    )

    plot_comparison(
        time,
        euler_x,
        euler_v,
        rkhg_x,
        rkhg_v,
        euler_temperature,
        rkhg_temperature,
    )

    print("Photo-style Langevin equation: Euler vs RKHG")
    print("Equation: dx/dt = v, dv/dt = -v + eta(t)")
    print(f"trajectory dt = {DT}, trajectory T_max = {T_MAX_TRAJECTORY}")
    print(f"temperature dt range = {DT_VALUES[0]:.2f} to {DT_VALUES[-1]:.2f}")
    print(f"target temperature = {TEMPERATURE:.1f}")
    print(f"Euler T_est(dt=0.01) = {euler_temperature[0]:.6f}")
    print(f"Euler T_est(dt=0.20) = {euler_temperature[-1]:.6f}")
    print(f"Euler error(dt=0.20) = {euler_temperature[-1] - TEMPERATURE:.6f}")
    print(f"RKHG T_est(dt=0.01) = {rkhg_temperature[0]:.6f}")
    print(f"RKHG T_est(dt=0.20) = {rkhg_temperature[-1]:.6f}")
    print(f"RKHG error(dt=0.20) = {rkhg_temperature[-1] - TEMPERATURE:.6f}")
    print(f"Saved graph to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
