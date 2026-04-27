import numpy as np
import matplotlib.pyplot as plt


def analytic_velocity(t, g, gamma):
    """Analytic solution for dv/dt = g - gamma * v, v(0) = 0."""
    return (g / gamma) * (1.0 - np.exp(-gamma * t))


def acceleration(v, g, gamma):
    return g - gamma * v


def euler_velocity(t_end, dt, g, gamma):
    t = np.arange(0.0, t_end + dt, dt)
    v = np.zeros_like(t)

    for i in range(len(t) - 1):
        v[i + 1] = v[i] + dt * acceleration(v[i], g, gamma)

    return t, v


def modified_euler_velocity(t_end, dt, g, gamma):
    """Modified Euler method using the predictor-corrector form."""
    t = np.arange(0.0, t_end + dt, dt)
    v = np.zeros_like(t)

    for i in range(len(t) - 1):
        k1 = acceleration(v[i], g, gamma)
        v_predict = v[i] + dt * k1
        k2 = acceleration(v_predict, g, gamma)
        v[i + 1] = v[i] + 0.5 * dt * (k1 + k2)

    return t, v


def runge_kutta_velocity(t_end, dt, g, gamma):
    """Fourth-order Runge-Kutta method."""
    t = np.arange(0.0, t_end + dt, dt)
    v = np.zeros_like(t)

    for i in range(len(t) - 1):
        k1 = acceleration(v[i], g, gamma)
        k2 = acceleration(v[i] + 0.5 * dt * k1, g, gamma)
        k3 = acceleration(v[i] + 0.5 * dt * k2, g, gamma)
        k4 = acceleration(v[i] + dt * k3, g, gamma)
        v[i + 1] = v[i] + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    return t, v


def main():
    g = 9.8
    gamma = 0.5
    t_end = 10.0
    dt = 0.5
    methods = [
        ("Euler", euler_velocity),
        ("Modified Euler", modified_euler_velocity),
        ("Runge-Kutta 4th", runge_kutta_velocity),
    ]

    print("Velocity-time graph and method error for free fall with fluid resistance")
    print(f"g = {g}, gamma = {gamma}, t_end = {t_end}, dt = {dt}")
    print("-" * 60)
    print(f"{'method':>18} {'max abs error':>18} {'final error':>18}")
    print("-" * 60)

    t_exact = np.linspace(0.0, t_end, 1000)
    v_exact_smooth = analytic_velocity(t_exact, g, gamma)

    fig, (ax_velocity, ax_error) = plt.subplots(
        2,
        1,
        figsize=(10, 8),
        sharex=True,
        gridspec_kw={"height_ratios": [2, 1]},
    )

    ax_velocity.plot(
        t_exact,
        v_exact_smooth,
        color="black",
        linewidth=2.0,
        label="analytic solution",
    )

    for method_name, method in methods:
        t, v_numeric = method(t_end, dt, g, gamma)
        v_exact = analytic_velocity(t, g, gamma)
        error = np.abs(v_exact - v_numeric)

        print(f"{method_name:>18} {np.max(error):18.8f} {error[-1]:18.8f}")

        ax_velocity.plot(
            t,
            v_numeric,
            marker="o",
            markersize=3,
            linewidth=1.2,
            label=method_name,
        )
        ax_error.plot(
            t,
            error,
            marker="o",
            markersize=3,
            linewidth=1.2,
            label=method_name,
        )

    terminal_velocity = g / gamma
    ax_velocity.axhline(
        terminal_velocity,
        color="gray",
        linestyle="--",
        linewidth=1.0,
        label=f"terminal velocity = {terminal_velocity:.1f} m/s",
    )

    ax_velocity.set_title(f"Free Fall with Fluid Resistance: v-t Graph (dt={dt})")
    ax_velocity.set_ylabel("velocity v (m/s)")
    ax_velocity.grid(True, alpha=0.3)
    ax_velocity.legend()

    ax_error.set_xlabel("time t (s)")
    ax_error.set_ylabel("absolute error (m/s)")
    ax_error.grid(True, alpha=0.3)
    ax_error.legend()

    fig.tight_layout()
    fig.savefig("example-1A.png", dpi=200)
    print("\nSaved graph to example-1A.png")


if __name__ == "__main__":
    main()
