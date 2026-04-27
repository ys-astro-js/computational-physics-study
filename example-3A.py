import numpy as np
import matplotlib.pyplot as plt


def analytic_trajectory(x, vx0, vy0, g):
    return (vy0 / vx0) * x - (g / (2.0 * vx0**2)) * x**2


def integrate_for_slope(x, initial_slope, acceleration_x):
    y = np.zeros_like(x)
    slope = initial_slope

    for i in range(len(x) - 1):
        dx = x[i + 1] - x[i]
        y[i + 1] = y[i] + slope * dx + 0.5 * acceleration_x * dx**2
        slope = slope + acceleration_x * dx

    return y


def shooting_method(x, target_y, acceleration_x, guess0, guess1, tolerance=1.0e-12):
    history = []
    slope0 = guess0
    slope1 = guess1
    miss0 = integrate_for_slope(x, slope0, acceleration_x)[-1] - target_y
    miss1 = integrate_for_slope(x, slope1, acceleration_x)[-1] - target_y
    history.append((0, slope0, miss0))
    history.append((1, slope1, miss1))

    for iteration in range(2, 20):
        slope2 = slope1 - miss1 * (slope1 - slope0) / (miss1 - miss0)
        y = integrate_for_slope(x, slope2, acceleration_x)
        miss2 = y[-1] - target_y
        history.append((iteration, slope2, miss2))

        if abs(miss2) < tolerance:
            return slope2, y, history

        slope0, miss0 = slope1, miss1
        slope1, miss1 = slope2, miss2

    return slope2, y, history


def relaxation_method(
    x,
    y_left,
    y_right,
    acceleration_x,
    snapshot_iterations,
    omega=1.95,
    tolerance=1.0e-11,
    max_iterations=10000,
):
    dx = x[1] - x[0]
    y = np.linspace(y_left, y_right, len(x))
    snapshots = {0: y.copy()}

    for iteration in range(1, max_iterations + 1):
        max_change = 0.0

        for i in range(1, len(x) - 1):
            relaxed = 0.5 * (y[i - 1] + y[i + 1] - acceleration_x * dx**2)
            y_new = y[i] + omega * (relaxed - y[i])
            max_change = max(max_change, abs(y_new - y[i]))
            y[i] = y_new

        if iteration in snapshot_iterations:
            snapshots[iteration] = y.copy()

        if max_change < tolerance:
            snapshots[iteration] = y.copy()
            return y, snapshots, iteration, max_change

    snapshots[max_iterations] = y.copy()
    return y, snapshots, max_iterations, max_change


def main():
    g = 9.8
    v0 = 10.0
    theta_deg = 45.0
    theta = np.deg2rad(theta_deg)
    dx_target = 0.05

    vx0 = v0 * np.cos(theta)
    vy0 = v0 * np.sin(theta)
    target_x = v0**2 * np.sin(2.0 * theta) / g
    target_y = 0.0
    acceleration_x = -g / vx0**2

    n_intervals = int(round(target_x / dx_target))
    x = np.linspace(0.0, target_x, n_intervals + 1)
    dx = x[1] - x[0]
    y_exact = analytic_trajectory(x, vx0, vy0, g)

    shooting_slope, y_shooting, shooting_history = shooting_method(
        x=x,
        target_y=target_y,
        acceleration_x=acceleration_x,
        guess0=0.2,
        guess1=1.4,
    )

    snapshot_iterations = {0, 1, 3, 10, 30, 100, 300, 1000, 3000}
    y_relax, relaxation_snapshots, final_iteration, final_change = relaxation_method(
        x=x,
        y_left=0.0,
        y_right=target_y,
        acceleration_x=acceleration_x,
        snapshot_iterations=snapshot_iterations,
    )

    shooting_error = np.max(np.abs(y_shooting - y_exact))
    relaxation_error = np.max(np.abs(y_relax - y_exact))

    print("Projectile motion as a boundary-value problem")
    print(f"v0 = {v0} m/s, theta = {theta_deg} deg, g = {g} m/s^2")
    print(f"target x = {target_x:.6f} m, target y = {target_y:.6f} m")
    print(f"grid dx target = {dx_target}, actual dx = {dx:.8f}, points = {len(x)}")
    print("-" * 72)
    print("Shooting method secant iterations")
    print(f"{'iteration':>10} {'initial slope':>18} {'miss at target':>18}")
    print("-" * 72)

    for iteration, slope, miss in shooting_history:
        print(f"{iteration:10d} {slope:18.10f} {miss:18.10e}")

    print("-" * 72)
    print(f"exact initial slope dy/dx = {vy0 / vx0:.10f}")
    print(f"shooting initial slope = {shooting_slope:.10f}")
    print(f"relaxation iterations = {final_iteration}, final max change = {final_change:.3e}")
    print(f"max |shooting - analytic| = {shooting_error:.3e} m")
    print(f"max |relaxation - analytic| = {relaxation_error:.3e} m")

    fig, (ax_final, ax_relax) = plt.subplots(
        2,
        1,
        figsize=(10, 9),
        sharex=True,
        gridspec_kw={"height_ratios": [1, 1]},
    )

    ax_final.plot(x, y_exact, linewidth=2.2, label="analytic solution")
    ax_final.plot(
        x,
        y_shooting,
        linestyle="--",
        linewidth=1.8,
        label="shooting method",
    )
    ax_final.plot(
        x,
        y_relax,
        linestyle=":",
        linewidth=2.2,
        label="relaxation method",
    )
    ax_final.scatter([target_x], [target_y], color="black", s=28, label="target")
    ax_final.set_title("Shooting and Relaxation Methods for Projectile Motion")
    ax_final.set_ylabel("y (m)")
    ax_final.grid(True, alpha=0.3)
    ax_final.legend()

    for iteration in sorted(relaxation_snapshots):
        y_snapshot = relaxation_snapshots[iteration]
        if iteration == final_iteration:
            label = f"final ({iteration})"
            linewidth = 2.2
            alpha = 1.0
        else:
            label = f"iteration {iteration}"
            linewidth = 1.1
            alpha = 0.55

        ax_relax.plot(x, y_snapshot, linewidth=linewidth, alpha=alpha, label=label)

    ax_relax.plot(
        x,
        y_exact,
        color="black",
        linestyle="--",
        linewidth=1.4,
        label="analytic solution",
    )
    ax_relax.set_title("Relaxation Method: Intermediate Iterations")
    ax_relax.set_xlabel("x (m)")
    ax_relax.set_ylabel("y (m)")
    ax_relax.grid(True, alpha=0.3)
    ax_relax.legend(ncol=2, fontsize=8)

    fig.tight_layout()
    fig.savefig("example-3A.png", dpi=200)
    print("\nSaved graph to example-3A.png")
    plt.show()


if __name__ == "__main__":
    main()
