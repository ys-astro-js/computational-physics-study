import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def horizontal_velocity(v0, vy):
    return np.sqrt(v0**2 - vy**2)


def projectile_range(v0, vy, g):
    vx = horizontal_velocity(v0, vy)
    flight_time = 2.0 * vy / g
    return vx * flight_time


def velocity_equation(vy, v0, target_x, g):
    return projectile_range(v0, vy, g) - target_x


def velocity_equation_derivative(vy, v0, g):
    vx = horizontal_velocity(v0, vy)
    return 2.0 * (v0**2 - 2.0 * vy**2) / (g * vx)


def newton_method(v0, target_x, g, initial_vy, tolerance=1.0e-12, max_iter=50):
    vy = initial_vy
    history = [vy]

    for _ in range(max_iter):
        f = velocity_equation(vy, v0, target_x, g)
        df = velocity_equation_derivative(vy, v0, g)

        if abs(df) < 1.0e-14:
            raise ZeroDivisionError("Newton derivative became too small.")

        next_vy = vy - f / df
        history.append(next_vy)

        if abs(next_vy - vy) < tolerance:
            return next_vy, history

        vy = next_vy

    raise RuntimeError("Newton method did not converge.")


def secant_method(v0, target_x, g, vy1, vy2, tolerance=1.0e-12, max_iter=50):
    history = [vy1, vy2]

    for _ in range(max_iter):
        f1 = velocity_equation(vy1, v0, target_x, g)
        f2 = velocity_equation(vy2, v0, target_x, g)
        denominator = f2 - f1

        if abs(denominator) < 1.0e-14:
            raise ZeroDivisionError("Secant denominator became too small.")

        next_vy = vy2 - f2 * (vy2 - vy1) / denominator
        history.append(next_vy)

        if abs(next_vy - vy2) < tolerance:
            return next_vy, history

        vy1, vy2 = vy2, next_vy

    raise RuntimeError("Secant method did not converge.")


def bisection_method(v0, target_x, g, left, right, tolerance=1.0e-12, max_iter=100):
    f_left = velocity_equation(left, v0, target_x, g)
    f_right = velocity_equation(right, v0, target_x, g)

    if f_left * f_right > 0.0:
        raise ValueError("Bisection interval does not bracket a root.")

    history = []

    for _ in range(max_iter):
        middle = 0.5 * (left + right)
        f_middle = velocity_equation(middle, v0, target_x, g)
        history.append(middle)

        if abs(f_middle) < tolerance or 0.5 * (right - left) < tolerance:
            return middle, history

        if f_left * f_middle < 0.0:
            right = middle
            f_right = f_middle
        else:
            left = middle
            f_left = f_middle

    raise RuntimeError("Bisection method did not converge.")


def time_method(repeats, solver):
    start = time.perf_counter()
    root = None
    history = None

    for _ in range(repeats):
        root, history = solver()

    elapsed = time.perf_counter() - start
    return root, history, elapsed / repeats


def trajectory(v0, vy, g, points=500):
    vx = horizontal_velocity(v0, vy)
    t_end = 2.0 * vy / g
    t = np.linspace(0.0, t_end, points)
    x = vx * t
    y = vy * t - 0.5 * g * t**2
    return x, y


def angle_degrees(v0, vy):
    vx = horizontal_velocity(v0, vy)
    return np.rad2deg(np.arctan2(vy, vx))


def plot_results(results, v0, target_x, g, output_path):
    fig, (ax_traj, ax_time, ax_conv) = plt.subplots(1, 3, figsize=(17, 5))

    for result in results:
        x, y = trajectory(v0, result["vy"], g)
        label = f"{result['name']} ({angle_degrees(v0, result['vy']):.6f} deg)"
        ax_traj.plot(x, y, linewidth=2.0, label=label)

    ax_traj.scatter([target_x], [0.0], color="tab:red", zorder=4, label="target")
    ax_traj.set_title("Projectile Trajectories")
    ax_traj.set_xlabel("x (m)")
    ax_traj.set_ylabel("y (m)")
    ax_traj.grid(True, alpha=0.3)
    ax_traj.legend()

    names = [result["name"] for result in results]
    times = [result["average_time"] * 1.0e6 for result in results]
    colors = ["tab:blue", "tab:orange", "tab:green"]
    ax_time.bar(names, times, color=colors)
    ax_time.set_title("Runtime Comparison")
    ax_time.set_ylabel("average time per solve (microseconds)")
    ax_time.grid(True, axis="y", alpha=0.3)

    for result in results:
        errors = np.abs(
            [
                velocity_equation(vy, v0, target_x, g)
                for vy in result["history"]
                if 0.0 < vy < v0
            ]
        )
        ax_conv.semilogy(range(len(errors)), errors, marker="o", label=result["name"])

    ax_conv.set_title("Convergence History")
    ax_conv.set_xlabel("iteration")
    ax_conv.set_ylabel("|f(vy)| (m)")
    ax_conv.grid(True, which="both", alpha=0.3)
    ax_conv.legend()

    fig.suptitle("Newton, Secant, and Bisection Methods for Projectile Angle")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)


def main():
    v0 = 100.0
    target_x = 100.0
    g = 9.8
    repeats = 20000

    newton_initial_vy = 1.0
    secant_v1 = 1.0
    secant_v2 = 2.0
    bisection_left = 1.0
    bisection_right = 10.0

    solvers = [
        (
            "Newton",
            lambda: newton_method(v0, target_x, g, newton_initial_vy),
        ),
        (
            "Secant",
            lambda: secant_method(v0, target_x, g, secant_v1, secant_v2),
        ),
        (
            "Bisection",
            lambda: bisection_method(v0, target_x, g, bisection_left, bisection_right),
        ),
    ]

    results = []
    for name, solver in solvers:
        vy, history, average_time = time_method(repeats, solver)
        results.append(
            {
                "name": name,
                "vy": vy,
                "history": history,
                "iterations": len(history) - 1,
                "average_time": average_time,
            }
        )

    print("Projectile angle from vertical initial velocity vy")
    print(f"v0 = {v0} m/s, target distance = {target_x} m, g = {g} m/s^2")
    print(f"Secant initial velocities: v1 = {secant_v1}, v2 = {secant_v2} m/s")
    print(f"Runtime average: {repeats} solves per method")
    print("-" * 104)
    print(
        f"{'method':>12} {'vy (m/s)':>16} {'angle (deg)':>16}"
        f" {'range error (m)':>18} {'iterations':>12} {'avg time (us)':>16}"
    )
    print("-" * 104)

    for result in results:
        vy = result["vy"]
        print(
            f"{result['name']:>12} {vy:16.10f} {angle_degrees(v0, vy):16.10f}"
            f" {velocity_equation(vy, v0, target_x, g):18.8e}"
            f" {result['iterations']:12d} {result['average_time'] * 1.0e6:16.6f}"
        )

    output_path = "example-6.png"
    plot_results(results, v0, target_x, g, output_path)
    print(f"\nSaved graph to {output_path}")


if __name__ == "__main__":
    main()
