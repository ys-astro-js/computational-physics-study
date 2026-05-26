import argparse
import math
import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt


DERIVATIVE_EPSILON = 1.0e-14
DEFAULT_TOLERANCE = 1.0e-12


@dataclass(frozen=True)
class ProjectileConfig:
    v0: float = 100.0
    target_x: float = 100.0
    g: float = 9.8
    dt: float = 0.05
    repeats: int = 10000


@dataclass(frozen=True)
class SolverResult:
    label: str
    vy: float
    history: list[tuple]
    iterations: int
    total_time: float
    average_time: float


@dataclass(frozen=True)
class AnimatedTrajectory:
    vy: float
    vx: float
    label: str
    t_end: float
    max_height: float


def horizontal_velocity(v0, vy):
    if vy < 0.0 or vy > v0:
        raise ValueError("vy must satisfy 0 <= vy <= v0.")

    return math.sqrt(v0**2 - vy**2)


def angle_from_vertical_velocity(v0, vy):
    vx = horizontal_velocity(v0, vy)
    return math.atan2(vy, vx)


def projectile_range_from_vy(v0, vy, g):
    vx = horizontal_velocity(v0, vy)
    flight_time = 2.0 * vy / g
    return vx * flight_time


def velocity_equation(vy, v0, target_x, g):
    return projectile_range_from_vy(v0, vy, g) - target_x


def velocity_equation_derivative(vy, v0, g):
    vx = horizontal_velocity(v0, vy)
    return 2.0 * (v0**2 - 2.0 * vy**2) / (g * vx)


def newton_velocity(
    v0,
    target_x,
    g,
    vy0,
    tolerance=DEFAULT_TOLERANCE,
    max_iter=50,
    keep_history=True,
):
    vy = vy0
    history = []

    for iteration in range(1, max_iter + 1):
        f = velocity_equation(vy, v0, target_x, g)
        df = velocity_equation_derivative(vy, v0, g)

        if abs(df) < DERIVATIVE_EPSILON:
            raise ZeroDivisionError("Newton derivative became too small.")

        vy_next = vy - f / df
        if keep_history:
            history.append((iteration, vy, f, df, vy_next))

        if abs(vy_next - vy) < tolerance:
            return vy_next, history

        vy = vy_next

    raise RuntimeError("Newton method did not converge.")


def secant_velocity(
    v0,
    target_x,
    g,
    vy0,
    vy1,
    tolerance=DEFAULT_TOLERANCE,
    max_iter=50,
    keep_history=True,
):
    history = []

    for iteration in range(1, max_iter + 1):
        f0 = velocity_equation(vy0, v0, target_x, g)
        f1 = velocity_equation(vy1, v0, target_x, g)
        denominator = f1 - f0

        if abs(denominator) < DERIVATIVE_EPSILON:
            raise ZeroDivisionError("Secant denominator became too small.")

        vy_next = vy1 - f1 * (vy1 - vy0) / denominator
        if keep_history:
            history.append((iteration, vy0, vy1, f0, f1, vy_next))

        if abs(vy_next - vy1) < tolerance:
            return vy_next, history

        vy0, vy1 = vy1, vy_next

    raise RuntimeError("Secant method did not converge.")


def bisection_velocity(
    v0,
    target_x,
    g,
    left,
    right,
    tolerance=DEFAULT_TOLERANCE,
    max_iter=100,
    keep_history=True,
):
    f_left = velocity_equation(left, v0, target_x, g)
    f_right = velocity_equation(right, v0, target_x, g)

    if abs(f_left) < tolerance:
        return left, []

    if abs(f_right) < tolerance:
        return right, []

    if f_left * f_right > 0.0:
        raise ValueError("Bisection interval does not bracket a root.")

    history = []

    for iteration in range(1, max_iter + 1):
        middle = 0.5 * (left + right)
        f_middle = velocity_equation(middle, v0, target_x, g)
        if keep_history:
            history.append((iteration, left, right, middle, f_middle))

        if abs(f_middle) < tolerance or 0.5 * (right - left) < tolerance:
            return middle, history

        if f_left * f_middle < 0.0:
            right = middle
        else:
            left = middle
            f_left = f_middle

    raise RuntimeError("Bisection method did not converge.")


def flight_time(vy, g):
    return 2.0 * vy / g


def print_newton_history(initial_vy, history):
    print(f"\nNewton method: initial vy = {initial_vy:.6f} m/s")
    print("-" * 100)
    print(
        f"{'n':>3} {'vy_n (m/s)':>16} {'f(vy_n)':>16}"
        f" {'f_prime(vy_n)':>18} {'vy_n+1 (m/s)':>18}"
    )
    print("-" * 100)

    for iteration, vy, f, df, vy_next in history:
        print(
            f"{iteration:3d} {vy:16.10f} {f:16.8e}"
            f" {df:18.8e} {vy_next:18.10f}"
        )


def print_secant_history(vy0, vy1, history):
    print(f"\nSecant method: initial vy values = {vy0:.6f}, {vy1:.6f} m/s")
    print("-" * 118)
    print(
        f"{'n':>3} {'vy_n-1 (m/s)':>18} {'vy_n (m/s)':>16}"
        f" {'f(vy_n-1)':>18} {'f(vy_n)':>16} {'vy_n+1 (m/s)':>18}"
    )
    print("-" * 118)

    for iteration, prev_vy, vy, f_prev, f, vy_next in history:
        print(
            f"{iteration:3d} {prev_vy:18.10f} {vy:16.10f}"
            f" {f_prev:18.8e} {f:16.8e} {vy_next:18.10f}"
        )


def print_bisection_history(left, right, history):
    print(f"\nBisection method: vy interval = [{left:.6f}, {right:.6f}] m/s")
    print("-" * 96)
    print(
        f"{'n':>3} {'left (m/s)':>14} {'right (m/s)':>14}"
        f" {'middle (m/s)':>16} {'f(middle)':>16}"
    )
    print("-" * 96)

    for iteration, left_i, right_i, middle, f_middle in history:
        print(
            f"{iteration:3d} {left_i:14.10f} {right_i:14.10f}"
            f" {middle:16.10f} {f_middle:16.8e}"
        )


def timed_solve(
    label: str,
    repeats: int,
    solve_once: Callable[[bool], tuple[float, list[tuple]]],
):
    vy, history = solve_once(True)

    start = time.perf_counter()
    for _ in range(repeats):
        vy, _ = solve_once(False)

    elapsed = time.perf_counter() - start
    return SolverResult(
        label=label,
        vy=vy,
        history=history,
        iterations=len(history),
        total_time=elapsed,
        average_time=elapsed / repeats,
    )


def print_solution(label, vy, v0, target_x, g):
    vx = horizontal_velocity(v0, vy)
    theta = angle_from_vertical_velocity(v0, vy)
    distance = projectile_range_from_vy(v0, vy, g)
    print(
        f"{label} solution: vy = {vy:.10f} m/s, vx = {vx:.10f} m/s, "
        f"theta = {math.degrees(theta):.10f} deg, range = {distance:.8f} m"
    )


def print_runtime_comparison(results, repeats, v0, target_x, g):
    print(f"\nRuntime comparison averaged over {repeats} runs")
    print("-" * 116)
    print(
        f"{'method':>18} {'vy (m/s)':>16} {'theta (deg)':>16}"
        f" {'iterations':>12} {'range error (m)':>18} {'avg time (us)':>16}"
    )
    print("-" * 116)

    for result in results:
        vy = result.vy
        theta = angle_from_vertical_velocity(v0, vy)
        range_error = velocity_equation(vy, v0, target_x, g)
        print(
            f"{result.label:>18} {vy:16.10f} {math.degrees(theta):16.10f}"
            f" {result.iterations:12d} {range_error:18.8e}"
            f" {result.average_time * 1.0e6:16.6f}"
        )


def prepare_trajectory(v0, vy, g, label):
    return AnimatedTrajectory(
        vy=vy,
        vx=horizontal_velocity(v0, vy),
        label=label,
        t_end=flight_time(vy, g),
        max_height=vy**2 / (2.0 * g),
    )


def trajectory_from_velocity(vx, vy, g, t):
    return vx * t, vy * t - 0.5 * g * t**2


def animate_trajectories(solutions, v0, target_x, g, dt):
    trajectories = [prepare_trajectory(v0, vy, g, label) for vy, label in solutions]
    max_time = max(trajectory.t_end for trajectory in trajectories)
    max_height = max(trajectory.max_height for trajectory in trajectories)

    plt.ion()
    fig, ax = plt.subplots(figsize=(10, 6))
    lines = []
    points = []

    for trajectory in trajectories:
        (line,) = ax.plot([], [], linewidth=2.0, label=trajectory.label)
        (point,) = ax.plot([], [], "o", markersize=7)
        lines.append(line)
        points.append(point)

    ax.scatter([target_x], [0.0], color="red", zorder=3, label="target")
    time_text = ax.text(0.02, 0.95, "", transform=ax.transAxes)
    ax.set_title(f"Projectile Trajectory Updated Every dt = {dt} s")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_xlim(0.0, target_x * 1.05)
    ax.set_ylim(0.0, max_height * 1.08)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    t_values = np.arange(0.0, max_time + dt, dt)
    path_times = [t_values[t_values <= trajectory.t_end] for trajectory in trajectories]

    for t in t_values:
        for i, trajectory in enumerate(trajectories):
            t_path = path_times[i][path_times[i] <= t]
            x_path, y_path = trajectory_from_velocity(
                trajectory.vx, trajectory.vy, g, t_path
            )
            lines[i].set_data(x_path, y_path)

            if t <= trajectory.t_end:
                x_now, y_now = trajectory_from_velocity(
                    trajectory.vx, trajectory.vy, g, t
                )
                points[i].set_data([x_now], [max(y_now, 0.0)])
            else:
                x_end, y_end = trajectory_from_velocity(
                    trajectory.vx, trajectory.vy, g, trajectory.t_end
                )
                points[i].set_data([x_end], [y_end])

        time_text.set_text(f"t = {t:.2f} s")
        fig.canvas.draw_idle()
        plt.pause(0.02)

    plt.ioff()
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--v0", type=float, default=100.0, help="Initial speed in m/s.")
    parser.add_argument(
        "--target-x", type=float, default=100.0, help="Target distance in meters."
    )
    parser.add_argument("--g", type=float, default=9.8, help="Gravity in m/s^2.")
    parser.add_argument(
        "--dt", type=float, default=0.05, help="Animation time step in seconds."
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=10000,
        help="Number of repeated solves used for runtime comparison.",
    )
    parser.add_argument(
        "--no-animation",
        action="store_true",
        help="Only print numerical results and skip the matplotlib animation.",
    )
    args = parser.parse_args()
    config = ProjectileConfig(
        v0=args.v0,
        target_x=args.target_x,
        g=args.g,
        dt=args.dt,
        repeats=args.repeats,
    )

    newton_initial_vy = 1.0
    secant_initial_vy_1 = 1.0
    secant_initial_vy_2 = 2.0
    bisection_left = 1.0
    bisection_right = 10.0

    print("Projectile angle using Newton, Secant, and Bisection methods")
    print(
        f"fixed initial speed = {config.v0} m/s, target distance = {config.target_x} m, "
        f"g = {config.g} m/s^2"
    )
    print("The unknown variable is vertical initial velocity vy, not angle theta.")
    print("After solving vy, theta is computed from theta = atan2(vy, vx).")

    newton_result = timed_solve(
        "Newton",
        config.repeats,
        lambda keep_history: newton_velocity(
            config.v0,
            config.target_x,
            config.g,
            newton_initial_vy,
            keep_history=keep_history,
        ),
    )
    secant_result = timed_solve(
        "Secant",
        config.repeats,
        lambda keep_history: secant_velocity(
            config.v0,
            config.target_x,
            config.g,
            secant_initial_vy_1,
            secant_initial_vy_2,
            keep_history=keep_history,
        ),
    )
    bisection_result = timed_solve(
        "Bisection",
        config.repeats,
        lambda keep_history: bisection_velocity(
            config.v0,
            config.target_x,
            config.g,
            bisection_left,
            bisection_right,
            keep_history=keep_history,
        ),
    )
    results = [newton_result, secant_result, bisection_result]

    print_newton_history(newton_initial_vy, newton_result.history)
    print_solution("Newton", newton_result.vy, config.v0, config.target_x, config.g)

    print_secant_history(
        secant_initial_vy_1, secant_initial_vy_2, secant_result.history
    )
    print_solution("Secant", secant_result.vy, config.v0, config.target_x, config.g)

    print_bisection_history(bisection_left, bisection_right, bisection_result.history)
    print_solution(
        "Bisection", bisection_result.vy, config.v0, config.target_x, config.g
    )

    print_runtime_comparison(
        results, config.repeats, config.v0, config.target_x, config.g
    )

    if args.no_animation:
        return

    print(f"\nAnimating trajectories with dt = {config.dt} s")
    solutions = []
    for result in results:
        vy = result.vy
        theta = angle_from_vertical_velocity(config.v0, vy)
        label = f"{result.label} theta={math.degrees(theta):.6f} deg"
        solutions.append((vy, label))

    animate_trajectories(solutions, config.v0, config.target_x, config.g, config.dt)


if __name__ == "__main__":
    main()
