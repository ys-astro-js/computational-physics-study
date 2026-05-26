import time

import matplotlib.pyplot as plt
import numpy as np


WEIERSTRASS_A = 0.5
WEIERSTRASS_B = 3
WEIERSTRASS_TERMS = 8
TARGET_VALUE = -0.975
X_MIN = 0.0
X_MAX = 1.5


def weierstrass_function(x):
    x = np.asarray(x, dtype=float)
    total = np.zeros_like(x, dtype=float)

    for n in range(WEIERSTRASS_TERMS):
        total += WEIERSTRASS_A**n * np.cos((WEIERSTRASS_B**n) * np.pi * x)

    return total


def root_equation(x):
    return weierstrass_function(x) - TARGET_VALUE


def numerical_derivative(x, h=1.0e-6):
    return (root_equation(x + h) - root_equation(x - h)) / (2.0 * h)


def newton_method(initial_x, tolerance=1.0e-12, max_iter=50):
    x = initial_x
    history = [x]

    for _ in range(max_iter):
        f = root_equation(x)
        df = numerical_derivative(x)

        if abs(df) < 1.0e-14:
            raise ZeroDivisionError("Newton derivative became too small.")

        next_x = x - f / df
        history.append(next_x)

        if abs(next_x - x) < tolerance or abs(root_equation(next_x)) < tolerance:
            return next_x, history

        x = next_x

    raise RuntimeError("Newton method did not converge.")


def secant_method(x0, x1, tolerance=1.0e-12, max_iter=50):
    history = [x0, x1]

    for _ in range(max_iter):
        f0 = root_equation(x0)
        f1 = root_equation(x1)
        denominator = f1 - f0

        if abs(denominator) < 1.0e-14:
            raise ZeroDivisionError("Secant denominator became too small.")

        next_x = x1 - f1 * (x1 - x0) / denominator
        history.append(next_x)

        if abs(next_x - x1) < tolerance or abs(root_equation(next_x)) < tolerance:
            return next_x, history

        x0, x1 = x1, next_x

    raise RuntimeError("Secant method did not converge.")


def bisection_method(left, right, tolerance=1.0e-12, max_iter=100):
    f_left = root_equation(left)
    f_right = root_equation(right)

    if f_left * f_right > 0.0:
        raise ValueError("Bisection interval does not bracket a root.")

    history = []

    for _ in range(max_iter):
        middle = 0.5 * (left + right)
        f_middle = root_equation(middle)
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


def show_process(results):
    fig, (ax_function, ax_conv, ax_time) = plt.subplots(1, 3, figsize=(17, 5))

    x_values = np.linspace(X_MIN, X_MAX, 2000)
    y_values = root_equation(x_values)
    ax_function.plot(x_values, y_values, color="black", linewidth=1.4, label="W(x) - target")
    ax_function.axhline(0.0, color="gray", linestyle="--", linewidth=1.0)
    ax_function.set_xlim(X_MIN, X_MAX)

    ax_function.set_title("Weierstrass Equation")
    ax_function.set_xlabel("x")
    ax_function.set_ylabel("W(x) - target")
    ax_function.grid(True, alpha=0.3)

    ax_conv.set_title("Convergence History")
    ax_conv.set_xlabel("iteration")
    ax_conv.set_ylabel("|f(x)|")
    ax_conv.grid(True, which="both", alpha=0.3)

    names = [result["name"] for result in results]
    times = [result["average_time"] * 1.0e6 for result in results]
    colors = ["tab:blue", "tab:orange", "tab:green"]
    ax_time.bar(names, times, color=colors)
    ax_time.set_title("Runtime Comparison")
    ax_time.set_ylabel("average time per solve (microseconds)")
    ax_time.grid(True, axis="y", alpha=0.3)

    fig.suptitle("Newton, Secant, and Bisection Methods for a Nondifferentiable Problem")
    fig.tight_layout()

    markers = ["o", "s", "^"]
    path_lines = []
    error_lines = []

    for result, marker in zip(results, markers):
        (path_line,) = ax_function.plot(
            [],
            [],
            marker=marker,
            linewidth=1.3,
            markersize=5,
            label=result["name"],
        )
        (error_line,) = ax_conv.semilogy([], [], marker="o", label=result["name"])
        path_lines.append(path_line)
        error_lines.append(error_line)

    ax_function.legend()
    ax_conv.legend()
    plt.ion()
    plt.show(block=False)

    max_steps = max(len(result["history"]) for result in results)
    for step in range(1, max_steps + 1):
        for result, path_line, error_line in zip(results, path_lines, error_lines):
            xs = np.array(result["history"][:step])
            if len(xs) == 0:
                continue

            ys = root_equation(xs)
            errors = np.abs(ys)
            path_line.set_data(xs, ys)
            error_line.set_data(range(len(errors)), errors)

        ax_conv.relim()
        ax_conv.autoscale_view()
        fig.canvas.draw_idle()
        plt.pause(0.35)

    plt.ioff()
    plt.show()


def main():
    repeats = 20000
    newton_initial_x = 0.7775
    secant_initial_x0 = 0.777
    secant_initial_x1 = 0.778
    bisection_left = 0.777
    bisection_right = 0.778

    solvers = [
        ("Newton", lambda: newton_method(newton_initial_x)),
        ("Secant", lambda: secant_method(secant_initial_x0, secant_initial_x1)),
        ("Bisection", lambda: bisection_method(bisection_left, bisection_right)),
    ]

    results = []
    for name, solver in solvers:
        root, history, average_time = time_method(repeats, solver)
        results.append(
            {
                "name": name,
                "root": root,
                "history": history,
                "iterations": len(history) - 1,
                "average_time": average_time,
            }
        )

    print("Weierstrass root-finding problem")
    print(
        "W(x) = sum(a^n cos(b^n pi x)), "
        f"a = {WEIERSTRASS_A}, b = {WEIERSTRASS_B}, terms = {WEIERSTRASS_TERMS}"
    )
    print(f"Solving W(x) = {TARGET_VALUE} on x range [{X_MIN}, {X_MAX}]")
    print("Newton uses a central-difference slope because the Weierstrass function is nondifferentiable.")
    print(f"Newton initial x = {newton_initial_x}")
    print(f"Secant initial x values = {secant_initial_x0}, {secant_initial_x1}")
    print(f"Bisection interval = [{bisection_left}, {bisection_right}]")
    print(f"Runtime average: {repeats} solves per method")
    print("-" * 88)
    print(
        f"{'method':>12} {'root x':>18} {'f(root)':>18}"
        f" {'iterations':>12} {'avg time (us)':>16}"
    )
    print("-" * 88)

    for result in results:
        root = result["root"]
        print(
            f"{result['name']:>12} {root:18.12f}"
            f" {root_equation(root):18.8e}"
            f" {result['iterations']:12d} {result['average_time'] * 1.0e6:16.6f}"
        )

    print("\nShowing matplotlib animation. Close the figure window to finish.")
    show_process(results)


if __name__ == "__main__":
    main()
