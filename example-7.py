import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def rhs(state):
    _temperature, gradient = state
    return np.array([gradient, 0.0])


def integrate_temperature(length, left_temperature, initial_gradient, n_points):
    x_values = np.linspace(0.0, length, n_points)
    dx = x_values[1] - x_values[0]
    states = np.zeros((n_points, 2))
    states[0] = [left_temperature, initial_gradient]

    for i in range(n_points - 1):
        y = states[i]
        k1 = rhs(y)
        k2 = rhs(y + 0.5 * dx * k1)
        k3 = rhs(y + 0.5 * dx * k2)
        k4 = rhs(y + dx * k3)
        states[i + 1] = y + dx * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0

    return x_values, states[:, 0]


def boundary_residual(length, left_temperature, right_temperature, initial_gradient):
    _, temperatures = integrate_temperature(
        length, left_temperature, initial_gradient, n_points=101
    )
    return temperatures[-1] - right_temperature


def shooting_method(
    length,
    left_temperature,
    right_temperature,
    gradient_0,
    gradient_1,
    tolerance=1.0e-12,
    max_iter=50,
):
    residual_0 = boundary_residual(
        length, left_temperature, right_temperature, gradient_0
    )
    residual_1 = boundary_residual(
        length, left_temperature, right_temperature, gradient_1
    )
    history = [(0, gradient_0, residual_0), (1, gradient_1, residual_1)]

    for iteration in range(2, max_iter + 1):
        denominator = residual_1 - residual_0
        if abs(denominator) < 1.0e-14:
            raise ZeroDivisionError("Secant denominator became too small.")

        gradient_next = gradient_1 - residual_1 * (
            gradient_1 - gradient_0
        ) / denominator
        residual_next = boundary_residual(
            length, left_temperature, right_temperature, gradient_next
        )
        history.append((iteration, gradient_next, residual_next))

        if abs(residual_next) < tolerance:
            return gradient_next, history

        gradient_0, residual_0 = gradient_1, residual_1
        gradient_1, residual_1 = gradient_next, residual_next

    raise RuntimeError("Shooting method did not converge.")


def exact_temperature(x_values, length, left_temperature, right_temperature):
    return left_temperature + (right_temperature - left_temperature) * x_values / length


def plot_temperature(x_values, numerical, exact, output_path):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x_values, numerical, linewidth=2.0, label="shooting method")
    ax.plot(x_values, exact, "--", linewidth=2.0, label="exact solution")
    ax.set_title("Steady Temperature Distribution in a 1D Rod")
    ax.set_xlabel("x")
    ax.set_ylabel("T(x)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)


def main():
    length = 1.0
    left_temperature = 100.0
    right_temperature = 300.0
    n_points = 101

    gradient_0 = 0.0
    gradient_1 = 1.0

    gradient, history = shooting_method(
        length,
        left_temperature,
        right_temperature,
        gradient_0,
        gradient_1,
    )
    x_values, numerical = integrate_temperature(
        length, left_temperature, gradient, n_points
    )
    exact = exact_temperature(x_values, length, left_temperature, right_temperature)
    max_error = np.max(np.abs(numerical - exact))

    print("Steady 1D rod temperature using the shooting method")
    print(f"L = {length}, T_L = {left_temperature}, T_R = {right_temperature}")
    print("ODE: T''(x) = 0, boundary conditions: T(0)=T_L, T(L)=T_R")
    print("-" * 72)
    print(f"{'iteration':>9} {'T_prime(0)':>18} {'T(L)-T_R':>18}")
    print("-" * 72)
    for iteration, trial_gradient, residual in history:
        print(f"{iteration:9d} {trial_gradient:18.10f} {residual:18.8e}")

    print("\nSolution")
    print(f"T'(0) = {gradient:.10f}")
    print(f"T(x) = {left_temperature:.10f} + ({gradient:.10f}) x")
    print(f"max error against exact linear solution = {max_error:.8e}")

    output_path = "example-7.png"
    plot_temperature(x_values, numerical, exact, output_path)
    print(f"Saved graph to {output_path}")


if __name__ == "__main__":
    main()
