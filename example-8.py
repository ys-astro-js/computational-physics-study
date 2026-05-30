import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def thermal_conductivity(temperature, coefficient):
    return coefficient / temperature


def rhs(state):
    temperature, gradient = state
    return np.array([gradient, gradient**2 / temperature])


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

    return x_values, states[:, 0], states[:, 1]


def boundary_residual(
    length, left_temperature, right_temperature, initial_gradient, n_points
):
    _, temperatures, _ = integrate_temperature(
        length, left_temperature, initial_gradient, n_points
    )
    return temperatures[-1] - right_temperature


def secant_shooting_method(
    length,
    left_temperature,
    right_temperature,
    gradient_0,
    gradient_1,
    n_points,
    tolerance=1.0e-12,
    max_iter=50,
):
    residual_0 = boundary_residual(
        length, left_temperature, right_temperature, gradient_0, n_points
    )
    residual_1 = boundary_residual(
        length, left_temperature, right_temperature, gradient_1, n_points
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
            length, left_temperature, right_temperature, gradient_next, n_points
        )
        history.append((iteration, gradient_next, residual_next))

        if abs(residual_next) < tolerance:
            return gradient_next, history

        gradient_0, residual_0 = gradient_1, residual_1
        gradient_1, residual_1 = gradient_next, residual_next

    raise RuntimeError("Secant shooting method did not converge.")


def exact_temperature(x_values, length, left_temperature, right_temperature):
    exponent = np.log(right_temperature / left_temperature) / length
    return left_temperature * np.exp(exponent * x_values)


def plot_temperature(x_values, numerical, exact, output_path):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x_values, numerical, linewidth=2.0, label="secant shooting method")
    ax.plot(x_values, exact, "--", linewidth=2.0, label="exact solution")
    ax.set_title("Steady Temperature Distribution for k(T) = C/T")
    ax.set_xlabel("x")
    ax.set_ylabel("T(x)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)


def main():
    length = 10.0
    left_temperature = 100.0
    right_temperature = 200.0
    coefficient = 10.0
    n_points = 1001

    gradient_0 = 5.0
    gradient_1 = 10.0

    gradient, history = secant_shooting_method(
        length,
        left_temperature,
        right_temperature,
        gradient_0,
        gradient_1,
        n_points,
    )
    x_values, numerical, gradients = integrate_temperature(
        length, left_temperature, gradient, n_points
    )
    exact = exact_temperature(x_values, length, left_temperature, right_temperature)
    conductivity = thermal_conductivity(numerical, coefficient)
    heat_flux = -conductivity * gradients
    max_error = np.max(np.abs(numerical - exact))

    print("Steady 1D rod temperature using secant shooting")
    print(
        f"L = {length}, T_L = {left_temperature}, T_R = {right_temperature}, "
        f"C = {coefficient}"
    )
    print("ODE: d/dx((C/T) dT/dx) = 0, boundary conditions: T(0)=T_L, T(L)=T_R")
    print("-" * 72)
    print(f"{'iteration':>9} {'T_prime(0)':>18} {'T(L)-T_R':>18}")
    print("-" * 72)
    for iteration, trial_gradient, residual in history:
        print(f"{iteration:9d} {trial_gradient:18.10f} {residual:18.8e}")

    exponent = np.log(right_temperature / left_temperature) / length
    print("\nSolution")
    print(f"T'(0) = {gradient:.10f}")
    print(f"T(x) = {left_temperature:.10f} * exp({exponent:.10f} x)")
    print(f"heat flux = {np.mean(heat_flux):.10f}")
    print(f"max error against exact solution = {max_error:.8e}")

    output_path = "example-8.png"
    plot_temperature(x_values, numerical, exact, output_path)
    print(f"Saved graph to {output_path}")


if __name__ == "__main__":
    main()
