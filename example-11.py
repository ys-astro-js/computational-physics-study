import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


X_RANGE = (-3.0, 3.0)
Y_RANGE = (-2.5, 2.5)
GRID_POINTS = 121
SOURCE_SEPARATION = 1.1
SOURCE_WIDTH = 0.75
OUTPUT_PATH = "example-11.png"


def electric_potential(x_values, y_values):
    positive_bump = np.exp(
        -((x_values + SOURCE_SEPARATION) ** 2 + y_values**2) / SOURCE_WIDTH**2
    )
    negative_bump = np.exp(
        -((x_values - SOURCE_SEPARATION) ** 2 + y_values**2) / SOURCE_WIDTH**2
    )
    return positive_bump - negative_bump


def electric_field_from_potential(potential, x_grid, y_grid):
    dx = x_grid[1] - x_grid[0]
    dy = y_grid[1] - y_grid[0]
    dV_dy, dV_dx = np.gradient(potential, dy, dx)

    field_x = -dV_dx
    field_y = -dV_dy
    magnitude = np.hypot(field_x, field_y)

    return field_x, field_y, magnitude


def plot_result(x_values, y_values, potential, field_x, field_y, magnitude):
    fig, (ax_potential, ax_field) = plt.subplots(
        1, 2, figsize=(11, 4.5), constrained_layout=True
    )

    potential_plot = ax_potential.contourf(
        x_values, y_values, potential, levels=31, cmap="coolwarm"
    )
    fig.colorbar(potential_plot, ax=ax_potential, label="V")
    ax_potential.contour(x_values, y_values, potential, levels=15, colors="0.25")
    ax_potential.set_title("Electric Potential")
    ax_potential.set_xlabel("x")
    ax_potential.set_ylabel("y")
    ax_potential.set_aspect("equal")

    stride = 6
    ax_field.contour(x_values, y_values, potential, levels=15, colors="0.75")
    field_plot = ax_field.quiver(
        x_values[::stride, ::stride],
        y_values[::stride, ::stride],
        field_x[::stride, ::stride],
        field_y[::stride, ::stride],
        magnitude[::stride, ::stride],
        cmap="viridis",
        pivot="middle",
    )
    fig.colorbar(field_plot, ax=ax_field, label="|E|")
    ax_field.set_title("Electric Field from E = -grad V")
    ax_field.set_xlabel("x")
    ax_field.set_ylabel("y")
    ax_field.set_aspect("equal")

    fig.savefig(OUTPUT_PATH, dpi=200)


def main():
    x_grid = np.linspace(*X_RANGE, GRID_POINTS)
    y_grid = np.linspace(*Y_RANGE, GRID_POINTS)
    x_values, y_values = np.meshgrid(x_grid, y_grid)

    potential = electric_potential(x_values, y_values)
    field_x, field_y, magnitude = electric_field_from_potential(
        potential, x_grid, y_grid
    )

    print("Approximate electric field from electric potential")
    print("E_x = -dV/dx, E_y = -dV/dy")
    print(f"grid spacing: dx = {x_grid[1] - x_grid[0]:.4f}, dy = {y_grid[1] - y_grid[0]:.4f}")
    print(f"max |V| = {np.max(np.abs(potential)):.6f}")
    print(f"max |E| = {np.max(magnitude):.6f}")

    plot_result(x_values, y_values, potential, field_x, field_y, magnitude)
    print(f"Saved graph to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
