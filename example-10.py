import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


X_RANGE = (-3.0, 3.0)
Y_RANGE = (-3.0, 3.0)
GRID_POINTS = 121
SAMPLE_Y = 0.0
COULOMB_FACTOR = 1.0
POINT_CHARGE = 1.0
SINGULARITY_CUTOFF = 0.08
OUTPUT_PATH = "example-10.png"


def electric_potential(x_values, y_values):
    radius = np.sqrt(x_values**2 + y_values**2)
    return np.divide(
        COULOMB_FACTOR * POINT_CHARGE,
        radius,
        out=np.full_like(radius, np.nan),
        where=radius > 0.0,
    )


def electric_field_from_point_charge(x_values, y_values):
    radius = np.sqrt(x_values**2 + y_values**2)
    safe_radius = np.maximum(radius, SINGULARITY_CUTOFF)
    capped_magnitude = np.abs(COULOMB_FACTOR * POINT_CHARGE) / safe_radius**2

    unit_x = np.divide(
        x_values,
        radius,
        out=np.full_like(radius, np.nan),
        where=radius > 0.0,
    )
    unit_y = np.divide(
        y_values,
        radius,
        out=np.full_like(radius, np.nan),
        where=radius > 0.0,
    )

    charge_sign = np.sign(POINT_CHARGE)
    electric_field_x = charge_sign * capped_magnitude * unit_x
    electric_field_y = charge_sign * capped_magnitude * unit_y
    magnitude = np.where(radius > 0.0, capped_magnitude, np.nan)

    return electric_field_x, electric_field_y, magnitude


def plot_result(x_values, y_values, potential, field_x, field_y, magnitude, output_path):
    sample_row = np.argmin(np.abs(y_values[:, 0] - SAMPLE_Y))

    fig, (ax_potential, ax_field, ax_strength) = plt.subplots(
        1, 3, figsize=(15, 4.5), constrained_layout=True
    )

    displayed_potential = np.minimum(
        potential, np.abs(COULOMB_FACTOR * POINT_CHARGE) / SINGULARITY_CUTOFF
    )
    potential_plot = ax_potential.contourf(
        x_values, y_values, displayed_potential, levels=30, cmap="viridis"
    )
    fig.colorbar(potential_plot, ax=ax_potential, label="V(x, y)")
    ax_potential.scatter([0.0], [0.0], s=80, color="tab:red", zorder=3)
    ax_potential.set_title("Point-Charge Electric Potential")
    ax_potential.set_xlabel("x")
    ax_potential.set_ylabel("y")
    ax_potential.set_aspect("equal")

    stride = 8
    ax_field.contour(
        x_values,
        y_values,
        displayed_potential,
        levels=12,
        colors="0.7",
        linewidths=0.8,
    )
    ax_field.quiver(
        x_values[::stride, ::stride],
        y_values[::stride, ::stride],
        field_x[::stride, ::stride],
        field_y[::stride, ::stride],
        magnitude[::stride, ::stride],
        cmap="plasma",
    )
    ax_field.scatter([0.0], [0.0], s=80, color="tab:red", zorder=3)
    ax_field.set_title("Point-Charge Field E = kq r_hat / r^2")
    ax_field.set_xlabel("x")
    ax_field.set_ylabel("y")
    ax_field.set_aspect("equal")

    ax_strength.plot(x_values[sample_row], magnitude[sample_row], linewidth=2.0)
    ax_strength.set_title(f"Field Strength at y = {y_values[sample_row, 0]:.2f}")
    ax_strength.set_xlabel("x")
    ax_strength.set_ylabel("|E(x)|")
    ax_strength.grid(True, alpha=0.3)

    fig.savefig(output_path, dpi=200)


def main():
    x_grid = np.linspace(*X_RANGE, GRID_POINTS)
    y_grid = np.linspace(*Y_RANGE, GRID_POINTS)
    x_values, y_values = np.meshgrid(x_grid, y_grid)

    potential = electric_potential(x_values, y_values)
    field_x, field_y, magnitude = electric_field_from_point_charge(x_values, y_values)

    print("Electric field from a point-charge electric potential")
    print(
        f"V(x, y) = kq / r, k = {COULOMB_FACTOR:g}, q = {POINT_CHARGE:g}, "
        f"field strength capped for r < {SINGULARITY_CUTOFF:g}"
    )
    print("E = kq r_hat / r^2, with the singular point left undefined")
    print(f"max |E| = {np.nanmax(magnitude):.6f}")
    print(f"|E|(x) sampled at y = {SAMPLE_Y}")

    plot_result(x_values, y_values, potential, field_x, field_y, magnitude, OUTPUT_PATH)
    print(f"Saved graph to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
