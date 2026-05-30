import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import e as 기본전하량
from scipy.constants import epsilon_0 as VACUUM_PERMITTIVITY
from scipy.constants import nano


SPHERE_AREA_FACTOR = 4.0 * np.pi
EPSILON_0 = VACUUM_PERMITTIVITY
CHARGES = ((기본전하량, -1.0, 0.0), (-기본전하량, 1.0, 0.0))
LENGTH_UNIT = "nm"
LENGTH_SCALE = nano

FIELD_X_RANGE = (-2.5, 2.5)
FIELD_Y_RANGE = (-2.0, 2.0)
FIELD_X_POINTS = 35
FIELD_Y_POINTS = 29
FIELD_CUTOFF = 0.18
FIELD_X_LIMIT = (-2.6, 2.6)
FIELD_Y_LIMIT = (-2.1, 2.1)
FIELD_COLOR_FLOOR = 1.0e-4

SAMPLE_POINTS = 500
SAMPLE_Y = 0.5
POTENTIAL_Y = 0.0
POTENTIAL_DIVERGENCE_GAP = 0.04
POTENTIAL_Y_LIMIT = (-0.45, 0.45)

LOCAL_GAUSSIAN_SURFACE_RADII = (0.45, 0.75)
ENCLOSING_SURFACE_CENTER = (0.0, 0.0)
ENCLOSING_SURFACE_RADIUS = 1.65
FLUX_RADIUS_RANGE = (0.25, 1.2)
FLUX_POINTS = 200
PRINT_RADII = np.array([0.5, 1.0, 1.5, 2.0])

FIGURE_SIZE = (13, 10)
OUTPUT_PATH = "example-9.png"


def to_si_length(values):
    return np.asarray(values) * LENGTH_SCALE


def charges_to_si(charges):
    return tuple(
        (charge, x_charge * LENGTH_SCALE, y_charge * LENGTH_SCALE)
        for charge, x_charge, y_charge in charges
    )


def electric_field_magnitude_from_gauss(radius, charge, epsilon_0):
    gaussian_area = SPHERE_AREA_FACTOR * radius**2
    return charge / (epsilon_0 * gaussian_area)


def electric_field_from_gauss(x_values, y_values, charges, epsilon_0, cutoff):
    field_x = np.zeros_like(x_values)
    field_y = np.zeros_like(y_values)

    for charge, x_charge, y_charge in charges:
        dx = x_values - x_charge
        dy = y_values - y_charge
        radius = np.sqrt(dx**2 + dy**2)
        safe_radius = np.maximum(radius, cutoff)
        radial_field = electric_field_magnitude_from_gauss(
            safe_radius, charge, epsilon_0
        )

        unit_x = np.divide(
            dx,
            safe_radius,
            out=np.zeros_like(dx),
            where=safe_radius > 0.0,
        )
        unit_y = np.divide(
            dy,
            safe_radius,
            out=np.zeros_like(dy),
            where=safe_radius > 0.0,
        )

        field_x += radial_field * unit_x
        field_y += radial_field * unit_y

    magnitude = np.sqrt(field_x**2 + field_y**2)
    return field_x, field_y, magnitude


def electric_potential(x_values, y_values, charges, epsilon_0):
    potential = np.zeros_like(x_values)

    for charge, x_charge, y_charge in charges:
        dx = x_values - x_charge
        dy = y_values - y_charge
        radius = np.sqrt(dx**2 + dy**2)
        with np.errstate(divide="ignore", invalid="ignore"):
            potential += charge / (SPHERE_AREA_FACTOR * epsilon_0 * radius)

    return potential


def gaussian_flux(radius, charge, epsilon_0):
    field = electric_field_magnitude_from_gauss(radius, charge, epsilon_0)
    area = SPHERE_AREA_FACTOR * radius**2
    return field * area


def plot_electric_field(charges, epsilon_0, output_path):
    x_grid = np.linspace(*FIELD_X_RANGE, FIELD_X_POINTS)
    y_grid = np.linspace(*FIELD_Y_RANGE, FIELD_Y_POINTS)
    x_values, y_values = np.meshgrid(x_grid, y_grid)
    x_values_si = to_si_length(x_values)
    y_values_si = to_si_length(y_values)
    charges_si = charges_to_si(charges)

    field_x, field_y, magnitude = electric_field_from_gauss(
        x_values_si,
        y_values_si,
        charges_si,
        epsilon_0,
        FIELD_CUTOFF * LENGTH_SCALE,
    )

    display_scale = np.sqrt(field_x**2 + field_y**2)
    direction_x = np.divide(
        field_x, display_scale, out=np.zeros_like(field_x), where=display_scale > 0.0
    )
    direction_y = np.divide(
        field_y, display_scale, out=np.zeros_like(field_y), where=display_scale > 0.0
    )

    sample_x = np.linspace(*FIELD_X_RANGE, SAMPLE_POINTS)
    sample_y = np.full_like(sample_x, SAMPLE_Y)
    _, _, sample_magnitude = electric_field_from_gauss(
        to_si_length(sample_x),
        to_si_length(sample_y),
        charges_si,
        epsilon_0,
        FIELD_CUTOFF * LENGTH_SCALE,
    )
    potential_x = np.linspace(*FIELD_X_RANGE, SAMPLE_POINTS)
    potential_y = np.full_like(potential_x, POTENTIAL_Y)
    potential_1d = electric_potential(
        to_si_length(potential_x), to_si_length(potential_y), charges_si, epsilon_0
    )
    for _charge, x_charge, _y_charge in charges:
        near_charge = np.abs(potential_x - x_charge) < POTENTIAL_DIVERGENCE_GAP
        potential_1d[near_charge] = np.nan

    radii = np.linspace(*FLUX_RADIUS_RANGE, FLUX_POINTS)
    radii_si = to_si_length(radii)

    fig, ((ax_field, ax_potential), (ax_strength, ax_flux)) = plt.subplots(
        2, 2, figsize=FIGURE_SIZE, constrained_layout=True
    )

    color = np.log10(np.clip(magnitude, FIELD_COLOR_FLOOR, None))
    ax_field.streamplot(
        x_grid,
        y_grid,
        field_x,
        field_y,
        color="0.45",
        density=1.2,
        linewidth=0.8,
        arrowsize=1.0,
        zorder=1,
    )
    field_plot = ax_field.quiver(
        x_values,
        y_values,
        direction_x,
        direction_y,
        color,
        cmap="viridis",
        pivot="middle",
        scale=28,
        width=0.004,
        zorder=2,
    )
    fig.colorbar(field_plot, ax=ax_field, label="log10 |E_total| (V/m)")

    for charge, x_charge, y_charge in charges:
        charge_color = "tab:red" if charge > 0.0 else "tab:blue"
        for radius in LOCAL_GAUSSIAN_SURFACE_RADII:
            gaussian_surface = plt.Circle(
                (x_charge, y_charge),
                radius,
                fill=False,
                linestyle="--",
                linewidth=1.2,
                color=charge_color,
                alpha=0.75,
            )
            ax_field.add_patch(gaussian_surface)
        ax_field.scatter(
            [x_charge],
            [y_charge],
            s=90,
            color=charge_color,
            zorder=4,
            label=f"q = {charge / 기본전하량:g} e",
        )

    enclosing_surface = plt.Circle(
        ENCLOSING_SURFACE_CENTER,
        ENCLOSING_SURFACE_RADIUS,
        fill=False,
        linestyle="-.",
        linewidth=1.6,
        color="black",
        alpha=0.85,
        label="encloses q1 + q2 = 0",
    )
    ax_field.add_patch(enclosing_surface)

    ax_field.set_title("Two Point-Charge Field")
    ax_field.set_xlabel(f"x ({LENGTH_UNIT})")
    ax_field.set_ylabel(f"y ({LENGTH_UNIT})")
    ax_field.set_aspect("equal")
    ax_field.set_xlim(*FIELD_X_LIMIT)
    ax_field.set_ylim(*FIELD_Y_LIMIT)
    ax_field.grid(True, alpha=0.25)
    ax_field.legend(loc="upper right")

    ax_potential.axhline(0.0, color="0.35", linewidth=1.0, alpha=0.7)
    for charge, x_charge, y_charge in charges:
        charge_color = "tab:red" if charge > 0.0 else "tab:blue"
        ax_potential.axvline(
            x_charge,
            color=charge_color,
            linestyle=":",
            linewidth=2.0,
            alpha=0.8,
            label=f"V diverges at q = {charge / 기본전하량:g} e",
        )
    ax_potential.plot(
        potential_x,
        potential_1d,
        linewidth=2.0,
        color="tab:orange",
        label=f"V(x, {POTENTIAL_Y:g} {LENGTH_UNIT})",
    )

    ax_potential.set_title(f"Electric Potential Along y = {POTENTIAL_Y:g} {LENGTH_UNIT}")
    ax_potential.set_xlabel(f"x ({LENGTH_UNIT})")
    ax_potential.set_ylabel("V (V)")
    ax_potential.set_ylim(*POTENTIAL_Y_LIMIT)
    ax_potential.grid(True, alpha=0.3)
    ax_potential.legend()

    ax_strength.plot(sample_x, sample_magnitude, linewidth=2.0, color="tab:purple")
    ax_strength.set_title(
        f"Total Field Magnitude Along y = {SAMPLE_Y:g} {LENGTH_UNIT}"
    )
    ax_strength.set_xlabel(f"x ({LENGTH_UNIT})")
    ax_strength.set_ylabel("|E_total| (V/m)")
    ax_strength.grid(True, alpha=0.3)

    for charge, _x_charge, _y_charge in charges:
        flux_values = gaussian_flux(radii_si, charge, epsilon_0)
        ax_flux.plot(
            radii,
            flux_values,
            linewidth=2.0,
            label=f"q = {charge / 기본전하량:g} e: E_i 4 pi r^2",
        )

    total_flux = sum(charge for charge, _x_charge, _y_charge in charges) / epsilon_0
    ax_flux.axhline(
        total_flux,
        color="black",
        linestyle="-.",
        linewidth=1.5,
        label="enclosing surface: q_total / epsilon_0",
    )
    ax_flux.set_title("Flux for Gaussian Surfaces")
    ax_flux.set_xlabel(f"Gaussian radius r ({LENGTH_UNIT})")
    ax_flux.set_ylabel("electric flux (N m^2/C)")
    ax_flux.grid(True, alpha=0.3)
    ax_flux.legend()

    fig.suptitle("Two Point Charges Using Gauss Law and Superposition")
    fig.savefig(output_path, dpi=200)


def main():
    print("Two point charges electric field from Gauss law")
    print("Gauss law: integral(E dot dA) = q / epsilon_0")
    print("For each charge: E_i(r) * 4 pi r^2 = q_i / epsilon_0")
    print("Total field: E_total = E_1 + E_2")
    print(f"q1 = +e, q2 = -e, e = {기본전하량:.10e} C")
    print(f"epsilon_0 = {EPSILON_0:.10e} F/m")
    print(f"plot length unit = {LENGTH_UNIT}, SI length scale = {LENGTH_SCALE:.10e} m")
    print("-" * 106)
    print(
        f"{'charge':>8} {f'position ({LENGTH_UNIT})':>18}"
        f" {f'r ({LENGTH_UNIT})':>10} {'E_i(r) (V/m)':>18}"
        f" {'4 pi r^2 E_i(r)':>22} {'q_i / epsilon_0':>18}"
    )
    print("-" * 106)

    for charge, x_charge, y_charge in CHARGES:
        expected_flux = charge / EPSILON_0
        for radius in PRINT_RADII:
            radius_si = radius * LENGTH_SCALE
            field = electric_field_magnitude_from_gauss(radius_si, charge, EPSILON_0)
            flux = gaussian_flux(radius_si, charge, EPSILON_0)
            print(
                f"{charge / 기본전하량:8.3f}e"
                f" {f'({x_charge:g}, {y_charge:g})':>18}"
                f" {radius:10.3f} {field:18.10e} {flux:22.10e}"
                f" {expected_flux:18.10e}"
            )

    total_charge = sum(charge for charge, _x_charge, _y_charge in CHARGES)
    print(
        "\nFlux through surface enclosing both charges = "
        f"{total_charge / EPSILON_0:.10e}"
    )

    plot_electric_field(CHARGES, EPSILON_0, OUTPUT_PATH)
    print(f"Saved graph to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
