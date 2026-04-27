import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


GM_SUN = 4.0 * np.pi**2


def acceleration(position):
    r = np.linalg.norm(position)
    return -GM_SUN * position / r**3


def leapfrog_orbit(speed_factor, dt, t_end, escape_radius=None):
    position = np.array([1.0, 0.0], dtype=float)
    velocity = np.array([0.0, speed_factor * 2.0 * np.pi], dtype=float)
    velocity_half = velocity + 0.5 * dt * acceleration(position)

    times = [0.0]
    positions = [position.copy()]
    speeds = [np.linalg.norm(velocity)]

    n_steps = int(np.ceil(t_end / dt))
    for step in range(1, n_steps + 1):
        position = position + dt * velocity_half
        accel = acceleration(position)
        velocity_half = velocity_half + dt * accel
        velocity_full = velocity_half - 0.5 * dt * accel
        t = step * dt

        times.append(t)
        positions.append(position.copy())
        speeds.append(np.linalg.norm(velocity_full))

        if escape_radius is not None and np.linalg.norm(position) >= escape_radius:
            break

    return np.array(times), np.array(positions), np.array(speeds)


def theoretical_orbit(speed_factor):
    eccentricity = speed_factor**2 - 1.0

    if speed_factor < np.sqrt(2.0):
        semi_major_axis = 1.0 / (2.0 - speed_factor**2)
        period = semi_major_axis**1.5
        aphelion = semi_major_axis * (1.0 + eccentricity)
        return "bound", semi_major_axis, eccentricity, aphelion, period

    return "escape", np.inf, eccentricity, np.inf, np.inf


def set_equal_axis(ax, x, y):
    x_min, x_max = np.min(x), np.max(x)
    y_min, y_max = np.min(y), np.max(y)
    center_x = 0.5 * (x_min + x_max)
    center_y = 0.5 * (y_min + y_max)
    radius = 0.55 * max(x_max - x_min, y_max - y_min, 1.0)

    ax.set_xlim(center_x - radius, center_x + radius)
    ax.set_ylim(center_y - radius, center_y + radius)
    ax.set_aspect("equal", adjustable="box")


def main():
    speed_factors = [1.1, 1.2, 1.3, 1.4, 1.5]
    dt = 0.001
    escape_radius = 55.0

    print("Earth orbit around the Sun using the leapfrog method")
    print("units: distance = AU, time = year, GM_sun = 4*pi^2")
    print(f"initial position = (1 AU, 0), circular speed = 2*pi AU/year, dt = {dt}")
    print("-" * 98)
    print(
        f"{'v/v_c':>7} {'type':>8} {'a (AU)':>12} {'e':>10}"
        f" {'r_max theory (AU)':>18} {'simulated years':>16} {'r_max sim (AU)':>16}"
    )
    print("-" * 98)

    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    axes = axes.ravel()

    for ax, factor in zip(axes, speed_factors):
        orbit_type, semi_major_axis, eccentricity, aphelion, period = theoretical_orbit(
            factor
        )

        if orbit_type == "bound":
            t_end = 1.02 * period
            stop_radius = None
        else:
            t_end = 25.0
            stop_radius = escape_radius

        times, positions, speeds = leapfrog_orbit(factor, dt, t_end, stop_radius)
        x = positions[:, 0]
        y = positions[:, 1]
        r = np.linalg.norm(positions, axis=1)

        print(
            f"{factor:7.1f} {orbit_type:>8} {semi_major_axis:12.4f}"
            f" {eccentricity:10.4f} {aphelion:18.4f}"
            f" {times[-1]:16.4f} {np.max(r):16.4f}"
        )

        ax.plot(x, y, linewidth=1.6, label="leapfrog orbit")
        ax.scatter([0.0], [0.0], s=80, color="gold", edgecolor="black", label="Sun")
        ax.scatter([1.0], [0.0], s=30, color="tab:blue", label="start")
        ax.set_title(f"initial speed = {factor:.1f} $v_c$ ({orbit_type})")
        ax.set_xlabel("x (AU)")
        ax.set_ylabel("y (AU)")
        ax.grid(True, alpha=0.3)
        set_equal_axis(ax, x, y)

    axes[-1].axis("off")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3)
    fig.suptitle("Earth Orbit after an Instantaneous Speed Increase", fontsize=16)
    fig.tight_layout(rect=[0.0, 0.05, 1.0, 0.96])
    fig.savefig("example-4.png", dpi=200)
    print("\nSaved graph to example-4.png")


if __name__ == "__main__":
    main()
