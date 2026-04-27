import numpy as np
import matplotlib.pyplot as plt


def analytic_trajectory(t, v0, theta, g):
    vx0 = v0 * np.cos(theta)
    vy0 = v0 * np.sin(theta)
    x = vx0 * t
    y = vy0 * t - 0.5 * g * t**2
    return x, y


def euler_projectile(v0, theta, dt, g):
    x = [0.0]
    y = [0.0]
    vx = v0 * np.cos(theta)
    vy = v0 * np.sin(theta)
    t = [0.0]

    while True:
        x_next = x[-1] + vx * dt
        y_next = y[-1] + vy * dt
        vy_next = vy - g * dt
        t_next = t[-1] + dt

        if y_next < 0.0:
            frac = y[-1] / (y[-1] - y_next)
            x.append(x[-1] + frac * (x_next - x[-1]))
            y.append(0.0)
            t.append(t[-1] + frac * dt)
            break

        x.append(x_next)
        y.append(y_next)
        t.append(t_next)
        vy = vy_next

    return np.array(t), np.array(x), np.array(y)


def modified_euler_projectile(v0, theta, dt, g):
    x = [0.0]
    y = [0.0]
    vx = v0 * np.cos(theta)
    vy = v0 * np.sin(theta)
    t = [0.0]

    while True:
        vx_next = vx
        vy_next = vy - g * dt
        x_next = x[-1] + 0.5 * (vx + vx_next) * dt
        y_next = y[-1] + 0.5 * (vy + vy_next) * dt
        t_next = t[-1] + dt

        if y_next < 0.0:
            frac = y[-1] / (y[-1] - y_next)
            x.append(x[-1] + frac * (x_next - x[-1]))
            y.append(0.0)
            t.append(t[-1] + frac * dt)
            break

        x.append(x_next)
        y.append(y_next)
        t.append(t_next)
        vx = vx_next
        vy = vy_next

    return np.array(t), np.array(x), np.array(y)


def leapfrog_projectile(v0, theta, dt, g):
    x = [0.0]
    y = [0.0]
    vx_half = v0 * np.cos(theta)
    vy_half = v0 * np.sin(theta) - 0.5 * g * dt
    t = [0.0]

    while True:
        x_next = x[-1] + vx_half * dt
        y_next = y[-1] + vy_half * dt
        t_next = t[-1] + dt

        if y_next < 0.0:
            frac = y[-1] / (y[-1] - y_next)
            x.append(x[-1] + frac * (x_next - x[-1]))
            y.append(0.0)
            t.append(t[-1] + frac * dt)
            break

        x.append(x_next)
        y.append(y_next)
        t.append(t_next)
        vy_half = vy_half - g * dt

    return np.array(t), np.array(x), np.array(y)


def main():
    g = 9.8
    v0 = 10.0
    theta_deg = 45.0
    theta = np.deg2rad(theta_deg)
    dt = 0.05

    time_of_flight = 2.0 * v0 * np.sin(theta) / g
    analytic_range = v0**2 * np.sin(2.0 * theta) / g
    max_height = (v0 * np.sin(theta)) ** 2 / (2.0 * g)

    t_exact = np.linspace(0.0, time_of_flight, 400)
    x_exact, y_exact = analytic_trajectory(t_exact, v0, theta, g)

    t_euler, x_euler, y_euler = euler_projectile(v0, theta, dt, g)
    t_mod, x_mod, y_mod = modified_euler_projectile(v0, theta, dt, g)
    t_leap, x_leap, y_leap = leapfrog_projectile(v0, theta, dt, g)

    print("Projectile motion without air resistance")
    print(f"v0 = {v0} m/s, theta = {theta_deg} deg, g = {g} m/s^2, dt = {dt} s")
    print("-" * 72)
    print(f"{'method':>18} {'time of flight (s)':>20} {'range (m)':>14} {'range error (m)':>16}")
    print("-" * 72)
    print(f"{'analytic':>18} {time_of_flight:20.6f} {analytic_range:14.6f} {0.0:16.6f}")
    print(
        f"{'euler':>18} {t_euler[-1]:20.6f} {x_euler[-1]:14.6f}"
        f" {x_euler[-1] - analytic_range:16.6f}"
    )
    print(
        f"{'modified euler':>18} {t_mod[-1]:20.6f} {x_mod[-1]:14.6f}"
        f" {x_mod[-1] - analytic_range:16.6f}"
    )
    print(
        f"{'leap-frog':>18} {t_leap[-1]:20.6f} {x_leap[-1]:14.6f}"
        f" {x_leap[-1] - analytic_range:16.6f}"
    )
    print("-" * 72)
    print(f"analytic maximum height = {max_height:.6f} m")

    plt.figure(figsize=(10, 6))
    plt.plot(x_exact, y_exact, linewidth=2.0, label="analytic solution")
    plt.plot(
        x_euler,
        y_euler,
        marker="o",
        markersize=4,
        linewidth=1.4,
        label="Euler method",
    )
    plt.plot(
        x_mod,
        y_mod,
        marker="s",
        markersize=4,
        linewidth=1.4,
        label="modified Euler method",
    )
    plt.plot(
        x_leap,
        y_leap,
        marker="^",
        markersize=4,
        linewidth=1.4,
        label="leap-frog method",
    )
    plt.title("Projectile Motion without Air Resistance (theta = 45 deg)")
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.axis("equal")
    plt.tight_layout()
    plt.savefig("example-3.png", dpi=200)
    print("\nSaved graph to example-3.png")
    plt.show()


if __name__ == "__main__":
    main()
