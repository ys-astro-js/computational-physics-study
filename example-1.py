import numpy as np
import matplotlib.pyplot as plt


def analytic_velocity(t, g, gamma):
    """Analytic solution for dv/dt = g - gamma * v, v(0) = 0."""
    return (g / gamma) * (1.0 - np.exp(-gamma * t))


def euler_velocity(t_end, dt, g, gamma):
    t = np.arange(0.0, t_end + dt, dt)
    v = np.zeros_like(t)

    for i in range(len(t) - 1):
        v[i + 1] = v[i] + dt * (g - gamma * v[i])

    return t, v


def main():
    g = 9.8
    gamma = 0.5
    t_end = 10.0
    dt_values = np.array([1.0, 0.5, 0.1, 0.05])

    print("Euler method error for free fall with fluid resistance")
    print(f"g = {g}, gamma = {gamma}, t_end = {t_end}")
    print("-" * 60)
    print(f"{'dt':>8} {'max abs error':>18} {'final error':>18}")
    print("-" * 60)

    plt.figure(figsize=(10, 6))

    for dt in dt_values:
        t, v_euler = euler_velocity(t_end, dt, g, gamma)
        v_exact = analytic_velocity(t, g, gamma)
        error = np.abs(v_exact - v_euler)

        print(f"{dt:8.3f} {np.max(error):18.8f} {error[-1]:18.8f}")
        plt.plot(t, error, marker="o", markersize=3, linewidth=1.2, label=f"dt={dt}")

    plt.title("Euler Method Error for Free Fall with Fluid Resistance")
    plt.xlabel("time t (s)")
    plt.ylabel("absolute error |v_exact - v_Euler| (m/s)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("example-1.png", dpi=200)
    print("\nSaved graph to example-1.png")


if __name__ == "__main__":
    main()
