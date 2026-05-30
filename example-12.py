import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


h = 0.1
MAX = 5000
Error = 1.0e-8
CHARGE = 1.0
EPSILON_0 = 1.0 / (4.0 * np.pi)
OUTPUT_PATH = "example-12.png"

x = np.arange(-1, 1 + 0.5 * h, h)
y = np.arange(-1, 1 + 0.5 * h, h)
X, Y = np.meshgrid(x, y)

Nx = np.size(x)
Ny = np.size(y)
phi = np.ones((Nx, Ny))
phi *= 0.1
phi[0, :] = phi[-1, :] = phi[:, 0] = phi[:, -1] = 0.0

rho = np.zeros((Nx, Ny))
rho[Nx // 2, Ny // 2] = CHARGE / h**2

for i in range(MAX):
    phi0 = phi.copy()

    for nx in range(1, Nx - 1):
        for ny in range(1, Ny - 1):
            phi[nx, ny] = (
                phi0[nx - 1, ny]
                + phi0[nx + 1, ny]
                + phi0[nx, ny - 1]
                + phi0[nx, ny + 1]
            ) / 4.0 + h**2 * rho[nx, ny] / (4.0 * EPSILON_0)

    if np.max(np.abs(phi - phi0)) < Error:
        break

fig = plt.figure(figsize=(10, 4.5), constrained_layout=True)

ax1 = fig.add_subplot(1, 2, 1)
ax1.contour(
    X,
    Y,
    phi.T,
    levels=np.linspace(np.min(phi), np.max(phi), 100),
)
ax1.set_title("Potential of a Point Charge")
ax1.set_xlabel("x")
ax1.set_ylabel("y")
ax1.set_aspect("equal")

ax2 = fig.add_subplot(1, 2, 2, projection="3d")
ax2.plot_surface(X, Y, phi.T)
ax2.set_title("Relaxation Solution")
ax2.set_xlabel("x")
ax2.set_ylabel("y")
ax2.set_zlabel("phi")

fig.savefig(OUTPUT_PATH, dpi=200)
print(f"iteration = {i + 1}")
print(f"max error = {np.max(np.abs(phi - phi0)):.6e}")
print(f"phi(0, 0) = {phi[Nx // 2, Ny // 2]:.6f}")
print(f"Saved graph to {OUTPUT_PATH}")
