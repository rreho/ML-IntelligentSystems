"""A.I.3 -- Least squares: closed-form linear regression."""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from data import PowerPlantData, ROOT
from transform import ZScore


class LeastSquares:
    """h(x) = theta_0 + theta_1 x, fitted by the normal equations."""

    def fit(self, x, y):
        X = np.column_stack([np.ones_like(x), x])              # design matrix [1, x_i], shape (m, 2)
        self.theta = np.linalg.solve(X.T @ X, X.T @ y)          # (X^T X) theta = X^T y
        return self

    def predict(self, x):
        return self.theta[0] + self.theta[1] * x                # h_theta(x)


def to_physical(theta, tx, ty):
    """Map (theta_0, theta_1) fitted on z-scores back to MW and C (Eq. eq:A3_phys)."""
    t1 = ty.sigma / tx.sigma * theta[1]                         # slope    [MW / C]
    t0 = ty.mu + ty.sigma * theta[0] - t1 * tx.mu               # intercept [MW]
    return np.array([t0, t1])


def plot(d, xs, ys, model, phys, path):
    """Data + regression line, standardized (left) and physical units (right)."""
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.6))
    for a, (u, v, th, xl, yl) in zip(ax, [(xs, ys, model.theta, r"$\tilde x$", r"$\tilde y$"),
                                          (d.x, d.y, phys, r"$x$ [$^\circ$C]", r"$y$ [MW]")]):
        g = np.linspace(u.min(), u.max(), 2)                    # two points define the line
        a.scatter(u, v, s=25, color="0.25", label="data", zorder=3)
        a.plot(g, th[0] + th[1] * g, color="C0", lw=2, label="least squares")
        a.set(xlabel=xl, ylabel=yl)
        a.grid(alpha=0.3)
    ax[0].legend(frameon=False)
    fig.tight_layout()
    Path(path).parent.mkdir(exist_ok=True)
    fig.savefig(path)


if __name__ == "__main__":
    d  = PowerPlantData(n=1)
    tx, ty = ZScore().fit(d.x), ZScore().fit(d.y)
    xs, ys = tx.transform(d.x), ty.transform(d.y)
    model = LeastSquares().fit(xs, ys)
    phys  = to_physical(model.theta, tx, ty)
    print(f"standardized: theta_0 = {model.theta[0]:+.6f}, theta_1 = {model.theta[1]:+.6f}")
    print(f"physical    : theta_0 = {phys[0]:.4f} MW, theta_1 = {phys[1]:.4f} MW/C")
    Path(ROOT / "tables").mkdir(exist_ok=True)
    (ROOT / "tables" / "A3_theta.tex").write_text(              # numbers quoted in the report
        rf"\def\LSthO{{{model.theta[0]:.1e}}}\def\LSthI{{{model.theta[1]:.4f}}}"
        rf"\def\LSphO{{{phys[0]:.2f}}}\def\LSphI{{{phys[1]:.4f}}}" + "\n")
    plot(d, xs, ys, model, phys, ROOT / "figs" / "A3_least_squares.pdf")
