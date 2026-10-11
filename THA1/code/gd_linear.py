"""A.I.5-7 -- Run gradient descent on the standardized data: first, second and last iteration."""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from data import PowerPlantData, ROOT
from transform import ZScore
from least_squares import LeastSquares, to_physical
from gradient_descent import LinearGD

SEED, ALPHA = 0, 0.1                                            # reproducible random start, learning rate
N_MAX, TOL  = 1000, 1e-6                                        # max iterations, stop when |grad J| < TOL


def fmt(v):
    return f"{v:+.4f}"


def table_first_two(gd, x, y, path):
    """Values of A.I.5-6: theta^(t), grad J(theta^(t)), theta^(t+1), J for t = 0, 1."""
    rows = []
    for t in (0, 1):
        th, J = gd.history[t]
        g = LinearGD(th, gd.alpha).gradient(x, y)               # gradient at theta^(t)
        th1, J1 = gd.history[t + 1]
        rows.append(rf"{t+1} & ${fmt(th[0])}$ & ${fmt(th[1])}$ & ${fmt(g[0])}$ & ${fmt(g[1])}$ "
                    rf"& ${fmt(th1[0])}$ & ${fmt(th1[1])}$ & {J:.4f} & {J1:.4f} \\")
    head = [r"\begin{tabular}{c rr rr rr cc}", r"\toprule",
            r"$t+1$ & $\theta_0^{(t)}$ & $\theta_1^{(t)}$ & $\partial_{\theta_0}J$ & $\partial_{\theta_1}J$ "
            r"& $\theta_0^{(t+1)}$ & $\theta_1^{(t+1)}$ & $J(\bth^{(t)})$ & $J(\bth^{(t+1)})$ \\", r"\midrule"]
    Path(path).write_text("\n".join(head + rows + [r"\bottomrule", r"\end{tabular}"]) + "\n")


def plot_fits(x, y, gd, ls, path):
    """Regression line after iteration 1, 2 and the last one, with the least-squares line for reference."""
    T = len(gd.history) - 1
    fig, ax = plt.subplots(1, 3, figsize=(11, 3.4), sharey=True)
    g = np.linspace(x.min(), x.max(), 2)
    for a, t in zip(ax, (1, 2, T)):
        th = gd.history[t][0]
        a.scatter(x, y, s=20, color="0.25", zorder=3, label="data")
        a.plot(g, th[0] + th[1] * g, color="C0", lw=2, label="gradient descent")
        a.plot(g, ls.predict(g), color="0.5", lw=1, ls="--", label="least squares")
        a.set(title=f"iteration {t}", xlabel=r"$\tilde x$")
        a.grid(alpha=0.3)
    ax[0].set_ylabel(r"$\tilde y$")
    ax[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path)


def plot_cost(gd, J_ls, path):
    """J(theta^(t)) for all iterations, log scale, with the least-squares minimum."""
    J = np.array([c for _, c in gd.history])
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
    ax[0].plot(J, color="C0", lw=2)
    ax[0].axhline(J_ls, color="0.5", ls="--", lw=1, label=r"$J$ at least squares")
    ax[0].set(xlabel="iteration $t$", ylabel=r"$J(\theta^{(t)})$")
    ax[0].legend(frameon=False)
    ax[1].semilogy(J - J_ls, color="C0", lw=2)                  # excess cost, decays geometrically
    ax[1].set(xlabel="iteration $t$", ylabel=r"$J(\theta^{(t)}) - J_{\min}$")
    for a in ax:
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path)


if __name__ == "__main__":
    d  = PowerPlantData(n=1)
    tx, ty = ZScore().fit(d.x), ZScore().fit(d.y)
    xs, ys = tx.transform(d.x), ty.transform(d.y)
    ls = LeastSquares().fit(xs, ys)
    J_ls = LinearGD(ls.theta, 0).cost(xs, ys)                   # minimum of J

    theta0 = np.random.default_rng(SEED).uniform(-1, 1, 2)      # A.I.5: random initial parameters
    gd = LinearGD(theta0, ALPHA).run(xs, ys, N_MAX, TOL)
    T  = len(gd.history) - 1
    th = gd.theta
    ph = to_physical(th, tx, ty)
    print(f"theta^(0) = {theta0},  alpha = {ALPHA}")
    for t in (1, 2):
        print(f"iteration {t}: theta = {gd.history[t][0]},  J = {gd.history[t][1]:.6f}")
    print(f"last iteration {T}: theta = {th},  J = {gd.history[-1][1]:.8f}  (LS: {ls.theta}, J = {J_ls:.8f})")
    print(f"physical: theta_0 = {ph[0]:.4f} MW, theta_1 = {ph[1]:.4f} MW/C")

    (ROOT / "figs").mkdir(exist_ok=True)
    (ROOT / "tables").mkdir(exist_ok=True)
    table_first_two(gd, xs, ys, ROOT / "tables" / "A5_iterations.tex")
    (ROOT / "tables" / "A5_values.tex").write_text(             # numbers quoted in the report
        rf"\def\GDalpha{{{ALPHA}}}\def\GDseed{{{SEED}}}\def\GDtol{{{TOL:.0e}}}\def\GDT{{{T}}}"
        rf"\def\GDthOinit{{{theta0[0]:.4f}}}\def\GDthIinit{{{theta0[1]:.4f}}}"
        rf"\def\GDthO{{{th[0]:.1e}}}\def\GDthI{{{th[1]:.6f}}}\def\GDJ{{{gd.history[-1][1]:.6f}}}"
        rf"\def\GDphO{{{ph[0]:.2f}}}\def\GDphI{{{ph[1]:.4f}}}"
        rf"\def\LSthIsix{{{ls.theta[1]:.6f}}}\def\LSJ{{{J_ls:.6f}}}"
        rf"\def\GDdiff{{{np.abs(th - ls.theta).max():.1e}}}" + "\n")
    plot_fits(xs, ys, gd, ls, ROOT / "figs" / "A5_fits.pdf")
    plot_cost(gd, J_ls, ROOT / "figs" / "A7_cost.pdf")
