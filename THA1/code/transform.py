"""A.I.2 -- Data transformation: Z-score standardization."""
from pathlib import Path
import numpy as np
from data import PowerPlantData, ROOT


class ZScore:
    """v -> (v - mu) / sigma, with mu, sigma fitted on the data (Eq. eq:A2_zscore)."""

    def fit(self, v):
        self.mu    = v.mean()                                   # sample mean
        self.sigma = v.std()                                    # population std (ddof = 0)
        return self

    def transform(self, v):
        return (v - self.mu) / self.sigma                       # standardized values

    def inverse(self, z):
        return self.mu + self.sigma * z                         # back to physical units


def to_latex(d, xs, ys, tx, ty, path):
    """Write raw and standardized data as a LaTeX tabular."""
    head = [r"\begin{tabular}{rrrrr}", r"\toprule",
            r"$i$ & $x^{(i)}$ [$^\circ$C] & $y^{(i)}$ [MW] & $\tilde x^{(i)}$ & $\tilde y^{(i)}$ \\", r"\midrule"]
    body = [rf"{i} & {x:.2f} & {y:.2f} & ${a:+.4f}$ & ${b:+.4f}$ \\"
            for i, (x, y, a, b) in enumerate(zip(d.x, d.y, xs, ys), start=1)]
    stats = [r"\midrule",
             rf"$\mu$ & {tx.mu:.4f} & {ty.mu:.4f} & & \\",
             rf"$\sigma$ & {tx.sigma:.4f} & {ty.sigma:.4f} & & \\"]
    tail = [r"\bottomrule", r"\end{tabular}"]
    Path(path).parent.mkdir(exist_ok=True)
    Path(path).write_text("\n".join(head + body + stats + tail) + "\n")


if __name__ == "__main__":
    d  = PowerPlantData(n=1)
    tx = ZScore().fit(d.x)                                      # scaler for inputs
    ty = ZScore().fit(d.y)                                      # scaler for outputs
    xs, ys = tx.transform(d.x), ty.transform(d.y)
    print(f"x: mu = {tx.mu:.4f}, sigma = {tx.sigma:.4f}")
    print(f"y: mu = {ty.mu:.4f}, sigma = {ty.sigma:.4f}")
    print(f"check: mean(xs) = {xs.mean():.1e}, std(xs) = {xs.std():.4f}")
    to_latex(d, xs, ys, tx, ty, ROOT / "tables" / "A2_zscore.tex")
