"""A.I.1 -- Data acquisition: load 20 points."""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT  = Path(__file__).resolve().parents[1]                    # ML-IntelligentSystems/
EXCEL = ROOT / "data" / "Data Take Home Assignment 1 Exercise A.xlsx"


class PowerPlantData:
    """Temperature X [C] -> net hourly electrical output Y [MW] for group n."""

    def __init__(self, n=1, size=20):
        """
        I adapted the start and end size according to panda convention where
        the first row of the excel file is the header. So, the first row of data is at 
        index 0, and the last row of data is at index 19.
        df.iloc[0] is the first row of the dataframe, and df.iloc[0:20] is the first 20 rows.
        """
        self.n, self.size = n, size
        start = (n - 1) * size                                  # 0-based index of first point
        df  = pd.read_excel(EXCEL)                              # Excel row 1 = header
        sub = df.iloc[start : start + size]                     # exactly `size` rows
        self.rows = sub.index.to_numpy() + 2                    # Excel row = index + 2 (header, 1-based)
        self.x = sub["X"].to_numpy(float)                       # inputs  x^(i)
        self.y = sub["Y"].to_numpy(float)                       # outputs y^(i)

    def to_latex(self, path):
        """Write the data as a LaTeX tabular (Excel row, i, x, y)."""
        head = [r"\begin{tabular}{rrrr}", r"\toprule",
                r"Excel row & $i$ & $x^{(i)}$ [$^\circ$C] & $y^{(i)}$ [MW] \\", r"\midrule"]
        body = [rf"{r} & {i} & {x:.2f} & {y:.2f} \\"
                for i, (r, x, y) in enumerate(zip(self.rows, self.x, self.y), start=1)]
        tail = [r"\bottomrule", r"\end{tabular}"]
        Path(path).parent.mkdir(exist_ok=True)
        Path(path).write_text("\n".join(head + body + tail) + "\n")


if __name__ == "__main__":
    d = PowerPlantData(n=1)
    print(f"Excel rows {d.rows[0]}-{d.rows[-1]}, {len(d.x)} points")
    print(np.column_stack([d.rows, d.x, d.y]))
    d.to_latex(ROOT / "tables" / "A1_data.tex")
