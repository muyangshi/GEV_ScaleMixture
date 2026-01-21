"""
plot-co2.py

Purpose
-------
Illustrate why calendar time (t) is used as the covariate in the marginal GEV
model, rather than greenhouse-gas concentrations or nonlinear functions of CO2.

Description
-----------
This script plots CO2-related quantities against time and demonstrates their
near-linear relationship with calendar year over the study period. Specifically,
it shows that:
- log(CO2 concentration),
- radiative forcing from CO2, and
- total greenhouse-gas radiative forcing
are all highly correlated with time.

Outputs
-------
- Fig_co2_time_trends_3panel.pdf: three-panel plot of CO2-related quantities versus
  year with linear fits and correlation coefficients.
"""
# %%
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

# ---- Load ----
here = Path(__file__).resolve().parent
csv_path = here / "co2_Data.csv"
df = pd.read_csv(csv_path)
# %%
# ---- 3-panel CO2 vs time with linear fit & r ----
# plt.style.use("ggplot")

# keep 1949–2023 window
df = df[df["year"].between(1949, 2023)].copy()

x = df["year"].to_numpy()
panels = [
    ("log CO2 concentrations", df["log_co2_ppm"].to_numpy(), "log concentration"),
    ("radiative forcing from CO2", df["co2_forcing"].to_numpy(), r"W m$^{-2}$"),
    ("sum-total radiative forcing, all GHGs", df["WMGHG"].to_numpy(), r"W m$^{-2}$"),
]

fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=300, constrained_layout=True)

for ax, (title, y, yunit) in zip(axes, panels):
    m = np.isfinite(x) & np.isfinite(y)
    xm, ym = x[m], y[m]

    # correlation and least-squares line
    r = np.corrcoef(xm, ym)[0, 1]
    b1, b0 = np.polyfit(xm, ym, 1)
    xx = np.linspace(xm.min(), xm.max(), 400)
    yy = b1 * xx + b0

    # series (black) + fit (blue)
    ax.plot(xm, ym, color="black", linewidth=1)
    ax.plot(xx, yy, linewidth=2)

    ax.set_xlabel("Year", fontsize=16)
    ax.tick_params(labelsize=14)        # <-- tick font size
    ax.grid(True, alpha=0.35)
    ax.set_title(
        f"{title}\nCorrelation with year = {r:.4f}",
        fontsize=16,
        # bbox=dict(facecolor="#cccccc", edgecolor="none", pad=1.0),
    )

# # one shared y label like the example
# fig.text(0.04, 0.5, "log concentration / W m$^{-2}$", va="center",
#          rotation="vertical", fontsize=16)

# # shared y-label, nudged left so it doesn't sit on the axis
# fig.supylabel("log concentration / W m$^{-2}$",
#               x=0.02, rotation="vertical", va="center", fontsize=16)

out = here / "Fig_co2_time_trends_3panel.pdf"
fig.savefig(out)
print(f"Saved: {out}")
plt.show()
# %%
