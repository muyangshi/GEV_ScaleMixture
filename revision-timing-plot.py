# %%
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

csv_path = Path(__file__).resolve().parent / "blockMax_JJA_centralUS_date.csv"
df = pd.read_csv(csv_path)

# First column is just site/index — drop it from the data block
data = df.drop(columns=df.columns[0])

# Parse everything as datetimes (robust to mixed formats); NaNs become NaT
data = data.apply(pd.to_datetime, errors="coerce")

# If your cells include the year (they probably do), it's fine;
# within a given column all entries are already the same year.
# Count unique dates per year (ignore missing)
unique_counts = data.apply(lambda s: s.dropna().nunique())

# Optional: put real years on the index if columns are year-like
try:
    unique_counts.index = pd.Index(range(1949, 1949 + data.shape[1]), name="year", dtype=int)
except Exception:
    pass

print("Unique JJA-peak dates per year:\n", unique_counts.describe())

# --- Histogram across years ---
plt.figure(figsize=(6.5, 4), dpi=300)
# # of unique days in JJA can't exceed ~92; use tight integer bins
max_u = int(unique_counts.max()) if len(unique_counts) else 1
min_u = int(unique_counts.min())
plt.hist(unique_counts.values, bins=range(min_u, max_u + 2))
plt.xlabel("Number of unique dates of JJA maxima (per year)", fontsize=16)
plt.ylabel("Number of years", fontsize=20)
plt.title("Distribution across years", fontsize=24)
plt.xticks(fontsize=20)   # x-axis tick label size
plt.yticks(fontsize=20)   # y-axis tick label size
plt.tight_layout()
plt.savefig(Path(__file__).resolve().parent / "Fig_unique_peak_dates_hist.pdf")
plt.show()

# # --- (Nice to have) per-year line to see trends ---
# plt.figure(figsize=(8.125, 5), dpi=300)
# plt.plot(unique_counts.index, unique_counts.values, marker="o", linewidth=1)
# plt.ylabel("# unique dates (within JJA)", fontsize=16)
# plt.xlabel("Year", fontsize = 16)
# plt.xticks(fontsize=14)   # x-axis tick label size
# plt.yticks(fontsize=14)   # y-axis tick label size
# plt.title("Per-year count of unique JJA peak dates across 590 sites", fontsize = 20)
# plt.tight_layout()
# plt.savefig(Path(__file__).resolve().parent / "Fig_unique_peak_dates_by_year.pdf")
# plt.show()
# %%
import rpy2.robjects as robjects
from rpy2.robjects import r 
from rpy2.robjects.numpy2ri import numpy2rpy
from rpy2.robjects.packages import importr
r('''load('JJA_precip_maxima_nonimputed.RData')''')
JJA_maxima         = np.array(r('JJA_maxima_nonimputed'))
# %% Plot by percentile

# --- Make a values DataFrame aligned to the dates DataFrame ---
# JJA_maxima should be (n_sites, n_years). If not, try transposing.
vals_np = JJA_maxima
if vals_np.shape != data.shape and vals_np.T.shape == data.shape:
    vals_np = vals_np.T
assert vals_np.shape == data.shape, f"Shape mismatch: values {vals_np.shape} vs dates {data.shape}"

vals = pd.DataFrame(vals_np, index=data.index, columns=data.columns)

# Years on x-axis (update start year if different)
years = pd.Index(range(1949, 1949 + data.shape[1]), name="year", dtype=int)

def count_unique_per_year(dates_df, mask_df=None):
    """Count unique calendar dates in each year (column), optionally masked."""
    used = dates_df if mask_df is None else dates_df.where(mask_df)
    out = used.apply(lambda s: s.dropna().nunique())  # column-wise
    out.index = years
    return out

# Baseline: all data
series_all = count_unique_per_year(data)

# Choose upper-percentile cutoffs (per-site, across years)
percentiles = [0.80, 0.90, 0.95, 0.99]

# Row-wise thresholds: one threshold per site
thr = {q: np.nanquantile(vals.to_numpy(), q, axis=1) for q in percentiles}

# Build masks and compute unique-date counts per year
series_by_q = {}
for q in percentiles:
    # keep (site,year) when value >= that site's q-quantile
    thr_row = pd.Series(thr[q], index=vals.index)
    mask_q = vals.ge(thr_row.to_numpy()[:, None])
    series_by_q[q] = count_unique_per_year(data, mask_q)

# --- Plot ---
plt.figure(figsize=(12, 7), dpi=300)
plt.plot(series_all.index, series_all.values, label="All data", linewidth=2.2, color="#d62728")

color_map = {0.80: "#1f77b4", 0.90: "#2ca02c", 0.95: "#9467bd", 0.99: "#ff7f0e"}
for q in percentiles:
    plt.plot(series_by_q[q].index, series_by_q[q].values,
             label=f">={int(q*100)}th %ile", linewidth=1.8, color=color_map[q])

plt.xlabel("Year", fontsize=24)
plt.ylabel("Number of unique days", fontsize=24)
plt.xticks(fontsize=24); plt.yticks(fontsize=24)
plt.grid(True, alpha=0.25)
plt.legend(frameon=False, loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=24)
plt.title("Unique JJA peak dates by upper-percentile subsets", fontsize=24)
plt.tight_layout()
plt.savefig(Path(__file__).resolve().parent / "Fig_unique_peak_dates_by_year_percentiles.pdf",
            bbox_inches="tight")
plt.show()

# %% Combined timing plots: histogram (left) + percentile trends (right)
import matplotlib.pyplot as plt
from pathlib import Path

# Assume you already have: unique_counts, series_all, series_by_q, percentiles

fig, (axL, axR) = plt.subplots(1, 2, figsize=(18, 5), dpi=300)
plt.subplots_adjust(wspace=0.3)

# --- Left: histogram across years ---
max_u = int(unique_counts.max()) if len(unique_counts) else 1
min_u = int(unique_counts.min())
bins = range(min_u, max_u + 2)

axL.hist(unique_counts.values, bins=bins, color="#1f77b4")
axL.set_xlabel("Number of unique dates", fontsize=24)
axL.set_ylabel("Number of years", fontsize=24)
axL.set_title("Distribution across years", fontsize=28)
axL.tick_params(labelsize=18)
axL.grid(alpha=0.3)

# --- Right: per-year lines by percentile ---
axR.plot(series_all.index, series_all.values, label="All data",
         linewidth=2.2, color="#d62728")

color_map = {0.80: "#1f77b4", 0.90: "#2ca02c", 0.95: "#9467bd", 0.99: "#ff7f0e"}
for q in percentiles:
    axR.plot(series_by_q[q].index, series_by_q[q].values,
             label=f">={int(q*100)}th %ile", linewidth=1.8, color=color_map[q])

axR.set_xlabel("Year", fontsize=24)
axR.set_ylabel("Number of unique days", fontsize=24)
axR.set_title("Unique JJA peak dates", fontsize=28)
axR.tick_params(labelsize=20)
axR.grid(True, alpha=0.25)
axR.legend(frameon=False, loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=24)

fig.tight_layout()
fig.savefig(Path(__file__).resolve().parent / "Fig_unique_peak_dates_combined.pdf",
            bbox_inches="tight")
plt.show()
# %%
