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
plt.ylabel("Number of years", fontsize=16)
plt.title("Distribution across years", fontsize=20)
plt.xticks(fontsize=14)   # x-axis tick label size
plt.yticks(fontsize=14)   # y-axis tick label size
plt.tight_layout()
plt.savefig(Path(__file__).resolve().parent / "Fig_unique_peak_dates_hist.pdf")
plt.show()

# --- (Nice to have) per-year line to see trends ---
plt.figure(figsize=(8.125, 5), dpi=300)
plt.plot(unique_counts.index, unique_counts.values, marker="o", linewidth=1)
plt.ylabel("# unique dates (within JJA)", fontsize=16)
plt.xlabel("Year", fontsize = 16)
plt.xticks(fontsize=14)   # x-axis tick label size
plt.yticks(fontsize=14)   # y-axis tick label size
plt.title("Per-year count of unique JJA peak dates across 590 sites", fontsize = 20)
plt.tight_layout()
plt.savefig(Path(__file__).resolve().parent / "Fig_unique_peak_dates_by_year.pdf")
plt.show()
# %%