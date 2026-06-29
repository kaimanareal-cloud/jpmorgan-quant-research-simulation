"""
Natural Gas Price Estimator
- Interpolates historical prices (Oct 2020 – Sep 2024)
- Extrapolates one year beyond the data using seasonal decomposition + trend
- get_price(date_str) returns an estimated purchase price for any date
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.interpolate import CubicSpline
from statsmodels.tsa.seasonal import seasonal_decompose
import warnings
warnings.filterwarnings("ignore")

# ── 1. Load data ─────────────────────────────────────────────────────────────

DATA_PATH = r"C:\Users\kaija\Downloads\Nat_Gas.csv"

def load_data(path=DATA_PATH):
    df = pd.read_csv(path)
    df.columns = ["date", "price"]
    df["date"] = pd.to_datetime(df["date"], format="%m/%d/%y")
    df = df.sort_values("date").reset_index(drop=True)
    return df

# ── 2. Build interpolator + extrapolator ─────────────────────────────────────

def build_model(df):
    """
    Returns a callable price_fn(date) valid from first data point to
    one year beyond the last data point.

    Strategy:
      - Inside the data range: cubic spline interpolation.
      - Beyond the data range (up to +1 year): trend + seasonal model.
        * Linear trend fitted on full series.
        * Monthly seasonal factors from STL decomposition.
    """
    # Numeric days since epoch for spline
    t0 = df["date"].iloc[0]
    days = (df["date"] - t0).dt.days.values.astype(float)
    prices = df["price"].values.astype(float)

    spline = CubicSpline(days, prices, bc_type="not-a-knot")

    # Seasonal decomposition (additive, period=12 months)
    ts = df.set_index("date")["price"].asfreq("ME")  # month-end freq
    decomp = seasonal_decompose(ts, model="additive", period=12, extrapolate_trend="freq")

    seasonal = decomp.seasonal      # one full cycle of seasonal factors
    trend_comp = decomp.trend

    # Fit linear trend on the trend component (drop NaNs at edges)
    trend_clean = trend_comp.dropna()
    trend_days = ((trend_clean.index - t0).days).values.astype(float)
    trend_coeff = np.polyfit(trend_days, trend_clean.values, 1)  # slope, intercept

    # Monthly seasonal lookup (month number 1-12 → seasonal adjustment)
    monthly_seasonal = seasonal.groupby(seasonal.index.month).mean().to_dict()

    last_day = float((df["date"].iloc[-1] - t0).days)
    extrap_end = last_day + 365.25

    def price_fn(query_date):
        if isinstance(query_date, str):
            query_date = pd.to_datetime(query_date)
        d = float((query_date - t0).days)

        if d < 0:
            raise ValueError(f"Date {query_date.date()} is before the data start {t0.date()}")
        if d > extrap_end:
            raise ValueError(
                f"Date {query_date.date()} is more than 1 year beyond the data end "
                f"({df['date'].iloc[-1].date()}). Extrapolation not supported past that."
            )

        if d <= last_day:
            return float(spline(d))
        else:
            # Extrapolate: linear trend + seasonal factor
            trend_val = np.polyval(trend_coeff, d)
            seas_val = monthly_seasonal.get(query_date.month, 0.0)
            return float(trend_val + seas_val)

    return price_fn, t0, last_day, extrap_end, trend_coeff, monthly_seasonal, t0

# ── 3. Public API ─────────────────────────────────────────────────────────────

_df = load_data()
_price_fn, _t0, _last_day, _extrap_end, _trend_coeff, _monthly_seasonal, _origin = build_model(_df)

def get_price(date_input):
    """
    Return estimated natural gas purchase price for a given date.

    Parameters
    ----------
    date_input : str or datetime-like
        Any date from 2020-10-31 to ~2025-09-30 (one year beyond data end).

    Returns
    -------
    float : estimated price (same units as source data, USD/MMBtu)
    """
    price = _price_fn(date_input)
    return round(price, 4)

# ── 4. Visualisation ──────────────────────────────────────────────────────────

def plot_analysis():
    df = _df.copy()

    # Build dense grid for smooth curves
    all_days = np.linspace(0, _extrap_end, 800)
    all_dates = [_t0 + timedelta(days=float(d)) for d in all_days]
    hist_mask = all_days <= _last_day
    extrap_mask = all_days >= _last_day

    hist_dates  = [d for d, m in zip(all_dates, hist_mask) if m]
    hist_prices = [_price_fn(d) for d in hist_dates]
    ext_dates   = [d for d, m in zip(all_dates, extrap_mask) if m]
    ext_prices  = [_price_fn(d) for d in ext_dates]

    # Seasonal bar chart
    months = list(range(1, 13))
    seas_vals = [_monthly_seasonal.get(m, 0) for m in months]
    month_labels = ["Jan","Feb","Mar","Apr","May","Jun",
                    "Jul","Aug","Sep","Oct","Nov","Dec"]

    fig, axes = plt.subplots(3, 1, figsize=(13, 14))
    fig.suptitle("Natural Gas Price Analysis & Forecast", fontsize=16, fontweight="bold", y=0.98)

    # ── Panel 1: Raw data + spline + extrapolation ────────────────────────────
    ax1 = axes[0]
    ax1.scatter(df["date"], df["price"], color="#1f4e79", zorder=5, s=55,
                label="Monthly data points", edgecolors="white", linewidths=0.6)
    ax1.plot(hist_dates, hist_prices, color="#2e86de", linewidth=2,
             label="Cubic spline (interpolation)")
    ax1.plot(ext_dates, ext_prices, color="#e84118", linewidth=2, linestyle="--",
             label="Trend + seasonal (extrapolation)")
    ax1.axvline(df["date"].iloc[-1], color="gray", linestyle=":", linewidth=1.2, label="Data end")
    ax1.set_title("Price History & 12-Month Forecast", fontsize=12)
    ax1.set_ylabel("Price (USD/MMBtu)")
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=30, ha="right")
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.3)

    # ── Panel 2: Seasonal pattern ─────────────────────────────────────────────
    ax2 = axes[1]
    colors = ["#e84118" if v > 0 else "#2e86de" for v in seas_vals]
    bars = ax2.bar(month_labels, seas_vals, color=colors, edgecolor="white", linewidth=0.8)
    ax2.axhline(0, color="black", linewidth=0.8)
    ax2.set_title("Average Seasonal Price Deviation by Month", fontsize=12)
    ax2.set_ylabel("Seasonal Component (USD/MMBtu)")
    ax2.grid(axis="y", alpha=0.3)
    for bar, val in zip(bars, seas_vals):
        ax2.text(bar.get_x() + bar.get_width() / 2, val + (0.003 if val >= 0 else -0.006),
                 f"{val:+.3f}", ha="center", va="bottom" if val >= 0 else "top", fontsize=8)

    # ── Panel 3: Year-over-year comparison ───────────────────────────────────
    ax3 = axes[2]
    df["month"] = df["date"].dt.month
    df["year"]  = df["date"].dt.year
    pivot = df.pivot_table(index="month", columns="year", values="price")
    for yr in pivot.columns:
        ax3.plot(month_labels[:len(pivot[yr].dropna())],
                 pivot[yr].dropna().values,
                 marker="o", markersize=5, label=str(yr), linewidth=1.8)
    ax3.set_title("Year-over-Year Monthly Prices", fontsize=12)
    ax3.set_ylabel("Price (USD/MMBtu)")
    ax3.legend(title="Year", fontsize=9)
    ax3.grid(alpha=0.3)

    plt.tight_layout()
    out = r"C:\Users\kaija\quant research\nat_gas_analysis.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"\nPlot saved to: {out}")

# ── 5. CLI demo ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 55)
    print("  Natural Gas Price Estimator")
    print("  Data: Oct 2020 – Sep 2024 | Extrap: +1 year")
    print("=" * 55)

    # Sample queries
    test_dates = [
        "2021-06-15",   # mid-history
        "2022-12-31",   # end of year
        "2023-08-10",   # summer
        "2024-09-30",   # last data point
        "2024-11-15",   # 2 months into extrapolation
        "2025-03-01",   # ~6 months out
        "2025-09-28",   # ~1 year out (near limit)
    ]

    for d in test_dates:
        try:
            p = get_price(d)
            label = "(extrapolated)" if pd.to_datetime(d) > _df["date"].iloc[-1] else "(interpolated)"
            print(f"  {d}  ->  ${p:.4f}  {label}")
        except ValueError as e:
            print(f"  {d}  ->  ERROR: {e}")

    print("\nGenerating analysis plots...")
    plot_analysis()

    print("\n── Interactive mode ──")
    print("Enter a date (YYYY-MM-DD) or 'q' to quit.")
    while True:
        try:
            user_input = input("Date: ").strip()
        except EOFError:
            break
        if user_input.lower() in ("q", "quit", "exit", ""):
            break
        try:
            price = get_price(user_input)
            dt = pd.to_datetime(user_input)
            label = "extrapolated" if dt > _df["date"].iloc[-1] else "interpolated"
            print(f"  Estimated price: ${price:.4f} USD/MMBtu  ({label})")
        except Exception as e:
            print(f"  Error: {e}")
