"""
eda.py
------
Exploratory Data Analysis — FitBit Activity & Health Data
Dataset: FitBit_data.csv (457 records, 15 features, 33 unique users)
Generates 8 production-quality charts.

Usage:
    python src/eda.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import warnings
import os
warnings.filterwarnings("ignore")

DATA_PATH  = "data/FitBit_data.csv"
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

sns.set_style("whitegrid")
plt.rcParams.update({
    "figure.dpi": 150,
    "axes.titlesize": 14,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
})


# ── Load & prep ─────────────────────────────────────────────────────────────

def load_and_clean() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df["ActivityDate"] = pd.to_datetime(df["ActivityDate"])
    df["DayOfWeek"]    = df["ActivityDate"].dt.day_name()
    df["TotalActiveMinutes"] = (
        df["VeryActiveMinutes"] +
        df["FairlyActiveMinutes"] +
        df["LightlyActiveMinutes"]
    )
    df["SedentaryHours"] = df["SedentaryMinutes"] / 60

    # Activity intensity label
    df["ActivityLevel"] = pd.cut(
        df["TotalSteps"],
        bins=[0, 5000, 10000, 15000, np.inf],
        labels=["Sedentary (<5k)", "Low Active (5k-10k)",
                "Active (10k-15k)", "Very Active (>15k)"]
    )

    print(f"[INFO] Loaded {len(df):,} records | {df['Id'].nunique()} unique users")
    print(f"[INFO] Date range: {df['ActivityDate'].min().date()} → {df['ActivityDate'].max().date()}")
    return df


def save_fig(name: str) -> None:
    path = os.path.join(OUTPUT_DIR, name)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[SAVED] {path}")


# ── Plots ───────────────────────────────────────────────────────────────────

def plot_steps_distribution(df):
    """Histogram — distribution of daily step counts."""
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.histplot(df["TotalSteps"], bins=40, kde=True,
                 color="#1565C0", ax=ax)
    ax.axvline(10000, color="red", linestyle="--", linewidth=1.5, label="10,000 step goal")
    ax.axvline(df["TotalSteps"].mean(), color="orange", linestyle="--",
               linewidth=1.5, label=f"Mean: {df['TotalSteps'].mean():,.0f}")
    ax.set_title("Daily Step Count Distribution")
    ax.set_xlabel("Total Steps")
    ax.set_ylabel("Frequency")
    ax.legend()
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    plt.tight_layout()
    save_fig("01_steps_distribution.png")


def plot_activity_level_breakdown(df):
    """Bar — how many days fall in each activity level."""
    al = df["ActivityLevel"].value_counts().reindex(
        ["Sedentary (<5k)", "Low Active (5k-10k)", "Active (10k-15k)", "Very Active (>15k)"]
    )
    colors = ["#E53935", "#FB8C00", "#43A047", "#1565C0"]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(al.index, al.values, color=colors, edgecolor="white")
    ax.set_title("Activity Level Distribution (Based on Daily Steps)")
    ax.set_ylabel("Number of Days")
    ax.set_xlabel("")
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1,
                f"{bar.get_height():,}\n({bar.get_height()/len(df)*100:.1f}%)",
                ha="center", fontsize=9, fontweight="bold")
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    save_fig("02_activity_level_breakdown.png")


def plot_steps_vs_calories(df):
    """Scatter — steps vs calories burned with regression line."""
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(df["TotalSteps"], df["Calories"],
               alpha=0.4, color="#1565C0", s=20, label="Data points")

    # Regression line
    z = np.polyfit(df["TotalSteps"].dropna(), df["Calories"].dropna(), 1)
    p = np.poly1d(z)
    x_line = np.linspace(df["TotalSteps"].min(), df["TotalSteps"].max(), 200)
    ax.plot(x_line, p(x_line), "r--", linewidth=2, label="Trend line")

    corr = df["TotalSteps"].corr(df["Calories"])
    ax.set_title(f"Steps vs Calories Burned (r = {corr:.2f})")
    ax.set_xlabel("Total Steps")
    ax.set_ylabel("Calories Burned")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.legend()
    plt.tight_layout()
    save_fig("03_steps_vs_calories.png")


def plot_active_time_breakdown(df):
    """Avg pie — how total minutes are distributed across activity types."""
    avg_minutes = {
        "Very Active": df["VeryActiveMinutes"].mean(),
        "Fairly Active": df["FairlyActiveMinutes"].mean(),
        "Lightly Active": df["LightlyActiveMinutes"].mean(),
        "Sedentary": df["SedentaryMinutes"].mean(),
    }
    colors = ["#1565C0", "#42A5F5", "#A5D6A7", "#FFCCBC"]

    fig, ax = plt.subplots(figsize=(8, 8))
    wedges, texts, autotexts = ax.pie(
        avg_minutes.values(),
        labels=avg_minutes.keys(),
        autopct="%1.1f%%",
        colors=colors,
        startangle=90,
        wedgeprops=dict(width=0.6),
        textprops={"fontsize": 11}
    )
    ax.set_title("Average Daily Time by Activity Type", pad=20)
    plt.tight_layout()
    save_fig("04_activity_time_breakdown.png")


def plot_steps_by_day_of_week(df):
    """Bar — avg steps by day of week."""
    order = ["Monday", "Tuesday", "Wednesday", "Thursday",
             "Friday", "Saturday", "Sunday"]
    daily = df.groupby("DayOfWeek")["TotalSteps"].mean().reindex(order)

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(daily.index, daily.values,
                  color=sns.color_palette("Blues_r", 7))
    ax.set_title("Average Daily Steps by Day of Week")
    ax.set_ylabel("Avg Steps")
    ax.set_xlabel("")
    ax.axhline(10000, color="red", linestyle="--", linewidth=1.5, label="10k goal")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.legend()
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 100,
                f"{bar.get_height():,.0f}",
                ha="center", fontsize=8)
    plt.tight_layout()
    save_fig("05_steps_by_day.png")


def plot_calories_by_day(df):
    """Line — avg calories burned by day of week."""
    order = ["Monday", "Tuesday", "Wednesday", "Thursday",
             "Friday", "Saturday", "Sunday"]
    cal_day = df.groupby("DayOfWeek")["Calories"].mean().reindex(order)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(cal_day.index, cal_day.values, marker="o",
            linewidth=2.5, color="#E65100", markersize=8)
    ax.fill_between(cal_day.index, cal_day.values, alpha=0.15, color="#E65100")
    ax.set_title("Average Calories Burned by Day of Week")
    ax.set_ylabel("Avg Calories")
    ax.set_xlabel("")
    for i, v in enumerate(cal_day.values):
        ax.text(i, v + 5, f"{v:.0f}", ha="center", fontsize=9)
    plt.tight_layout()
    save_fig("06_calories_by_day.png")


def plot_sedentary_vs_active(df):
    """Scatter — sedentary hours vs total active minutes."""
    fig, ax = plt.subplots(figsize=(9, 6))
    scatter = ax.scatter(
        df["SedentaryHours"], df["TotalActiveMinutes"],
        c=df["Calories"], cmap="YlOrRd", alpha=0.5, s=30
    )
    plt.colorbar(scatter, ax=ax, label="Calories Burned")
    ax.set_title("Sedentary Hours vs Active Minutes\n(Color = Calories)")
    ax.set_xlabel("Sedentary Hours")
    ax.set_ylabel("Total Active Minutes")
    corr = df["SedentaryHours"].corr(df["TotalActiveMinutes"])
    ax.annotate(f"r = {corr:.2f}", xy=(0.05, 0.92), xycoords="axes fraction",
                fontsize=11, color="navy")
    plt.tight_layout()
    save_fig("07_sedentary_vs_active.png")


def plot_user_avg_steps(df):
    """Bar — top 10 and bottom 10 users by avg daily steps."""
    user_steps = df.groupby("Id")["TotalSteps"].mean().sort_values(ascending=False)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    top10 = user_steps.head(10)
    axes[0].barh(top10.index.astype(str)[::-1], top10.values[::-1],
                 color="#43A047")
    axes[0].set_title("Top 10 Most Active Users (Avg Steps)")
    axes[0].set_xlabel("Avg Daily Steps")

    bot10 = user_steps.tail(10)
    axes[1].barh(bot10.index.astype(str), bot10.values,
                 color="#E53935")
    axes[1].set_title("Bottom 10 Least Active Users (Avg Steps)")
    axes[1].set_xlabel("Avg Daily Steps")

    plt.suptitle("User Activity Comparison — FitBit 2016", fontsize=15, fontweight="bold")
    plt.tight_layout()
    save_fig("08_user_avg_steps.png")


def print_summary(df):
    print("\n" + "=" * 50)
    print("  FITBIT EDA — KEY STATS")
    print("=" * 50)
    print(f"  Total Records         : {len(df):,}")
    print(f"  Unique Users          : {df['Id'].nunique()}")
    print(f"  Avg Daily Steps       : {df['TotalSteps'].mean():,.0f}")
    pct_10k = (df["TotalSteps"] >= 10000).mean() * 100
    print(f"  Days Meeting 10k Goal : {pct_10k:.1f}%")
    print(f"  Avg Calories/Day      : {df['Calories'].mean():,.0f}")
    print(f"  Avg Sedentary Hrs/Day : {df['SedentaryHours'].mean():.1f}h")
    print(f"  Steps-Calories Corr   : {df['TotalSteps'].corr(df['Calories']):.2f}")
    print("=" * 50)


def main():
    print("=" * 55)
    print("  FitBit Activity EDA — Analysis Pipeline")
    print("=" * 55)
    df = load_and_clean()
    print_summary(df)
    print("\n[INFO] Generating plots...")
    plot_steps_distribution(df)
    plot_activity_level_breakdown(df)
    plot_steps_vs_calories(df)
    plot_active_time_breakdown(df)
    plot_steps_by_day_of_week(df)
    plot_calories_by_day(df)
    plot_sedentary_vs_active(df)
    plot_user_avg_steps(df)
    print(f"\n[DONE] All 8 plots saved to '{OUTPUT_DIR}/'")


if __name__ == "__main__":
    main()
