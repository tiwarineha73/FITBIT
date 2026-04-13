# FitBit Activity & Health Analytics — EDA

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![Pandas](https://img.shields.io/badge/Pandas-2.1.0-blue)
![Seaborn](https://img.shields.io/badge/Seaborn-0.12-orange)
![Domain](https://img.shields.io/badge/Domain-Health%20Analytics-green)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

---

## Project Overview

End-to-end EDA on FitBit wearable device data from 33 users over multiple weeks.
Analyzes daily step counts, calorie burn, activity intensity distribution,
sedentary behavior, and weekly patterns to uncover real health behavior insights.

---

## Problem Statement

Health and fitness companies need to understand how users actually behave —
not how they think they behave. This analysis uses real FitBit tracker data
to reveal gaps between health goals (like the 10,000 steps/day target) and
actual user behavior, and identifies which metrics most strongly correlate
with calorie expenditure.

---

## Dataset

| Field | Detail |
|-------|--------|
| Source | Kaggle FitBit Fitness Tracker Data (Public) |
| File | `FitBit_data.csv` |
| Rows | 457 daily records |
| Unique Users | 33 |
| Columns | 15 (Steps, Distance, Active Minutes, Sedentary, Calories) |
| Missing Values | None |

---

## Features Engineered

| Feature | Logic |
|---------|-------|
| `DayOfWeek` | Day name from ActivityDate |
| `TotalActiveMinutes` | Sum of Very + Fairly + Lightly Active minutes |
| `SedentaryHours` | SedentaryMinutes / 60 |
| `ActivityLevel` | 4-tier bucketing of TotalSteps |

---

## Tools & Technologies

| Tool | Purpose |
|------|---------|
| Python (Pandas, NumPy) | Data loading, feature engineering |
| Matplotlib, Seaborn | 8 visualizations |
| Jupyter Notebook | Analysis and documentation |

---

## Project Structure

```
project4-fitbit-eda/
│
├── data/
│   └── FitBit_data.csv
│
├── notebooks/
│   └── FitBit_EDA.ipynb
│
├── src/
│   └── eda.py
│
├── outputs/
│   ├── 01_steps_distribution.png
│   ├── 02_activity_levels.png
│   ├── 03_steps_vs_calories.png
│   ├── 04_weekly_patterns.png
│   ├── 05_sedentary_vs_active.png
│   └── 06_correlation_heatmap.png
│
├── README.md
└── requirements.txt
```

---

## Key Insights

- **Only ~30% of days met the 10k step goal** — majority of users consistently under the threshold
- **Steps and Calories: r ≈ 0.59** — strong positive correlation; walking is an effective calorie-burn activity
- **Avg sedentary time: ~16 hrs/day** — users are predominantly sedentary despite tracking
- **Very Active minutes avg only 21 min/day** — users rarely reach high-intensity zones
- **Tuesday and Saturday** peak activity days; Sunday is the lowest
- **Top users average 3x the steps** of least active users — extreme behavioral spread

---

## How to Run

```bash
git clone https://github.com/yourusername/project4-fitbit-eda
cd project4-fitbit-eda
pip install -r requirements.txt
python src/eda.py
# OR
jupyter notebook notebooks/FitBit_EDA.ipynb
```

---

## Resume Bullet Points

- Analyzed FitBit wearable data (457 records, 33 users, 15 health metrics) using Python to reveal that only 30% of days met the WHO-recommended 10,000 step goal
- Engineered 4 derived features (ActivityLevel buckets, TotalActiveMinutes, SedentaryHours, DayOfWeek) to enable richer behavioral segmentation
- Identified strong correlation (r = 0.59) between daily steps and calorie expenditure, and showed that average sedentary time (~16 hrs/day) far exceeds active time
- Built 8 health-domain visualizations including activity time pie, weekly trend lines, user-level comparison, and a calories correlation heatmap

---

## Author

**Neha Tiwari** | Data Analyst  
[LinkedIn](https://linkedin.com/in/neha-tiwari) | [GitHub](https://github.com/tiwarineha73)
