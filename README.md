# python-projects-training

# Mini Data Science Projects

This repository contains two beginner-friendly projects that demonstrate the use of **Python**, **Pandas**, **Matplotlib**, and **Machine Learning** libraries for data analysis and forecasting.

---

## Projects

### Mini Time Series Project - Sales Forecasting
- **Goal**: Forecast monthly sales using synthetic data.
- **Steps**:
  - Generate 3 years of monthly sales data (2020–2022).
  - Build a simple ARIMA model with `statsmodels`.
  - Predict sales for the next 6 months.
- **Libraries**: `pandas`, `matplotlib`, `statsmodels`

**Example output:**
Sample data:
2020-01-31 444.00
2020-02-29 369.42
2020-03-31 406.85
...

Forecasted sales for next 6 months:
2023-01-31 431.96
2023-02-28 435.41
2023-03-31 438.86
2023-04-30 442.31
2023-05-31 445.76
2023-06-30 449.22

---

### Mini Machine Learning Project - Simple Regression
- **Goal**: Build a simple regression model to predict sales based on advertising budget.
- **Steps**:
  - Create a synthetic dataset with `Advertising Budget` and `Sales`.
  - Use `LinearRegression` from `scikit-learn`.
  - Visualize data points and the fitted regression line.
- **Libraries**: `pandas`, `matplotlib`, `scikit-learn`

**Example output:**
Advertising Budget (X): [10, 20, 30, ...]
Sales (y): [25, 45, 65, ...]

Predicted Sales for Budget=60:
≈ 125.4

How to Run:
python "Mini Time Series Project - Sales Forecast.py"
python "Mini ML Project - Simple Regression.py"

Purpose
These projects are part of a learning journey to:
Practice Python for Data Science.
Understand the basics of time series forecasting and machine learning regression.
Build a foundation for more advanced projects.
