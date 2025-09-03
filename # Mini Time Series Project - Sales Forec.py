# Mini Time Series Project - Sales Forecasting
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Create sample monthly sales data (3 years)
dates = pd.date_range(start="2020-01", periods=36, freq="ME")  # ME = month end
sales = np.random.randint(200, 500, size=36) + np.linspace(
    0, 50, 36
)  # trend + randomness

df = pd.DataFrame({"Date": dates, "Sales": sales})
df.set_index("Date", inplace=True)

print("Sample data:\n", df.head())

# Plot historical data
df["Sales"].plot(figsize=(10, 5), title="Monthly Sales Data", marker="o")
plt.savefig("sales_data_plot.png")  # Save historical data plot
plt.show()

# Train model (Holt-Winters Exponential Smoothing)
model = ExponentialSmoothing(df["Sales"], trend="add", seasonal=None)
fit = model.fit()

# Forecast next 6 months
forecast = fit.forecast(6)
print("\nForecasted sales for next 6 months:\n", forecast)

# Plot results with forecast
plt.figure(figsize=(10, 5))
plt.plot(df.index, df["Sales"], label="Historical Sales")
plt.plot(forecast.index, forecast, label="Forecast", marker="o")
plt.legend()
plt.title("Sales Forecasting with Exponential Smoothing")
plt.savefig("sales_forecast.png")  # Save forecast plot
plt.show()
