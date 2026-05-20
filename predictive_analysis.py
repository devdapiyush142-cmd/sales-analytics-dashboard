import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

try:
    df = pd.read_csv("historical_data.csv")
    print("Historical data successfully parsed!")
except FileNotFoundError:
    print("Error: 'historical_data.csv' missing from folder.")
    exit()

X = df[['Marketing_Spend_k']]
y = df['Sales_Revenue_k']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
mae = metrics.mean_absolute_error(y_test, y_pred)
r2 = metrics.r2_score(y_test, y_pred)

print("\n--- Model Accuracy Metrics ---")
print(f"Mean Absolute Error (MAE): {mae:.2f} k")
print(f"Model Accuracy Score (R-squared): {r2*100:.2f}%")

future_budgets = np.array([[60], [65], [70]])
future_forecasts = model.predict(future_budgets)

print("\n--- Future Revenue Forecasts ---")
for budget, forecast in zip(future_budgets, future_forecasts):
    print(f"If we spend {budget[0]}k on marketing -> Predicted Sales: {forecast:.2f} k")

plt.figure(figsize=(10, 6))
sns.scatterplot(x='Marketing_Spend_k', y='Sales_Revenue_k', data=df, color='blue', s=80, label='Actual Historical Data')
plt.plot(df['Marketing_Spend_k'], model.predict(X), color='orange', linewidth=2, label='Predictive Trend Line')
plt.scatter(future_budgets, future_forecasts, color='red', marker='X', s=150, label='Future Estimates')

plt.title('Predictive Revenue Forecast Modeling', fontsize=14)
plt.xlabel('Marketing Spend (k)', fontsize=12)
plt.ylabel('Sales Revenue (k)', fontsize=12)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)

plt.savefig('revenue_predictions.png')
print("\nForecast visualization generated as 'revenue_predictions.png'!")
plt.show()
