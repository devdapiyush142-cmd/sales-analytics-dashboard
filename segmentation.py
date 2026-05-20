import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

try:
    df = pd.read_csv('customer_data.csv')
    print('Data loaded successfully!')
except FileNotFoundError:
    print('Error: customer_data.csv not found.')
    exit()

X = df[['Annual_Income_k', 'Spending_Score']]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df['Cluster'] = kmeans.fit_predict(X_scaled)

print('\n--- Cluster Profiles ---')
print(df.groupby('Cluster')[['Age', 'Annual_Income_k', 'Spending_Score']].mean())

plt.figure(figsize=(10, 6))
sns.scatterplot(x='Annual_Income_k', y='Spending_Score', hue='Cluster', palette='viridis', data=df, s=100)
plt.title('Customer Segments')
plt.savefig('customer_segments.png')
print('\nPlot saved as customer_segments.png!')
plt.show()
