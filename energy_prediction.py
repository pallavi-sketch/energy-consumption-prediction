import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Load dataset
data = pd.read_csv("energy_data.csv")

# Features
X = data[['Temperature', 'Humidity', 'WindSpeed']]

# Target
y = data['EnergyConsumption']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Prediction
predictions = model.predict(X_test)

print("Predicted Energy Consumption:", predictions)

# Graph
plt.scatter(y_test, predictions)
plt.xlabel("Actual Energy")
plt.ylabel("Predicted Energy")
plt.title("Energy Consumption Prediction")
plt.show()