import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load the dataset
df = pd.read_csv("water_potability.csv")

# 1. Dataset Summary in Pie and Bar chart
st.header("Dataset Summary")

# Pie chart
st.subheader("Potability Distribution (Pie Chart)")
potability_counts = df['Potability'].value_counts()
st.write(potability_counts)
fig1, ax1 = plt.subplots()
ax1.pie(potability_counts, labels=potability_counts.index, autopct='%1.1f%%', startangle=90)
ax1.axis('equal')
st.pyplot(fig1)

# Bar chart
st.subheader("Potability Distribution (Bar Chart)")
st.bar_chart(potability_counts)

# 2. Correlation Heat Map
st.header("Correlation Heat Map")
corr = df.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
st.pyplot()

# 3. Regression and Error Calculation
st.header("Regression Analysis")

# Select features and target
features = ['ph', 'Hardness', 'Solids', 'Chloramines', 'Sulfate', 'Conductivity', 'Organic_carbon', 'Trihalomethanes', 'Turbidity']
target = 'Potability'

X = df[features]
y = df[target]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Calculate MAE and MSE
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

# Display results
st.write(f"Mean Absolute Error (MAE): {mae:.4f}")
st.write(f"Mean Squared Error (MSE): {mse:.4f}")

# Scatter plot of true vs. predicted values
st.subheader("Scatter Plot of True vs. Predicted Values")
plt.scatter(y_test, y_pred)
plt.xlabel("True Values")
plt.ylabel("Predictions")
plt.title("True vs. Predicted")
st.pyplot()
