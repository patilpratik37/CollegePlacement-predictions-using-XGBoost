import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('placement-dataset.csv')
df = df.drop("Unnamed: 0", axis=1)
df
df.describe()

df['cgpa'].hist()
plt.title('CGPA Distribution')
plt.xlabel('CGPA')
plt.ylabel('Frequency')
plt.show()

df['iq'].hist()
plt.title('CGPA Distribution')
plt.xlabel('CGPA')
plt.ylabel('Frequency')
plt.show()

# Assuming df is already defined and has 'cgpa' and 'iq' columns
correlation = df['cgpa'].corr(df['iq'])

print("The correlation coefficient between CGPA and IQ is:", correlation)

plt.scatter(df['iq'], df['cgpa'])
plt.title('Scatter plot of IQ vs CGPA')
plt.xlabel('IQ')
plt.ylabel('CGPA')
plt.grid(True)
plt.show()

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

X = df[['cgpa', 'iq']]  # Features
y = df['placement']  # Target variable

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
print("Accuracy:", accuracy)
print("Confusion Matrix:\n", conf_matrix)

# Using Matplotlib
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()

#CGPA vs. Placement (XGBoost with constant IQ)
import xgboost as xgb

X = df[['cgpa', 'iq']]  # Using both 'cgpa' and 'iq' as features
y = df['placement']  # Target variable

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the XGBoost model
model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
model.fit(X_train, y_train)

# Make predictions and evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

# Print out the accuracy and confusion matrix
print("Accuracy:", accuracy)
print("Confusion Matrix:\n", conf_matrix)

# Visualization
# Choose a representative value for 'iq' (e.g., the mean)
mean_iq = X_train['iq'].mean()

# Generate a sequence of CGPA values from the min to max range
cgpa_range = np.linspace(X_train['cgpa'].min(), X_train['cgpa'].max(), 100).reshape(-1, 1)

# Create a 2D array combining 'cgpa_range' and the constant 'iq' value
combined_input = np.hstack((cgpa_range, np.full_like(cgpa_range, mean_iq)))

# Predict placement for these CGPA and constant IQ values
predicted_placement = model.predict(combined_input)

# Plotting
plt.figure(figsize=(10, 6))
plt.scatter(X_train['cgpa'], y_train, color='blue', label='Training data')
plt.plot(cgpa_range, predicted_placement, color='red', label='XGBoost Decision Boundary')
plt.xlabel('CGPA')
plt.ylabel('Placement')
plt.title('CGPA vs. Placement (XGBoost with constant IQ)')
plt.legend()
plt.show()