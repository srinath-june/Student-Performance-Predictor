import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 1. DATA GENERATION & SETUP
# Setting a random seed for reproducible results
np.random.seed(42)
n_students = 500

data = {
    'Hours_Studied': np.random.uniform(1, 10, n_students),
    'Attendance_Pct': np.random.uniform(60, 100, n_students),
    'Sleep_Hours': np.random.uniform(5, 9, n_students),
    'Extracurricular': np.random.choice([0, 1], size=n_students, p=[0.7, 0.3]),
    'Previous_Score': np.random.normal(70, 12, n_students)
}

df = pd.DataFrame(data)

# Ensure scores stay within logical boundaries [0, 100]
df['Previous_Score'] = df['Previous_Score'].clip(0, 100)

# Introduce 5% missing values in 'Previous_Score' to simulate messy real-world data
df.loc[df.sample(frac=0.05, random_state=42).index, 'Previous_Score'] = np.nan

# Create target variable (Exam_Score) with synthetic relationships + random noise
df['Exam_Score'] = (
    (df['Hours_Studied'] * 4.2) + 
    (df['Attendance_Pct'] * 0.35) + 
    (df['Previous_Score'].fillna(70) * 0.25) + 
    np.random.normal(0, 4, n_students)
)
df['Exam_Score'] = df['Exam_Score'].clip(0, 100)

print("--- Step 1: Raw Data Sample ---")
print(df.head())
print(f"\nMissing values before cleaning:\n{df.isnull().sum()}")

# 2. DATA CLEANING
# Handle missing data by imputing the median value
median_prev_score = df['Previous_Score'].median()
df['Previous_Score'] = df['Previous_Score'].fillna(median_prev_score)

print("\n--- Step 2: Cleaned Data Check ---")
print(f"Missing values after cleaning:\n{df.isnull().sum()}")

# 3. FEATURE ENGINEERING
# Create a new feature capturing the compounding effect of studying and attending class
df['Study_Efficiency'] = df['Hours_Studied'] * (df['Attendance_Pct'] / 100)

print("\n--- Step 3: Engineered Data Sample ---")
print(df[['Hours_Studied', 'Attendance_Pct', 'Study_Efficiency']].head())

# 4. DATA SPLITTING & SCALING
# Split data into Features (X) and Target Label (y)
X = df.drop(columns=['Exam_Score'])
y = df['Exam_Score']

# Split into 80% Training set and 20% Testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features so that features with larger ranges don't dominate the model
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# =====================================================================
# 5. MODEL TRAINING
# =====================================================================
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# =====================================================================
# 6. MODEL EVALUATION
# =====================================================================
# Make predictions using the test set
y_pred = model.predict(X_test_scaled)

# Calculate evaluation metrics
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\n--- Step 6: Model Evaluation Metrics ---")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f} (On average, predictions err by {rmse:.2f} points)")
print(f"R-squared Score (R²): {r2:.2f} ({r2*100:.1f}% of variance explained by features)")

# View feature importance weights
coefficients = pd.DataFrame({
    'Feature': X.columns,
    'Weight (Coefficient)': model.coef_
}).sort_values(by='Weight (Coefficient)', ascending=False)

print("\n--- Feature Importances ---")
print(coefficients)

# =====================================================================
# 7. DATA VISUALIZATION
# =====================================================================
plt.figure(figsize=(14, 5))

# Subplot 1: Actual vs Predicted Scatter Plot
plt.subplot(1, 2, 1)
plt.scatter(y_test, y_pred, alpha=0.7, color='#1f77b4', edgecolors='k')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Perfect Prediction Line')
plt.title('Actual vs. Predicted Exam Scores', fontsize=12)
plt.xlabel('Actual Scores')
plt.ylabel('Predicted Scores')
plt.legend()
plt.grid(True, linestyle=':', alpha=0.6)

# Subplot 2: Feature Importance Bar Chart
plt.subplot(1, 2, 2)
sns.barplot(x='Weight (Coefficient)', y='Feature', data=coefficients, palette='coolwarm')
plt.title('Feature Influence Weight on Final Score', fontsize=12)
plt.xlabel('Impact Degree (Coefficient Coefficient Value)')
plt.ylabel('Features')

plt.tight_layout()
plt.show()
