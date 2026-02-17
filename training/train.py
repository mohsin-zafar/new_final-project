"""
Manufacturing Equipment Output Prediction - Training Script
============================================================
This script trains a Linear Regression model to predict Parts_Per_Hour
using manufacturing machine parameters.

Author: ML Engineer
Date: 2024
"""

# ============================================
# STEP 1: IMPORT REQUIRED LIBRARIES
# ============================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("MANUFACTURING EQUIPMENT OUTPUT PREDICTION")
print("Linear Regression Model Training")
print("=" * 60)

# ============================================
# STEP 2: LOAD THE DATASET
# ============================================
print("\n[STEP 1] Loading Dataset...")

# For Google Colab - upload the file first
# from google.colab import files
# uploaded = files.upload()

# Load the dataset
df = pd.read_csv('manufacturing_dataset_1000_samples.csv')

print(f"✓ Dataset loaded successfully!")
print(f"✓ Shape: {df.shape[0]} rows, {df.shape[1]} columns")

# ============================================
# STEP 3: EXPLORATORY DATA ANALYSIS (EDA)
# ============================================
print("\n[STEP 2] Exploratory Data Analysis...")
print("\n--- Dataset Info ---")
print(f"Columns: {list(df.columns)}")
print(f"\nData Types:")
print(df.dtypes)

print("\n--- Statistical Summary ---")
print(df.describe())

print("\n--- First 5 Rows ---")
print(df.head())

# ============================================
# STEP 4: CHECK FOR MISSING VALUES
# ============================================
print("\n[STEP 3] Checking for Missing Values...")
missing_values = df.isnull().sum()
print(missing_values)

# Handle missing values if any
if df.isnull().sum().sum() > 0:
    print("\n⚠ Missing values found! Handling them...")
    # For numerical columns - fill with median
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    for col in numerical_cols:
        if df[col].isnull().sum() > 0:
            df[col].fillna(df[col].median(), inplace=True)
    
    # For categorical columns - fill with mode
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if df[col].isnull().sum() > 0:
            df[col].fillna(df[col].mode()[0], inplace=True)
    
    print("✓ Missing values handled!")
else:
    print("✓ No missing values found!")

# ============================================
# STEP 5: FEATURE ENGINEERING & PREPROCESSING
# ============================================
print("\n[STEP 4] Feature Engineering & Preprocessing...")

# Drop Timestamp column as it's not useful for prediction
if 'Timestamp' in df.columns:
    df = df.drop('Timestamp', axis=1)
    print("✓ Dropped 'Timestamp' column")

# Identify categorical and numerical columns
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()

# Remove target variable from numerical columns
target_col = 'Parts_Per_Hour'
if target_col in numerical_cols:
    numerical_cols.remove(target_col)

print(f"\nCategorical columns: {categorical_cols}")
print(f"Numerical columns: {numerical_cols}")
print(f"Target column: {target_col}")

# ============================================
# STEP 6: ENCODE CATEGORICAL VARIABLES
# ============================================
print("\n[STEP 5] Encoding Categorical Variables...")

# Create a dictionary to store label encoders
label_encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le
    print(f"✓ Encoded '{col}': {list(le.classes_)}")

# Save label encoders for later use
joblib.dump(label_encoders, 'label_encoders.pkl')
print("\n✓ Label encoders saved to 'label_encoders.pkl'")

# ============================================
# STEP 7: PREPARE FEATURES AND TARGET
# ============================================
print("\n[STEP 6] Preparing Features and Target...")

# Define features (X) and target (y)
X = df.drop(target_col, axis=1)
y = df[target_col]

print(f"✓ Features shape: {X.shape}")
print(f"✓ Target shape: {y.shape}")
print(f"\nFeature columns: {list(X.columns)}")

# Save feature names for prediction
feature_names = list(X.columns)
joblib.dump(feature_names, 'feature_names.pkl')
print("✓ Feature names saved to 'feature_names.pkl'")

# ============================================
# STEP 8: TRAIN-TEST SPLIT
# ============================================
print("\n[STEP 7] Splitting Data into Train and Test Sets...")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42
)

print(f"✓ Training set: {X_train.shape[0]} samples")
print(f"✓ Test set: {X_test.shape[0]} samples")

# ============================================
# STEP 9: FEATURE SCALING
# ============================================
print("\n[STEP 8] Feature Scaling (StandardScaler)...")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("✓ Features scaled using StandardScaler")

# Save the scaler
joblib.dump(scaler, 'scaler.pkl')
print("✓ Scaler saved to 'scaler.pkl'")

# ============================================
# STEP 10: TRAIN LINEAR REGRESSION MODEL
# ============================================
print("\n[STEP 9] Training Linear Regression Model...")

model = LinearRegression()
model.fit(X_train_scaled, y_train)

print("✓ Model trained successfully!")

# ============================================
# STEP 11: MODEL EVALUATION
# ============================================
print("\n[STEP 10] Model Evaluation...")

# Predictions
y_train_pred = model.predict(X_train_scaled)
y_test_pred = model.predict(X_test_scaled)

# Calculate metrics for training set
train_mse = mean_squared_error(y_train, y_train_pred)
train_rmse = np.sqrt(train_mse)
train_mae = mean_absolute_error(y_train, y_train_pred)
train_r2 = r2_score(y_train, y_train_pred)

# Calculate metrics for test set
test_mse = mean_squared_error(y_test, y_test_pred)
test_rmse = np.sqrt(test_mse)
test_mae = mean_absolute_error(y_test, y_test_pred)
test_r2 = r2_score(y_test, y_test_pred)

print("\n" + "=" * 50)
print("MODEL PERFORMANCE METRICS")
print("=" * 50)

print("\n--- Training Set ---")
print(f"MSE:  {train_mse:.4f}")
print(f"RMSE: {train_rmse:.4f}")
print(f"MAE:  {train_mae:.4f}")
print(f"R²:   {train_r2:.4f}")

print("\n--- Test Set ---")
print(f"MSE:  {test_mse:.4f}")
print(f"RMSE: {test_rmse:.4f}")
print(f"MAE:  {test_mae:.4f}")
print(f"R²:   {test_r2:.4f}")

# ============================================
# STEP 12: FEATURE IMPORTANCE
# ============================================
print("\n[STEP 11] Feature Importance Analysis...")

# Get coefficients
coefficients = pd.DataFrame({
    'Feature': feature_names,
    'Coefficient': model.coef_
})
coefficients['Abs_Coefficient'] = abs(coefficients['Coefficient'])
coefficients = coefficients.sort_values('Abs_Coefficient', ascending=False)

print("\nFeature Coefficients (sorted by importance):")
print(coefficients.to_string(index=False))

# ============================================
# STEP 13: SAVE THE MODEL
# ============================================
print("\n[STEP 12] Saving Model...")

joblib.dump(model, 'model.pkl')
print("✓ Model saved to 'model.pkl'")

# ============================================
# STEP 14: VISUALIZATIONS
# ============================================
print("\n[STEP 13] Creating Visualizations...")

# Create figure with subplots
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Actual vs Predicted (Test Set)
axes[0, 0].scatter(y_test, y_test_pred, alpha=0.5, color='blue')
axes[0, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
axes[0, 0].set_xlabel('Actual Parts Per Hour')
axes[0, 0].set_ylabel('Predicted Parts Per Hour')
axes[0, 0].set_title(f'Actual vs Predicted (Test Set)\nR² = {test_r2:.4f}')
axes[0, 0].grid(True, alpha=0.3)

# Plot 2: Residuals Distribution
residuals = y_test - y_test_pred
axes[0, 1].hist(residuals, bins=30, edgecolor='black', alpha=0.7, color='green')
axes[0, 1].axvline(x=0, color='red', linestyle='--', lw=2)
axes[0, 1].set_xlabel('Residuals')
axes[0, 1].set_ylabel('Frequency')
axes[0, 1].set_title('Residuals Distribution')
axes[0, 1].grid(True, alpha=0.3)

# Plot 3: Feature Coefficients
top_features = coefficients.head(10)
colors = ['green' if x > 0 else 'red' for x in top_features['Coefficient']]
axes[1, 0].barh(top_features['Feature'], top_features['Coefficient'], color=colors)
axes[1, 0].set_xlabel('Coefficient Value')
axes[1, 0].set_title('Top 10 Feature Coefficients')
axes[1, 0].grid(True, alpha=0.3)

# Plot 4: Prediction Error
axes[1, 1].scatter(y_test_pred, residuals, alpha=0.5, color='purple')
axes[1, 1].axhline(y=0, color='red', linestyle='--', lw=2)
axes[1, 1].set_xlabel('Predicted Values')
axes[1, 1].set_ylabel('Residuals')
axes[1, 1].set_title('Residuals vs Predicted Values')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('training_results.png', dpi=300, bbox_inches='tight')
plt.show()

print("✓ Visualizations saved to 'training_results.png'")

# ============================================
# STEP 15: TEST PREDICTION
# ============================================
print("\n[STEP 14] Test Prediction...")

# Load saved model and scaler for verification
loaded_model = joblib.load('model.pkl')
loaded_scaler = joblib.load('scaler.pkl')

# Take a sample from test set
sample = X_test.iloc[0:1]
sample_scaled = loaded_scaler.transform(sample)
prediction = loaded_model.predict(sample_scaled)

print(f"\nSample Input Features:")
print(sample.to_string())
print(f"\nPredicted Parts Per Hour: {prediction[0]:.2f}")
print(f"Actual Parts Per Hour: {y_test.iloc[0]:.2f}")

# ============================================
# SUMMARY
# ============================================
print("\n" + "=" * 60)
print("TRAINING COMPLETE!")
print("=" * 60)
print("\nSaved Files:")
print("  ✓ model.pkl - Trained Linear Regression model")
print("  ✓ scaler.pkl - StandardScaler for feature scaling")
print("  ✓ label_encoders.pkl - Label encoders for categorical variables")
print("  ✓ feature_names.pkl - List of feature names")
print("  ✓ training_results.png - Visualization plots")
print("\nModel Performance:")
print(f"  • Test R² Score: {test_r2:.4f}")
print(f"  • Test RMSE: {test_rmse:.4f}")
print("\nNext Steps:")
print("  1. Download model.pkl and scaler.pkl")
print("  2. Place them in the 'model' folder of your project")
print("  3. Run the backend and frontend applications")
print("=" * 60)
