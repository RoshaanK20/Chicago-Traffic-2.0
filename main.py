# ==========================================
# 🚦 Chicago Traffic Congestion Prediction
# ==========================================
# Author: Roshaan Ahmed Khan
# Objective: Predict congestion levels and estimate vehicle speeds 
#            using real-time Chicago traffic data.
# ==========================================

# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import *

#Style Settings
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("crest")
plt.rcParams.update({'figure.figsize': (8,5), 'axes.titlesize': 14, 'axes.labelsize': 12})


#Load and Clean Data
url = "https://data.cityofchicago.org/api/views/t2qc-9pjd/rows.csv?accessType=DOWNLOAD"
df = pd.read_csv(url)

# Clean columns and handle missing data
df.columns = df.columns.str.strip()
df.dropna(subset=['CURRENT_SPEED', 'REGION', 'LAST_UPDATED'], inplace=True)

# Extract hour from timestamp
df['HOUR'] = pd.to_datetime(df['LAST_UPDATED']).dt.hour

# One-hot encode categorical variable
df = pd.get_dummies(df, columns=['REGION'], drop_first=True)

print(f"✅ Dataset loaded successfully with {df.shape[0]:,} rows and {df.shape[1]} columns.")


#Feature Engineering
def congestion_level(speed):
    if speed >= 35:
        return 0  # Low congestion
    elif speed >= 20:
        return 1  # Medium congestion
    else:
        return 2  # High congestion

df['CongestionLevel'] = df['CURRENT_SPEED'].apply(congestion_level)

features = ['HOUR'] + [col for col in df.columns if col.startswith('REGION_')]
X = df[features]
y_cls = df['CongestionLevel']
y_reg = df['CURRENT_SPEED']


#Classification Model
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X, y_cls, test_size=0.25, random_state=42)

pipeline_cls = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', RandomForestClassifier(class_weight='balanced', random_state=42))
])

pipeline_cls.fit(X_train_c, y_train_c)
y_pred_cls = pipeline_cls.predict(X_test_c)


#Classification Metrics
acc = accuracy_score(y_test_c, y_pred_cls)
print(f"\nClassification Accuracy: {acc*100:.2f}%")

print("\nClassification Performance Report:")
print(classification_report(y_test_c, y_pred_cls, zero_division=0))

#Confusion Matrix
cm = confusion_matrix(y_test_c, y_pred_cls)
unique_labels = np.unique(np.concatenate((y_test_c, y_pred_cls)))
label_map = {0: "Low", 1: "Medium", 2: "High"}

disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=[label_map[l] for l in unique_labels])
disp.plot(cmap='crest', colorbar=False)
plt.title("Confusion Matrix - Congestion Level Classification")
plt.show()

#Feature Importance
model_cls = pipeline_cls.named_steps['classifier']
feat_imp = pd.Series(model_cls.feature_importances_, index=X.columns).sort_values(ascending=False)

plt.figure()
sns.barplot(x=feat_imp.values, y=feat_imp.index, palette='crest')
plt.title("Feature Importance - Random Forest Classifier")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.show()


#Regression Model
X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X, y_reg, test_size=0.25, random_state=42)

pipeline_reg = Pipeline([
    ('scaler', StandardScaler()),
    ('regressor', RandomForestRegressor(random_state=42))
])

pipeline_reg.fit(X_train_r, y_train_r)
y_pred_r = pipeline_reg.predict(X_test_r)

#Regression Metrics
mse = mean_squared_error(y_test_r, y_pred_r)
mae = mean_absolute_error(y_test_r, y_pred_r)
r2 = r2_score(y_test_r, y_pred_r)

print(f"\nRegression Performance Metrics:")
print(f"   • Mean Squared Error (MSE): {mse:.2f}")
print(f"   • Mean Absolute Error (MAE): {mae:.2f}")
print(f"   • R² Score: {r2:.3f}")

#Predicted vs Actual Plot
plt.figure()
plt.scatter(y_test_r, y_pred_r, alpha=0.6, color='#2a9d8f')
plt.plot([y_test_r.min(), y_test_r.max()], [y_test_r.min(), y_test_r.max()], 'r--', lw=2)
plt.title("Predicted vs Actual Speeds")
plt.xlabel("Actual Speed (mph)")
plt.ylabel("Predicted Speed (mph)")
plt.show()


#Data Insights & Visualization

# Speed distribution
plt.figure()
sns.histplot(df['CURRENT_SPEED'], bins=30, kde=True, color='#457b9d')
plt.title("Distribution of Current Speeds")
plt.xlabel("Speed (mph)")
plt.ylabel("Frequency")
plt.show()

# Correlation heatmap
plt.figure(figsize=(10,6))
sns.heatmap(df.corr(numeric_only=True), cmap='crest', annot=False)
plt.title("Correlation Heatmap of Numeric Features")
plt.show()

#Project Summary

print(f"""
🧾 Project Summary:
---------------------------------
• Dataset: Chicago Real-Time Traffic Speeds
• Classification Accuracy: {acc*100:.2f}%
• Regression R² Score: {r2:.3f}
• Regression MSE: {mse:.2f}
• Regression MAE: {mae:.2f}

🧠 Insights:
• The model effectively predicts congestion levels (Low / Medium / High).
• Region and Time of Day (Hour) strongly influence traffic speeds.
• Regression model performs well for estimating actual vehicle speeds.

🚀 Tools Used:
Python | Pandas | scikit-learn | Seaborn | Matplotlib
""")
