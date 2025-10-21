🚦 Chicago Traffic Congestion Prediction (ML Project)

This project uses real-world data from the **City of Chicago** to analyze and predict traffic congestion levels based on time and region using **Machine Learning** models.  
Originally built in my **3rd semester** as part of my BS in Artificial Intelligence program, and now improved in my **4th semester** with better accuracy, visuals, and cleaner code.

---

## 📊 Project Overview
- **Goal:** Predict congestion levels (Low, Medium, High) using features like hour of day and region.  
- **Dataset:** [Chicago Traffic Tracker - Congestion Estimates by Regions](https://data.cityofchicago.org/api/views/t2qc-9pjd/rows.csv?accessType=DOWNLOAD)  
- **Tech Stack:**  
  - Python  
  - Pandas, NumPy  
  - Scikit-learn  
  - Matplotlib, Seaborn  

---

## 🧠 Machine Learning Workflow

### 1️⃣ Data Preparation
- Cleaned and processed raw CSV data.  
- Extracted hour information from timestamps.  
- Encoded categorical region values using one-hot encoding.  
- Created a new target feature **CongestionLevel** based on current speed.

### 2️⃣ Model Training
Two ML pipelines were built:
- **Classification Model:** Predicts congestion level (Low, Medium, High)  
  - Algorithm: `RandomForestClassifier`
  - Metrics: Accuracy, Precision, Recall, F1-score
- **Regression Model:** Predicts the current traffic speed  
  - Algorithm: `LinearRegression`
  - Metrics: Mean Squared Error (MSE), Mean Absolute Error (MAE), R² Score

### 3️⃣ Evaluation
- Achieved strong accuracy in congestion classification.  
- Visualized results using:
  - Confusion Matrix
  - Residual Error Plots
  - Feature Importance Chart

---

## 📈 Key Results
| Metric | Value |
|--------|--------|
| Classification Accuracy | ~90% (balanced) |
| R² Score (Regression) | ~0.85 |
| MAE | Low error margin |

*(Values may vary depending on data refresh.)*

---

## 🖼 Visual Outputs
- Confusion Matrix  
- Regression Residuals Plot  
- Feature Importance Visualization  

Example:

![Project Preview](A_digital_graphic_design_showcases_a_project_title.png)

---

## 🚀 How to Run
1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/chicago-traffic-congestion.git
   cd chicago-traffic-congestion
