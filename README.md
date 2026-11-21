
# Banking & Financial Risk Prediction System Using Machine Learning

## Project Overview
This project uses Machine Learning techniques to analyze banking customer data and identify potential financial risks.

It implements three major machine learning approaches:
1. Regression – Predicts the historical BILL_AMT6 value.
2. Classification – Predicts whether a customer may default on credit card payments.
3. Clustering – Segments customers into Low Risk, Medium Risk, and High Risk groups.

An interactive Streamlit dashboard allows users to explore customer data and model predictions.

## Dataset
- Dataset: Default of Credit Card Clients
- Source: UCI Machine Learning Repository
- Records: 30,000
- Original Features: 24 input features + 1 target column
- Target: DEFAULT

Dataset: https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients

## Technologies Used
- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Streamlit
- Joblib

## Machine Learning Models

### 1. Regression
Algorithm: Random Forest Regressor

Target: BILL_AMT6

Results:
- MAE: 3844.46
- RMSE: 13533.05
- R2 Score: 0.9502

### 2. Classification
Algorithm: Random Forest Classifier

Target: DEFAULT

Results:
- Accuracy: 80%
- Default Class Precision: 56%
- Default Class Recall: 48%
- Default Class F1-Score: 52%

### 3. Clustering
Algorithm: K-Means

Number of Clusters: 3

Results:
- High Risk: 15,872 customers
- Medium Risk: 3,585 customers
- Low Risk: 10,543 customers
- Silhouette Score: 0.1948

Risk segments are exploratory groups based on observed default rates, not guaranteed individual risk assessments.

## Project Structure
- data/: Raw and processed datasets
- src/: Data processing and model training scripts
- models/: Saved machine learning models
- reports/: EDA charts and evaluation outputs
- app.py: Streamlit dashboard

## How to Run

### 1. Clone the repository
git clone YOUR_GITHUB_REPOSITORY_URL

### 2. Move into the project folder
cd banking-financial-risk-prediction

### 3. Create a virtual environment
python -m venv .venv

### 4. Activate environment (Windows)
.venv\Scripts\Activate.ps1

### 5. Install dependencies
python -m pip install -r requirements.txt

### 6. Run the dashboard
python -m streamlit run app.py

## Dashboard Features
- Customer data overview
- Credit default prediction
- Historical bill amount estimation
- Customer risk segmentation
- Interactive charts and visualizations

## Future Enhancements
- Improve classification recall and model performance
- Add XGBoost and Logistic Regression comparisons
- Implement model explainability using SHAP
- Add real-time customer risk analysis
- Deploy the application online

## Disclaimer
This project is developed for educational and research purposes. Predictions are not intended to be used as the sole basis for real financial or lending decisions.
