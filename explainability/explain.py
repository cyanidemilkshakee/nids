import shap
import pandas as pd
import joblib

df = pd.read_csv('../data/processed/unsw_selected_features.csv')  # or cicids_selected_features.csv
X = df.drop('label', axis=1)
model = joblib.load('../models/saved_model.joblib')

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X.iloc[:100])  # Explain 100 samples

shap.summary_plot(shap_values, X.iloc[:100])
