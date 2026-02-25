📄 README.md — Telco Customer Churn Prediction (XGBoost + SHAP)


📌 Project Overview
This project predicts customer churn for a telecom provider using the IBM Telco Customer dataset.
It uses a production‑grade ML pipeline including preprocessing, model tuning, evaluation, and SHAP explainability.
The final model achieves:

ROC‑AUC: 0.840
PR‑AUC: 0.656
Recall (Churn): 0.813

These results are strong for churn prediction where recall is more important because missing churners is costly.

🧠 Key Features

End‑to‑end scikit‑learn Pipeline
Tuned XGBoost classifier
Class imbalance handling (scale_pos_weight)
Explainability with SHAP
Confusion matrix, ROC, PR curves
Threshold selection for business use
Saved production model (.pkl)


📂 Project Structure

telco-churn-xgb/
 data/ WA_Fn-UseC_-Telco-Customer-Churn.csv

notebooks/ train.ipynb

src/
 preprocessing.py

src/
model.py

src/
evaluation.py

src/
utils.py

models/
xgb_churn_model.pkl

reports/
figures/
 confusion_matrix.png

reports/
figures/
confusion_matrix_percent.png

reports/
figures/
roc_curve.png

reports/
figures/
pr_curve.png

reports/
figures/
shap_beeswarm.png

reports/
figures/
shap_bar.png

reports/
figures/
xgb_feature_importance.png

reports/
Telco_Churn_Report.pdf

telco-churn-xgb/
requirements.txt

telco-churn-xgb/
README.md




🚀 How to Run
pip install -r requirements.txt
python train.py


📝 Future Improvements
Add Streamlit app for live scoring
Add hyperparameter search via Optuna
Deploy model with FastAPI