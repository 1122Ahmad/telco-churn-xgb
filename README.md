# Telco Customer Churn Prediction (XGBoost + SHAP)

## Project Overview
This project predicts customer churn for a telecom provider using the IBM Telco Customer Churn dataset.
It includes a production-style ML workflow: preprocessing, model tuning, evaluation, and SHAP explainability.

### Current model performance (test set)
- ROC-AUC: **0.840**
- PR-AUC: **0.656**
- Recall (Churn): **0.813**

These are strong results for churn prediction, where recall is often prioritized because missing churners is costly.

## Key Features
- End-to-end scikit-learn `Pipeline`
- Tuned `XGBClassifier` with `RandomizedSearchCV`
- Class imbalance handling using `scale_pos_weight`
- Explainability with SHAP
- Evaluation plots (confusion matrix, ROC, PR)
- Exported production model (`.pkl`)

## Project Structure
```text
telco-churn-xgb/
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
├── models/
│   └── xgb_churn_model.pkl
├── notebooks/
│   └── train.ipynb
├── reports/
│   ├── Telco_Churn_Report.md
│   └── figures/
│       ├── confusion_matrix.png
│       ├── confusion_matrix_percent.png
│       ├── pr_curve.png
│       ├── roc_curve.png
│       ├── shap_bar.png
│       ├── shap_beeswarm.png
│       └── xgb_feature_importance.png
├── src/
│   ├── evaluation.py
│   ├── model.py
│   ├── preprocessing.py
│   └── utils.py
├── requirements.txt
└── train.py
```

## How to Run
```bash
pip install -r requirements.txt
python train.py
```

### Optional training flags
```bash
python train.py --n-iter 20 --cv 3
```

## Business Impact
The model helps retention teams identify high-risk customers and prioritize proactive interventions.

## Future Improvements
- Add a Streamlit app for interactive scoring
- Add threshold optimization based on retention economics
- Package inference with FastAPI
