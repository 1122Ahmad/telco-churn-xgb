Telecom Customer Churn Prediction — XGBoost + SHAP
Business Report (1 Page)
1. Objective
The goal of this project is to predict customer churn for a telecom service provider using historical customer usage and account information. The model enables the business to identify customers likely to cancel their service and take proactive retention actions.

2. Dataset Summary

Source: IBM Telco Customer dataset
Rows: 7,043 customers
Features: 21 customer attributes
Target: Churn (Yes/No)
Includes contract type, monthly charges, internet service, security services, and tenure.


3. Modeling Approach


Preprocessing Pipeline (scikit‑learn)

Numeric features scaled
Categorical features one‑hot encoded (handle_unknown="ignore")
Missing values in TotalCharges fixed



Model Used:

XGBoost Classifier, tuned using RandomizedSearchCV



Imbalance Handling:

Applied scale_pos_weight = (negatives / positives)
This improves model sensitivity to minority churn cases.




4. Key Performance Metrics (Test Set)

ROC‑AUC: 0.840
PR‑AUC: 0.656
Accuracy: 0.727
Churn Recall: 0.813 (model catches 81% of churners)
Churn Precision: 0.492
F1‑Score (Churn): 0.613

Interpretation:

High recall is preferred because missing a churner costs more than contacting a non‑churner.
These results align with telecom industry benchmarks.


5. Confusion Matrix Insight




















Predicted NoPredicted YesActual No719314Actual Yes70304

The model correctly identifies 304 churners.
Only 70 churners were missed (False Negatives).
314 non‑churners were incorrectly flagged → acceptable for low-cost retention actions.


6. Explainability (SHAP Values)
Top drivers of churn:

Month‑to‑month contract
Low tenure
No Online Security
High Monthly Charges
Fiber Optic Internet
PaymentMethod: Electronic Check

Local SHAP explanations show why an individual customer was predicted to churn.

7. Business Recommendations

Prioritize customers with month‑to‑month contracts and high charges for retention offers.
Offer security/tech support bundles to reduce churn risk.
Target top 20% highest‑risk customers using model probabilities.
Use SHAP insights to design data-driven retention strategies.


8. Artifacts Produced

Tuned model: xgb_churn_model.pkl
Pipeline: preprocessing + XGBoost
Evaluation plots: ROC, PR, Confusion Matrix
SHAP global + local explanations
Fully modular code under src/