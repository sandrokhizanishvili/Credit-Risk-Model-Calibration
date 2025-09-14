import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score,
    balanced_accuracy_score,
    log_loss,
    precision_score,
    recall_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    brier_score_loss,

)

from venn_abers import VennAbersCalibrator
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
import joblib


# Read CSV file
df_test = pd.read_csv("DATA/preprocessed_test.csv")
df_train = pd.read_csv("DATA/preprocessed_train.csv")
df_valid = pd.read_csv("DATA/preprocessed_valid.csv")


X_train = df_train.drop(columns=["loan_status", "ID"])
y_train = df_train["loan_status"]

X_valid = df_valid.drop(columns=["loan_status", "ID"])
y_valid = df_valid["loan_status"]

X_test  = df_test.drop(columns=["loan_status", "ID"])
y_test  = df_test["loan_status"]



# Scale features 
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_valid = scaler.transform(X_valid)
X_test  = scaler.transform(X_test)


X_combined = np.vstack([X_train, X_valid])
y_combined = np.concatenate([y_train, y_valid])

# Build logistic regression model
model = LogisticRegression(max_iter=1000, solver="lbfgs")

# Train
model.fit(X_train, y_train)

# Validate
y_pred_valid = model.predict(X_valid)
print("\nValidation Accuracy:", round(accuracy_score(y_valid, y_pred_valid),4))

# Test
y_pred_test = model.predict(X_test)
print("\nTest Accuracy:", round(accuracy_score(y_test, y_pred_test),4))


proba_raw_test = model.predict_proba(X_test)[:, 1]
proba_raw_valid = model.predict_proba(X_valid)[:, 1]


platt = CalibratedClassifierCV(estimator=model, method="sigmoid", cv="prefit")
platt.fit(X_valid, y_valid)


proba_platt_test = platt.predict_proba(X_test)[:, 1]



isotonic = CalibratedClassifierCV(estimator=model, method="isotonic", cv="prefit")
isotonic.fit(X_valid, y_valid)

proba_iso_test = isotonic.predict_proba(X_test)[:, 1]


va_inductive = VennAbersCalibrator(estimator=model, inductive=True, cal_size=None, random_state=42)
va_inductive.fit(X_valid, y_valid)

p_cal = model.predict_proba(X_valid)
p_test = model.predict_proba(X_test)


VAC = VennAbersCalibrator()


# --- Calibrated probabilities & intervals for the TEST set ---
preds_test, intervals_test = VAC.predict_proba(
    p_cal=p_cal,
    y_cal=y_valid.values,
    p_test=p_test,
    p0_p1_output=True
)


proba_IVAP_test = preds_test[:, 1]
p1_IVAP_test=intervals_test[:, 1]
p0_IVAP_test=intervals_test[:, 0]


model1 = LogisticRegression(max_iter=100, solver="lbfgs")

va_cross = VennAbersCalibrator(
    estimator=model1,  
    inductive=False,
    n_splits=5,          
    random_state=42
)
va_cross.fit(X_combined, y_combined) #combine train and valid 


predictions_cross_test, p_va_cross_p0_p1 = va_cross.predict_proba(X_test, p0_p1_output=True)
proba_va_cross_test= predictions_cross_test[:, 1]


# --- Save all predictions and scores for final report TEST SET ---
df_output = pd.DataFrame({
    "ID": df_test["ID"],
    "loan_status": y_test,
    "raw_score": proba_raw_test,
    "predicted_class_raw":y_pred_test, #added
    "platt_score": proba_platt_test,
    "isotonic_score": proba_iso_test,
    "venn_abers_ind_": proba_IVAP_test,
    "p0_va_ind" : p0_IVAP_test,
    "p1_va_ind" : p1_IVAP_test,
    "venn_abers_cross_score": proba_va_cross_test,

})

df_output.to_csv("/Users/elisaterzini/Desktop/Credit-Risk-Model-Calibration/DATA/test_predictions_and_calibration.csv", index=False)
print("Saved predictions to 'test_final_predictions_and_calibration.csv'")