from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, KFold, LeaveOneOut, StratifiedKFold, GroupKFold
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd

# Cargar dataset Iris
data = load_iris()
X, y = data.data, data.target

# Crear modelo
model = RandomForestClassifier(random_state=42)

results = []

# 1. Hold-Out
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
model.fit(X_train, y_train)
y_pred_holdout = model.predict(X_test)
accuracy_holdout = accuracy_score(y_test, y_pred_holdout)
report_holdout = classification_report(y_test, y_pred_holdout, target_names=data.target_names, output_dict=True)
results.append({
    'CV Method': 'Hold-Out',
    'Accuracy': accuracy_holdout,
    'Precision (macro)': report_holdout['macro avg']['precision'],
    'Recall (macro)': report_holdout['macro avg']['recall'],
    'F1-score (macro)': report_holdout['macro avg']['f1-score'],
})

# Function to calculate cross-validation metrics
def evaluate_cv(cv, X, y, groups=None):
    accuracies = []
    precisions = []
    recalls = []
    f1_scores = []
    labels = np.unique(y)  # Get all unique class labels
    for train_index, test_index in cv.split(X, y, groups):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(
            y_test,
            y_pred,
            target_names=data.target_names,
            output_dict=True,
            labels=labels  # Explicitly provide all possible labels
        )
        accuracies.append(accuracy)
        precisions.append(report['macro avg']['precision'])
        recalls.append(report['macro avg']['recall'])
        f1_scores.append(report['macro avg']['f1-score'])
    return np.mean(accuracies), np.mean(precisions), np.mean(recalls), np.mean(f1_scores)
# 2. K-Fold
kf = KFold(n_splits=5, shuffle=True, random_state=42)
acc_kf, prec_kf, rec_kf, f1_kf = evaluate_cv(kf, X, y)
results.append({
    'CV Method': 'K-Fold (k=5)',
    'Accuracy': acc_kf,
    'Precision (macro)': prec_kf,
    'Recall (macro)': rec_kf,
    'F1-score (macro)': f1_kf,
})

# 3. Leave-One-Out
loo = LeaveOneOut()
acc_loo, prec_loo, rec_loo, f1_loo = evaluate_cv(loo, X, y)
results.append({
    'CV Method': 'Leave-One-Out',
    'Accuracy': acc_loo,
    'Precision (macro)': prec_loo,
    'Recall (macro)': rec_loo,
    'F1-score (macro)': f1_loo,
})

# 4. Stratified K-Fold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
acc_skf, prec_skf, rec_skf, f1_skf = evaluate_cv(skf, X, y)
results.append({
    'CV Method': 'Stratified K-Fold (k=5)',
    'Accuracy': acc_skf,
    'Precision (macro)': prec_skf,
    'Recall (macro)': rec_skf,
    'F1-score (macro)': f1_skf,
})

# 5. Group K-Fold
groups = np.random.randint(0, 3, size=len(X))
gkf = GroupKFold(n_splits=3)
acc_gkf, prec_gkf, rec_gkf, f1_gkf = evaluate_cv(gkf, X, y, groups=groups)
results.append({
    'CV Method': 'Group K-Fold (k=3)',
    'Accuracy': acc_gkf,
    'Precision (macro)': prec_gkf,
    'Recall (macro)': rec_gkf,
    'F1-score (macro)': f1_gkf,
})

# 6. Nested Cross-Validation (still using cross_val_score for simplicity of inner loop)
outer_cv = KFold(n_splits=5, shuffle=True, random_state=42)
inner_cv = KFold(n_splits=3, shuffle=True, random_state=123)
nested_accuracies = []
nested_precisions = []
nested_recalls = []
nested_f1_scores = []

for train_index, test_index in outer_cv.split(X, y):
    X_train_outer, X_test_outer = X[train_index], X[test_index]
    y_train_outer, y_test_outer = y[train_index], y[test_index]

    inner_acc, inner_prec, inner_rec, inner_f1 = evaluate_cv(inner_cv, X_train_outer, y_train_outer)
    nested_accuracies.append(inner_acc)
    nested_precisions.append(inner_prec)
    nested_recalls.append(inner_rec)
    nested_f1_scores.append(inner_f1)

results.append({
    'CV Method': 'Nested CV (outer k=5, inner k=3)',
    'Accuracy': np.mean(nested_accuracies),
    'Precision (macro)': np.mean(nested_precisions),
    'Recall (macro)': np.mean(nested_recalls),
    'F1-score (macro)': np.mean(nested_f1_scores),
})

# Create a Pandas DataFrame for better table formatting
df_results = pd.DataFrame(results)

# Print the table
print(df_results)