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
y_pred_train_holdout = model.predict(X_train)
y_pred_test_holdout = model.predict(X_test)
accuracy_train_holdout = accuracy_score(y_train, y_pred_train_holdout)
accuracy_test_holdout = accuracy_score(y_test, y_pred_test_holdout)
overfitting_holdout = accuracy_train_holdout - accuracy_test_holdout
report_holdout = classification_report(y_test, y_pred_test_holdout, target_names=data.target_names, output_dict=True, labels=np.unique(y))
results.append({
    'CV Method': 'Hold-Out',
    'Accuracy': accuracy_test_holdout,
    'Precision (macro)': report_holdout['macro avg']['precision'],
    'Recall (macro)': report_holdout['macro avg']['recall'],
    'F1-score (macro)': report_holdout['macro avg']['f1-score'],
    'Overfitting': overfitting_holdout,
})

# Function to calculate cross-validation metrics with overfitting
def evaluate_cv_with_overfitting(cv, X, y, groups=None):
    accuracies_train = []
    accuracies_test = []
    precisions = []
    recalls = []
    f1_scores = []
    labels = np.unique(y)
    for train_index, test_index in cv.split(X, y, groups):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        model.fit(X_train, y_train)
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        accuracy_train = accuracy_score(y_train, y_pred_train)
        accuracy_test = accuracy_score(y_test, y_pred_test)
        report = classification_report(
            y_test,
            y_pred_test,
            target_names=data.target_names,
            output_dict=True,
            labels=labels
        )
        accuracies_train.append(accuracy_train)
        accuracies_test.append(accuracy_test)
        precisions.append(report['macro avg']['precision'])
        recalls.append(report['macro avg']['recall'])
        f1_scores.append(report['macro avg']['f1-score'])
    overfitting = np.mean(accuracies_train) - np.mean(accuracies_test)
    return np.mean(accuracies_test), np.mean(precisions), np.mean(recalls), np.mean(f1_scores), overfitting

# 2. K-Fold
kf = KFold(n_splits=5, shuffle=True, random_state=42)
acc_kf, prec_kf, rec_kf, f1_kf, over_kf = evaluate_cv_with_overfitting(kf, X, y)
results.append({
    'CV Method': 'K-Fold (k=5)',
    'Accuracy': acc_kf,
    'Precision (macro)': prec_kf,
    'Recall (macro)': rec_kf,
    'F1-score (macro)': f1_kf,
    'Overfitting': over_kf,
})

# 3. Leave-One-Out
loo = LeaveOneOut()
acc_loo, prec_loo, rec_loo, f1_loo, over_loo = evaluate_cv_with_overfitting(loo, X, y)
results.append({
    'CV Method': 'Leave-One-Out',
    'Accuracy': acc_loo,
    'Precision (macro)': prec_loo,
    'Recall (macro)': rec_loo,
    'F1-score (macro)': f1_loo,
    'Overfitting': over_loo,
})

# 4. Stratified K-Fold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
acc_skf, prec_skf, rec_skf, f1_skf, over_skf = evaluate_cv_with_overfitting(skf, X, y)
results.append({
    'CV Method': 'Stratified K-Fold (k=5)',
    'Accuracy': acc_skf,
    'Precision (macro)': prec_skf,
    'Recall (macro)': rec_skf,
    'F1-score (macro)': f1_skf,
    'Overfitting': over_skf,
})

# 5. Group K-Fold
groups = np.random.randint(0, 3, size=len(X))
gkf = GroupKFold(n_splits=3)
acc_gkf, prec_gkf, rec_gkf, f1_gkf, over_gkf = evaluate_cv_with_overfitting(gkf, X, y, groups=groups)
results.append({
    'CV Method': 'Group K-Fold (k=3)',
    'Accuracy': acc_gkf,
    'Precision (macro)': prec_gkf,
    'Recall (macro)': rec_gkf,
    'F1-score (macro)': f1_gkf,
    'Overfitting': over_gkf,
})

# 6. Nested Cross-Validation
outer_cv = KFold(n_splits=5, shuffle=True, random_state=42)
inner_cv = KFold(n_splits=3, shuffle=True, random_state=123)
nested_accuracies_test = []
nested_precisions = []
nested_recalls = []
nested_f1_scores = []
nested_overfitting = []
labels = np.unique(y)

for train_index_outer, test_index_outer in outer_cv.split(X, y):
    X_train_outer, X_test_outer = X[train_index_outer], X[test_index_outer]
    y_train_outer, y_test_outer = y[train_index_outer], y[test_index_outer]

    inner_accuracies_train = []
    inner_accuracies_test = []
    inner_precisions = []
    inner_recalls = []
    inner_f1_scores = []

    for train_index_inner, test_index_inner in inner_cv.split(X_train_outer, y_train_outer):
        X_train_inner, X_test_inner = X_train_outer[train_index_inner], X_train_outer[test_index_inner]
        y_train_inner, y_test_inner = y_train_outer[train_index_inner], y_train_outer[test_index_inner]

        model.fit(X_train_inner, y_train_inner)
        y_pred_train_inner = model.predict(X_train_inner)
        y_pred_test_inner = model.predict(X_test_inner)

        inner_accuracies_train.append(accuracy_score(y_train_inner, y_pred_train_inner))
        inner_accuracies_test.append(accuracy_score(y_test_inner, y_pred_test_inner))
        report_inner = classification_report(
            y_test_inner,
            y_pred_test_inner,
            target_names=data.target_names,
            output_dict=True,
            labels=labels
        )
        inner_precisions.append(report_inner['macro avg']['precision'])
        inner_recalls.append(report_inner['macro avg']['recall'])
        inner_f1_scores.append(report_inner['macro avg']['f1-score'])

    model.fit(X_train_outer, y_train_outer)
    y_pred_outer = model.predict(X_test_outer)
    nested_accuracies_test.append(accuracy_score(y_test_outer, y_pred_outer))
    report_outer = classification_report(y_test_outer, y_pred_outer, target_names=data.target_names, output_dict=True, labels=labels)
    nested_precisions.append(report_outer['macro avg']['precision'])
    nested_recalls.append(report_outer['macro avg']['recall'])
    nested_f1_scores.append(report_outer['macro avg']['f1-score'])
    nested_overfitting.append(np.mean(inner_accuracies_train) - np.mean(inner_accuracies_test))


results.append({
    'CV Method': 'Nested CV (outer k=5, inner k=3)',
    'Accuracy': np.mean(nested_accuracies_test),
    'Precision (macro)': np.mean(nested_precisions),
    'Recall (macro)': np.mean(nested_recalls),
    'F1-score (macro)': np.mean(nested_f1_scores),
    'Overfitting': np.mean(nested_overfitting),
})

# Create a Pandas DataFrame for better table formatting
df_results = pd.DataFrame(results)

# Print the table
print(df_results)