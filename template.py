from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
import numpy as np

# Cargar dataset Iris
data = load_iris()
X, y = data.data, data.target

# Crear modelo
model = DecisionTreeClassifier()

# Validación cruzada con 5 folds, cross_val_score es una función que usa k-fold por defecto o stratified k-fold según el problema.
# scores = cross_val_score(model, X, y, cv=5)

kf = KFold(n_splits=5, shuffle=True, random_state=42)
scores=[]

for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    scores.append(acc)

print("Scores de cada fold:", scores)
print("Accuracy promedio:", np.mean(scores))



# Ejecutar el código tal cual.
# Cambiar el modelo a otro (por ejemplo RandomForest).
# Cambiar el número de folds (cv=3, cv=10).
# Observar cómo cambian los resultados.


# -----------------------------------------------------------
# Otras técnicas de validación cruzada que puedes probar:
# 
# 1️⃣ **StratifiedKFold**:  
#     - Asegura que la proporción de clases se mantenga en cada fold.
#     - Útil para datasets desbalanceados.
#     - Para implementarlo:  
#         from sklearn.model_selection import StratifiedKFold  
#         kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
#
# 2️⃣ **Leave-One-Out (LOO)**:  
#     - Crea tantos folds como muestras (cada muestra es test una vez).
#     - Muy costoso para datasets grandes.
#     - Para usarlo:  
#         from sklearn.model_selection import LeaveOneOut  
#         loo = LeaveOneOut()
#
# 3️⃣ **Leave-P-Out (LPO)**:  
#     - Parecido al anterior, pero dejando fuera *p* muestras en cada iteración.
#     - Para usarlo:  
#         from sklearn.model_selection import LeavePOut  
#         lpo = LeavePOut(p=2)  # Por ejemplo, deja 2 muestras fuera cada vez
#
# 4️⃣ **GroupKFold** (si tienes grupos):  
#     - Divide por grupos (por ejemplo, por paciente o por empresa).
#     - Se usa si tienes una variable 'groups' para indicar el grupo de cada muestra.
#     - Para usarlo:  
#         from sklearn.model_selection import GroupKFold  
#         gkf = GroupKFold(n_splits=5)  
#         Luego debes pasar el parámetro 'groups' en .split(X, y, groups)
#
# 5️⃣ **RepeatedKFold**:  
#     - Repite el K-Fold varias veces, con diferentes particiones aleatorias.
#     - Para usarlo:  
#         from sklearn.model_selection import RepeatedKFold  
#         rkf = RepeatedKFold(n_splits=5, n_repeats=3, random_state=42)
#
# -----------------------------------------------------------
#
# PASOS A SEGUIR (si quieres probar otras técnicas):
#
# ✅ Cambiar la técnica de validación cruzada:
#     - Sustituye 'KFold' por la técnica que quieras probar.
#     - Ajusta los parámetros necesarios (por ejemplo, n_splits, p, n_repeats, etc.).
#     - Si usas 'GroupKFold', necesitarás definir el vector 'groups' y pasarlo al método .split().
#
# ✅ Probar otros modelos:
#     - Sustituye 'DecisionTreeClassifier()' por otro, por ejemplo, 'RandomForestClassifier()'.
#
# ✅ Cambiar el número de folds:
#     - Ajusta el valor de 'n_splits' en KFold, StratifiedKFold, etc. (por ejemplo, n_splits=3 o 10).
#
# ✅ Ejecutar el código:
#     - Observa cómo cambian los resultados de accuracy.
#
# ✅ Reflexión final:
#     - Analiza cómo afecta la técnica de validación cruzada a la variabilidad de los resultados.
#     - Considera cuál es más adecuada para tu caso de uso.