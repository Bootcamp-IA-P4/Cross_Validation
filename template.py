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