import numpy as np
from sklearn.neighbors import KNeighborsClassifier as KNN
import numpy as np


# Импорт фичей
X_train_numpy = np.genfromtxt("X_train_V2_L_480.csv", delimiter=",")
X_test_numpy = np.genfromtxt("X_test_V2_L_480.csv", delimiter=",")
y_train_numpy = np.genfromtxt("y_train_V2_L_480.csv", delimiter=",")
y_test_numpy = np.genfromtxt("y_test_V2_L_480.csv", delimiter=",")

n_neig = 3
weight = 'distance'
metric = 'manhattan'

knn = KNN(n_neighbors=n_neig, weights=weight, metric=metric)
knn.fit(X_train_numpy, y_train_numpy)

y_pred = knn.predict(X_test_numpy)
accuracy = np.mean([y_pred == y_test_numpy])

print(f"Classes: {knn.classes_}")
print(f"Num of features: {knn.n_features_in_}")
print(f"Accuracy: {accuracy}")
