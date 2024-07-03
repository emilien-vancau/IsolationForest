import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm

# Générer des données de test
np.random.seed(0)
X = 0.3 * np.random.randn(100, 2)
X_train = np.r_[X + 2, X - 2]

# Entrainer le modèle OCSVM
clf = svm.OneClassSVM(nu=0.1, kernel="rbf", gamma=0.1)
clf.fit(X_train)

# Afficher la frontière de décision
xx, yy = np.meshgrid(np.linspace(-5, 5, 500), np.linspace(-5, 5, 500))
Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.contour(xx, yy, Z, levels=[0], linewidths=2, colors='black')
plt.scatter(X_train[:, 0], X_train[:, 1], color='red')
plt.xlim(-5, 5)
plt.ylim(-5, 5)
plt.title("Frontière de décision de l'OCSVM")
plt.show()