import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Génération de données aléatoires pour l'exemple
np.random.seed(0)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

# Entraînement d'un modèle de régression linéaire
model = LinearRegression()
model.fit(X, y)

# Prédictions du modèle
y_pred = model.predict(X)

# Plot des résultats
plt.figure(figsize=(10, 5))

# Plot des données réelles
plt.scatter(X, y, color='blue', label='Données réelles')

# Plot des valeurs prédites
plt.plot(X, y_pred, color='red', label='Valeurs prédites')

# Tracer les lignes d'erreur résiduelle
for i in range(len(X)):
    plt.plot([X[i], X[i]], [y[i], y_pred[i]], color='black', linestyle='dotted', linewidth=0.5)

plt.title('Régression linéaire simple')
plt.xlabel('X')
plt.ylabel('y')
plt.legend(['Valeurs réelles', 'Valeurs prédites', 'Erreur résiduelle'])
plt.grid(True)
fig = plt.gcf()
fig.savefig("simple linear regression")
plt.show()