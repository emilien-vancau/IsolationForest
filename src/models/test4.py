import matplotlib.pyplot as plt
import numpy as np

# Générer des données aléatoires pour le scatter plot
np.random.seed(0)
x = np.random.rand(50)
y = np.random.rand(50)

# Définir la première ligne de séparation pour la région 1
x_line_1 = np.linspace(0.2, 0.8, 100)
y_line_1 = 0.5 * np.ones_like(x_line_1)  # Séparation au niveau de y=0.5 dans la région 1

# Définir la deuxième ligne de séparation pour la région 1
x_line_2 = np.linspace(0.4, 0.6, 100)
y_line_2 = np.linspace(0.3, 0.7, 100)

# Créer le scatter plot
plt.figure(figsize=(8, 6))
plt.scatter(x, y, color='blue', label='Points')

# Ajouter la première ligne de séparation pour la région 1
plt.plot(x_line_1, y_line_1, color='green', linestyle='--', label='Ligne de séparation (Région 1)')

# Ajouter la deuxième ligne de séparation pour la région 1
plt.plot(x_line_2, y_line_2, color='orange', linestyle='--', label='Ligne de séparation (Région 1)')

# Ajouter des labels pour les régions
plt.text(0.5, 0.25, 'Région 1A', fontsize=12, color='green', ha='center')
plt.text(0.5, 0.65, 'Région 1B', fontsize=12, color='orange', ha='center')
plt.text(0.5, 0.85, 'Région 2', fontsize=12, color='purple', ha='center')

# Ajouter les légendes et les titres
plt.legend()
plt.title('Scatter Plot avec lignes de séparation pour 3 régions')
plt.xlabel('X')
plt.ylabel('Y')
plt.grid(True)

# Afficher le plot
plt.show()
