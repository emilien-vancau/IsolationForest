import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.tree import DecisionTreeClassifier

# Générer des données aléatoires
X, y = make_classification(n_samples=100, n_features=2, n_redundant=0, n_clusters_per_class=1, random_state=42)

# Entraîner un arbre de décision avec une plus grande profondeur
tree = DecisionTreeClassifier(random_state=42, max_depth=5, min_samples_split=2, min_samples_leaf=1)
tree.fit(X, y)

# Définir les limites du graphique
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

# Créer une grille de points
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 500),
                     np.linspace(y_min, y_max, 500))

# Fonction récursive pour tracer les droites de séparation
def plot_decision_boundaries(tree, node_id=0, depth=0, x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max):
    if node_id == -1:
        return

    # Obtenir les informations du nœud actuel
    if tree.tree_.children_left[node_id] != tree.tree_.children_right[node_id]:  # Si le nœud n'est pas une feuille
        feature = tree.tree_.feature[node_id]
        threshold = tree.tree_.threshold[node_id]

        if feature == 0:  # Division sur la première caractéristique
            plt.plot([threshold, threshold], [y_min, y_max], 'k--', lw=0.5)
            plot_decision_boundaries(tree, tree.tree_.children_left[node_id], depth+1, x_min, threshold, y_min, y_max)
            plot_decision_boundaries(tree, tree.tree_.children_right[node_id], depth+1, threshold, x_max, y_min, y_max)
        else:  # Division sur la deuxième caractéristique
            plt.plot([x_min, x_max], [threshold, threshold], 'k--', lw=0.5)
            plot_decision_boundaries(tree, tree.tree_.children_left[node_id], depth+1, x_min, x_max, y_min, threshold)
            plot_decision_boundaries(tree, tree.tree_.children_right[node_id], depth+1, x_min, x_max, threshold, y_max)

# Tracer les points de données
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', s=20, edgecolor='k')

# Tracer les droites de séparation
plot_decision_boundaries(tree)

plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Decision Boundaries of a Deeper Decision Tree')
plt.show()
