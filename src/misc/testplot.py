import matplotlib.pyplot as plt
import numpy as np


def plot_auc_contamination(data, labels, colormap='viridis'):

    plt.figure(figsize=(10, 6))

    # Create a colormap
    cmap = plt.get_cmap(colormap)
    colors = cmap(np.linspace(0, 1, len(data)))

    for i, (contamination_levels, auc_values) in enumerate(data):
        plt.plot(contamination_levels, auc_values, label=labels[i], marker='o', color=colors[i])

    plt.xlabel('Niveau de contamination')
    plt.ylabel('AUC')
    plt.title('AUC par niveau de contamination')
    plt.legend()
    plt.grid(True)

    fig = plt.gcf()

    plt.close()

    return fig


# Exemple de données
data = [
    ([0.01, 0.02, 0.05, 0.1], [0.95, 0.93, 0.90, 0.85]),
    ([0.01, 0.02, 0.05, 0.1], [0.92, 0.91, 0.88, 0.80]),
    ([0.01, 0.02, 0.05, 0.1], [0.90, 0.89, 0.87, 0.83]),
    ([0.01, 0.02, 0.05, 0.1], [0.93, 0.92, 0.89, 0.84]),
    ([0.01, 0.02, 0.05, 0.1], [0.94, 0.90, 0.86, 0.82]),
    ([0.01, 0.02, 0.05, 0.1], [0.91, 0.88, 0.85, 0.79])
]

labels = ['Height limit 1', 'Height limit 2', 'Height limit 3', 'Height limit 4', 'Height limit 5', 'Height limit 6']

plot_auc_contamination(data, labels)
