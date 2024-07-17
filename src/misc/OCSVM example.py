import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import datasets
from sklearn import svm
from sklearn.preprocessing import MinMaxScaler
from sklearn.manifold import TSNE

iris = datasets.load_iris()
X = pd.DataFrame(data=iris.data, columns=iris.feature_names)
y = iris.target

X_scaled = MinMaxScaler().fit_transform(X)
X_reduced = TSNE(n_components=2, random_state=0).fit_transform(X_scaled)

clf = svm.OneClassSVM(nu=0.1, kernel="rbf", gamma=0.1)
clf.fit(X_reduced)

x_predicted = clf.predict(X_reduced)

x_min, x_max = X_reduced[:, 0].min() - 5, X_reduced[:, 0].max() + 5
y_min, y_max = X_reduced[:, 1].min() - 5, X_reduced[:, 1].max() + 5

x_ = np.linspace(x_min, x_max, 500)
y_ = np.linspace(y_min, y_max, 500)

xx, yy = np.meshgrid(x_, y_)

z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
z = z.reshape(xx.shape)

plt.contourf(xx, yy, z, cmap='Blues')
a = plt.contour(xx, yy, z, levels=[0], linewidths=2, colors='darkred')
b = plt.scatter(X_reduced[x_predicted == 1, 0], X_reduced[x_predicted == 1, 1], c='white', edgecolors='k')
c = plt.scatter(X_reduced[x_predicted == -1, 0], X_reduced[x_predicted == -1, 1], c='gold', edgecolors='k')
plt.legend([b, c], ['inliers', 'outliers'], bbox_to_anchor=(1.05, 1))
plt.axis('tight')
plt.title('OCSVM example iris dataset')

fig = plt.gcf()

plt.show()

fig.savefig("OCSVM boundary decision example")
