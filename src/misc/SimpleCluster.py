import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

X, _ = make_blobs(n_samples=120, centers=1, cluster_std=0.6, random_state=42)

plt.scatter(X[:,0], X[:,1], c='b', s=20)
#plt.title('Simple cluster')
plt.xlabel('X')
plt.ylabel('Y')

fig = plt.gcf()
plt.show()

fig.savefig("simple normal cluster")