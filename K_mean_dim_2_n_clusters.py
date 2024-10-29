import numpy as np
import matplotlib.pyplot as plt
from K_mean_functions import *


# Clusters generation

spread = (1, 1)

cluster_1 = generate_cluster(mean=(4, 3), spread=spread, numb_points=100, dim=2)
cluster_2 = generate_cluster(mean=(3, -4), spread=spread, numb_points=100, dim=2)
cluster_3 = generate_cluster(mean=(-2, 4), spread=spread, numb_points=100, dim=2)
cluster_4 = generate_cluster(mean=(0, 0), spread=spread, numb_points=100, dim=2)
cluster_5 = generate_cluster(mean=(-2, -4), spread=spread, numb_points=100, dim=2)
cluster_6 = generate_cluster(mean=(-4, 0), spread=spread, numb_points=100, dim=2)

clusters = (cluster_1, cluster_2, cluster_3, cluster_4, cluster_5, cluster_6)

# Clusters visualisation

for cluster in clusters:
    plt.plot(np.transpose(cluster)[0], np.transpose(cluster)[1], "o")

plt.title("Generated clusters")
plt.savefig("Clusters_6.pdf")
plt.show()

# Contatenation and shuffle of the points

points = np.concatenate(clusters)
np.random.shuffle(points)


# Computation of the K-means on the cluster

centers, clusters_points, cluster_index = K_means(
    points=points, nb_clust=6, max_it=30, stop_val=0.1, show_evolution=True
)

# Visualisation of the K-means result

for i in range(len(centers)):
    cluster = np.array(clusters_points[i])
    plt.plot(
        np.transpose(cluster)[0],
        np.transpose(cluster)[1],
        "x",
        # label=f"cluster {i}",
    )
    plt.plot([centers[i][0]], [centers[i][1]], "o", label=f"center {i}")

plt.legend()
plt.title("Computed clusters w/ centers")
plt.savefig("Clusters_6_final.pdf")
plt.show()
