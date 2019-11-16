import numpy as np
import random
from sklearn.base import BaseEstimator, ClusterMixin

class KMEANSClustering(BaseEstimator, ClusterMixin):

    def __init__(self, k=3, debug=False): ## add parameters here
        """
        Args:
            k = how many final clusters to have
            debug = if debug is true use the first k instances as the initial centroids otherwise choose random points as the initial centroids.
        """
        self.k = k
        self.debug = debug
        self.centroids = []


    def fit(self, X, y=None):
        """ Fit the data; In this lab this will make the K clusters :D
            Train until convergence (centroids have stopped changing).
        Args:
            X (array-like): A 2D numpy array with the training data
            y (array-like): An optional argument. Clustering is usually unsupervised so you don't need labels
        Returns:
            self: this allows this to be chained, e.g. model.fit(X,y).predict(X_test)
        """

        # Initialization of centroids, either first k instances or randomly selected k instances.
        if self.debug:
            self.centroids = X[:self.k]
        else:
            self.centroids = np.asarray(random.choices(X, k=self.k))

        # Go through each instance and see which centroid has the shortest euclidean distance from it.
        distances = np.zeros((np.shape(X)[0],)).reshape((np.shape(X)[0],1))
        for centroid in self.centroids:
            data_copy = X
            distance = (data_copy - centroid) ** 2
            dist_sums = np.sum(distance, axis=1)
            dist_sqrt_sums = np.sqrt(dist_sums).reshape((np.shape(dist_sums)[0],1))
            distances = np.concatenate((distances,dist_sqrt_sums), axis=1)
        distances = distances[:,1:]

        # Gather index of points closest to each centroid
        centroid_groups = dict()
        for row in range(distances.shape[0]):
            instance = distances[row]
            closest_centroid = np.argmin(instance)
            if closest_centroid in centroid_groups:
                centroid_groups[closest_centroid] += [row]
            else:
                centroid_groups[closest_centroid]= []
                centroid_groups[closest_centroid].append(row)

        # TODO: Create clusters around each centroid
        for indicies in centroid_groups.values():
            pass

        # TODO: From current clusters, calculate new centroids.

        return self


    def save_clusters(self, filename):
        """
            f = open(filename,"w+") 
            Used for grading.
            write("{:d}\n".format(k))
            write("{:.4f}\n\n".format(total SSE))
            for each cluster and centroid:
                write(np.array2string(centroid,precision=4,seperator=","))
                write("\n")
                write("{:d}\n".format(size of cluster))
                write("{:.4f}\n\n".format(SSE of cluster))
            f.close()
        """
