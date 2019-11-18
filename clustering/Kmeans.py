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
        iterations = 0
        centroids_changed = True
        while centroids_changed:
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

            counter = 0
            same_centroids = 0
            for indexes in centroid_groups.values():
                # Create clusters around each centroid
                cluster = X[indexes]
                # From current clusters, calculate new centroids.
                new_centroid = np.mean(cluster,axis=0)

                if not np.array_equal(new_centroid, self.centroids[counter]):
                    print(f"Updating Centroid {counter}...")
                    self.centroids[counter] = new_centroid
                else:
                    print(f"Centroid {counter} did not change!")
                    same_centroids += 1

                counter += 1

            if same_centroids == len(self.centroids):
                centroids_changed = False

            iterations += 1

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
