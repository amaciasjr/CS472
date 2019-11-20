import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin

class HACClustering(BaseEstimator,ClusterMixin):

    def __init__(self, k=3, link_type='single'): ## add parameters here
        """
        Args:
            k = how many final clusters to have
            link_type = single or complete. when combining two clusters use complete link or single link
        """
        self.link_type = link_type
        self.k = k
        self.cluster_sizes = []
        self.centroids = []
        self.sses = []


    def fit(self, X, y=None):
        """ Fit the data; In this lab this will make the K clusters :D
        Args:
            X (array-like): A 2D numpy array with the training data
            y (array-like): An optional argument. Clustering is usually unsupervised so you don't need labels
        Returns:
            self: this allows this to be chained, e.g. model.fit(X,y).predict(X_test)
        """
        # TODO: If complete link use Max Distance. If single link use Min distance.
        # TODO: Recalculate Matrix, until K clusters is reached.

        # Initialize clusters
        clusters = [[X[i]] for i in range(len(X))]

        # Setup initial matrix
        matrix = None
        for row_index in range(len(X)):
            curr_row = []
            for col_index in range(len(X)):

                if row_index != col_index:
                    # Get euclidean distance between row and col values.
                    row_value = X[row_index]
                    col_value = X[col_index]
                    dist_squared = (col_value - row_value) ** 2
                    dist_squared_sum = np.sum(dist_squared)
                    sqrt_dist_ss = np.sqrt(dist_squared_sum)
                    curr_row += [sqrt_dist_ss]
                else:
                    if self.link_type == 'single':
                        curr_row += [np.nan]
                    elif self.link_type == 'complete':
                        curr_row += [0]
                    else:
                        print("Invalid link type for HAC")
                        return

            if matrix is not None:
                matrix = np.append(matrix, [np.asarray(curr_row)], axis=0)
            else:
                matrix = [np.asarray(curr_row)]

        while len(clusters) > self.k:

            # 1) 'single link' deals with finding min distance between clusters, find index of min distance in matrix.
            # 2) 'complete link' deals with finding max distance between clusters, find index of max distance in matrix.
            if self.link_type == 'single':
                value = np.nanmin(matrix)
            elif self.link_type == 'complete':
                value = np.nanmax(matrix)
            else:
                print("Invalid link type for HAC")
                return

            # Get indexes of desired value
            indexes = np.where(matrix == value)
            row1_to_check = indexes[0][0]
            row2_to_check = indexes[0][1]

            # Get row/col to compare from matrix
            row1 = matrix[row1_to_check]
            row2 = matrix[row2_to_check]


            assert len(row1) == len(row2)

            # Delete col and row of larger value between row1 & row2
            index_to_del = max(row1_to_check, row2_to_check)
            index_to_keep = min(row1_to_check, row2_to_check)

            for index in range(len(row1)):
                if index != row1_to_check and index != row2_to_check:
                    value1 = row1[index]
                    value2 = row2[index]
                    if self.link_type == 'single':
                        best_value = min(value1, value2)
                    elif self.link_type == 'complete':
                        best_value = max(value1, value2)
                    else:
                        print("Invalid link type for HAC")
                        return

                    matrix[index_to_keep][index] = best_value
                    matrix[index][index_to_keep] = best_value
                else:
                    continue


            # Delete `index_to_del` row/col from Matrix
            matrix = np.delete(matrix, index_to_del, 0)
            matrix = np.delete(matrix, index_to_del, 1)

            # Combine new cluster
            for cluster in clusters[index_to_del]:
                clusters[index_to_keep].append(cluster)

            del clusters[index_to_del]

        # Calculate centroids, SSEs, and cluster sizes for clusters
        for cluster in clusters:
            centroid = self._calculateClusterCentroid(cluster)
            self.centroids.append(centroid)
            sse = self._calculateClusterSSE(cluster,centroid)
            self.sses.append(sse)
            self.cluster_sizes.append(len(cluster))

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
        f = open(filename, "w+")
        f.write("{:d}\n".format(self.k))
        f.write("{:.4f}\n\n".format(np.sum(self.sses)))
        # For each cluster and centroid

        for index in range(len(self.centroids)):
            f.write(np.array2string(self.centroids[index], precision=4, separator=","))
            f.write("\n")
            f.write("{:d}\n".format(int(self.cluster_sizes[index])))
            f.write("{:.4f}\n\n".format(self.sses[index]))

        f.close()


    def _calculateClusterSSE(self, cluster, centroid):
        """
        Used to calculate the SSE of each cluster for the file report.
        :param cluster:
        :param centroid_index:
        :return: (int) sse for cluster.
        """
        sse = np.sum((cluster - centroid) ** 2)
        return sse

    def _calculateClusterCentroid(self, cluster):
        """
        Used to calculate the SSE of each cluster for the file report.
        :param cluster:
        :return: (ndarray) centroid for cluster.
        """
        centroid = np.mean(cluster, axis=0)
        return centroid
