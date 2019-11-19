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


    def fit(self, X, y=None):
        """ Fit the data; In this lab this will make the K clusters :D
        Args:
            X (array-like): A 2D numpy array with the training data
            y (array-like): An optional argument. Clustering is usually unsupervised so you don't need labels
        Returns:
            self: this allows this to be chained, e.g. model.fit(X,y).predict(X_test)
        """
        # TODO: If complete link use Max Distance. If single link use Min distance.
        # TODO: Update Matrix, until K clusters is reached.

        # Setup matrix
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
                    curr_row += [np.inf]

            if matrix is not None:
                matrix = np.append(matrix, [np.asarray(curr_row)], axis=0)
            else:
                matrix = [np.asarray(curr_row)]


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


