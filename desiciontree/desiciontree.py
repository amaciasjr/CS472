import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from arff import Arff

### NOTE: The only methods you are required to have are:
#   * predict
#   * fit
#   * score

class DTClassifier(BaseEstimator,ClassifierMixin):

    def __init__(self, counts):
        """ Initialize class with chosen hyperparameters.

        Args:
            hidden_layer_widths (list(int)): A list of integers which defines the width of each hidden layer
            lr (float): A learning rate / step size.
            shuffle: Whether to shuffle the training data each epoch. DO NOT SHUFFLE for evaluation / debug datasets.

        Example:
            DT  = DTClassifier()
        """
        self.counts = counts

    def fit(self, X, y):
        """ Fit the data; Make the Desicion tree

        Args:
            X (array-like): A 2D numpy array with the training data, excluding targets
            y (array-like): A 2D numpy array with the training targets

        Returns:
            self: this allows this to be chained, e.g. model.fit(X,y).predict(X_test)

        """
        features = X
        classes = y


        return self

    def predict(self, X):
        """ Predict all classes for a dataset X

        Args:
            X (array-like): A 2D numpy array with the training data, excluding targets

        Returns:
            array, shape (n_samples,)
                Predicted target values per element in X.
        """
        pass


    def score(self, X, y):
        """ Return accuracy of model on a given dataset. Must implement own score function.

        Args:
            X (array-like): A 2D numpy array with data, excluding targets
            y (array-li    def _shuffle_data(self, X, y):
        """
        return 0

    def _calc_entropy(self, p):
        if p != 0:
            return -p * np.log2(p)
        else:
            return 0


if __name__ == '__main__':

    # Debug Arff Paths
    arff_path = r"../data/decisiontree/debug/lenses.arff"
    arff_path2 = r"../data/decisiontree/debug/all_lenses.arff"

    # Evaluation Arff Paths
    # arff_path = r"../data/decisiontree/evaluation/zoo.arff"
    # arff_path2 = r"../data/decisiontree/evaluation/all_zoo.arff"

    mat = Arff(arff_path)

    counts = []  ## this is so you know how many types for each column

    for i in range(mat.data.shape[1]):
        counts += [mat.unique_value_count(i)]
    data = mat.data[:, 0:-1]
    labels = mat.data[:, -1].reshape(-1, 1)
    DTClass = DTClassifier(counts)
    DTClass.fit(data, labels)
    # mat2 = Arff(arff_path2)
    # data2 = mat2.data[:, 0:-1]
    # labels2 = mat2.data[:, -1]
    # pred = DTClass.predict(data2)
    # Acc = DTClass.score(data2, labels2)
    # np.savetxt("pred_lenses.csv", pred, delimiter=",")
    # print("Accuracy = [{:.2f}]".format(Acc))
