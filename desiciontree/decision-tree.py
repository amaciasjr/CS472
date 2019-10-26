import numpy as np
from math import log2
from sklearn.base import BaseEstimator, ClassifierMixin
from arff import Arff
from tree import Tree

### NOTE: The only methods you are required to have are:
#   * predict
#   * fit
#   * score

class DTClassifier(BaseEstimator, ClassifierMixin):


    def __init__(self, counts, validation_size = 0.0):


        """ Initialize class with chosen hyperparameters.

        Args:
            hidden_layer_widths (list(int)): A list of integers which defines the width of each hidden layer
            lr (float): A learning rate / step size.
            shuffle: Whether to shuffle the training data each epoch. DO NOT SHUFFLE for evaluation / debug datasets.

        Example:
            DT  = DTClassifier()
        """
        self.counts = counts
        self.validation_size = validation_size


    def fit(self, X, y):
        """ Fit the data; Make the Desicion tree

        Args:
            X (array-like): A 2D numpy array with the training data, excluding targets
            y (array-like): A 2D numpy array with the training targets

        Returns:
            self: this allows this to be chained, e.g. model.fit(X,y).predict(X_test)

        """
        data_set = np.concatenate((X, y), axis=1)

        best_feature = 0
        entropies = []

        class_col = np.shape(data_set)[1] - 1
        for feature in range(np.shape(data_set)[1] - np.shape(y)[1]):

            feature_info = {}
            total_observations = np.shape(data_set)[0]
            for observation in range(total_observations):
                observation_val = data_set[observation][feature]
                class_val = data_set[observation][class_col]
                if observation_val in feature_info:
                    feature_info[observation_val]['total'] = feature_info[observation_val]['total'] + 1
                    if class_val in feature_info[observation_val]:
                        feature_info[observation_val][class_val] = feature_info[observation_val][class_val] + 1.0
                    else:
                        feature_info[observation_val][class_val] = 1.0
                else:
                    feature_info[observation_val] = {class_val: 1.0, 'total' : 1}

            entropy_sum = 0
            for category in feature_info:
                entropy_sum = entropy_sum + self._calc_entropy(total_observations, feature_info[category])

            entropies.append(entropy_sum)
        min_entropy = min(entropies)
        best_feature = entropies.index(min_entropy)
        print(best_feature)
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

    # Helper Functions
    def _calc_entropy(self, all_obs, category_info):
        category_entropy_sum = 0
        category_total_obs = category_info['total']
        del category_info['total']

        for type in category_info:
            type_obs = category_info[type]
            probability = type_obs / category_total_obs
            category_entropy_sum = category_entropy_sum + (category_total_obs / all_obs) * (-(probability)*(log2(probability)))

        return category_entropy_sum


    # def _calc_entropy(self, p):
    #     if p != 0:
    #         return -p * np.log2(p)
    #     else:
    #         return 0


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
    data_shape = np.shape(data)
    labels_shape = np.shape(labels)
    data_set_cols = data_shape[1] + labels_shape[1]
    data_set_rows = 0

    if data_shape[0] == labels_shape[0]:
        data_set_rows = data_shape[0]
    else:
        print(f"Rows not the same size.\nRows in data: {data_shape[0]}\nRows in labels: {labels_shape[0]}")
    data_set_size = (data_set_rows, data_set_cols)
    DTClass = DTClassifier(counts)
    DTClass.fit(data, labels)
    # mat2 = Arff(arff_path2)
    # data2 = mat2.data[:, 0:-1]
    # labels2 = mat2.data[:, -1]
    # pred = DTClass.predict(data2)
    # Acc = DTClass.score(data2, labels2)
    # np.savetxt("pred_lenses.csv", pred, delimiter=",")
    # print("Accuracy = [{:.2f}]".format(Acc))
