import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split
from arff import Arff
from tree import Tree
from tree import Node
from scipy import stats

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
        self.tree = None


    def fit(self, X, y):
        """ Fit the data; Make the Desicion tree

        Args:
            X (array-like): A 2D numpy array with the training data, excluding targets
            y (array-like): A 2D numpy array with the training targets

        Returns:
            self: this allows this to be chained, e.g. model.fit(X,y).predict(X_test)

        """
        data_set = np.concatenate((X, y), axis=1)

        # Now that the 'Root node' info is known, start making tree.
        root_node = Node(data_set, self.counts)
        root_node.set_best_feature(y)
        root_node.create_children_data_sets()
        decision_tree = Tree(root_node)
        decision_tree.build_tree(root_node)
        self.tree = decision_tree
        return self

    def predict(self, X):
        """ Predict all classes for a dataset X

        Args:
            X (array-like): A 2D numpy array with the training data, excluding targets

        Returns:
            array, shape (n_samples,)
                Predicted target values per element in X.
        """

        predictions = []
        root_node = self.tree.root
        modes = stats.mode(X)[0][0]
        for row in X:
            prediction = root_node.check_children_outputs(row, modes)
            predictions += [prediction]

        return predictions


    def score(self, X, y):
        """ Return accuracy of model on a given dataset. Must implement own score function.

        Args:
            X (array-like): A 2D numpy array with data, excluding targets
            y (array-li    def _shuffle_data(self, X, y):
        """

        predictions = self.predict(X)
        correct_values = 0
        total_values = len(y)
        if len(predictions) == total_values:
            for index in range(total_values):
                if predictions[index] == y[index]:
                    correct_values += 1

        accuracy = correct_values/total_values

        return accuracy



if __name__ == '__main__':

    # Debug Arff Paths
    # arff_path = r"../data/decisiontree/debug/lenses.arff"
    # arff_path2 = r"../data/decisiontree/debug/all_lenses.arff"

    # Evaluation Arff Paths
    # arff_path = r"../data/decisiontree/evaluation/zoo.arff"
    # arff_path2 = r"../data/decisiontree/evaluation/all_zoo.arff"

    # Evaluation Part 2 Arff Paths:
    arff_path = r"../data/decisiontree/eval-part2/cars.arff"
    arff_path2 = r"../data/decisiontree/eval-part2/voting.arff"

    mat = Arff(arff_path)

    counts = []  ## this is so you know how many types for each column

    for i in range(mat.data.shape[1]):
        counts += [mat.unique_value_count(i)]
    data = mat.data[:, 0:-1]
    labels = mat.data[:, -1].reshape(-1, 1)
    X_train, X_test, y_train, y_test = train_test_split(
        data, labels, test_size=0.1)
    DTClass = DTClassifier(counts)
    DTClass.fit(X_train, y_train)
    # pred = DTClass.predict(data)
    # Acc = DTClass.score(data, labels)
    # print("Accuracy = [{:.2f}]".format(Acc))
    scores = cross_validate(DTClass, X_test, y_test, cv=10)
    print(f"Train Scores = {scores['train_score']}\nTest Scores = {scores['test_score']}")


    # mat2 = Arff(arff_path2)
    # data2 = mat2.data[:, 0:-1]
    # labels2 = mat2.data[:, -1].reshape(-1, 1)
    # scores = cross_val_score(DTClass, data2, labels2, cv=10)
    # pred = DTClass.predict(data2)
    # Acc = DTClass.score(data2, labels2)
    # np.savetxt("pred_cars.csv", pred, delimiter=",")
    # print("Accuracy = [{:.2f}]".format(Acc))


