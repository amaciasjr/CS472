import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
import math



### NOTE: The only methods you are required to have are:
#   * predict
#   * fit
#   * score
#   * get_weights
#   They must take at least the parameters below, exactly as specified. The output of
#   get_weights must be in the same format as the example provided.

class MLPClassifier(BaseEstimator,ClassifierMixin):

    def __init__(self, hidden_layer_widths, lr=.1, momentum=0, shuffle=True, deterministic=np.inf):
        """ Initialize class with chosen hyperparameters.

        Args:
            hidden_layer_widths (list(int)): A list of integers which defines the width of each hidden layer
            lr (float): A learning rate / step size.
            shuffle: Whether to shuffle the training data each epoch. DO NOT SHUFFLE for evaluation / debug datasets.

        Example:
            mlp = MLPClassifier([3,3]),  <--- this will create a model with two hidden layers, both 3 nodes wide
        """
        self.hidden_layer_widths = hidden_layer_widths
        self.lr = lr
        self.momentum = momentum
        self.shuffle = shuffle
        self.weights = None
        self.deterministic = deterministic
        self.features = None
        self.targets = None
        self.accuracy = 0.01
        self.numFeatures = 0


    def fit(self, X, y, initial_weights=None):
        """ Fit the data; run the algorithm and adjust the weights to find a good solution

        Args:
            X (array-like): A 2D numpy array with the training data, excluding targets
            y (array-like): A 2D numpy array with the training targets
            initial_weights (array-like): allows the user to provide initial weights

        Returns:
            self: this allows this to be chained, e.g. model.fit(X,y).predict(X_test)

        """
        numFeatures = X.shape[1]
        self.numFeatures = numFeatures

        self.initial_weights = self.initialize_weights() if not initial_weights else initial_weights
        self.weights = self.initial_weights
        self.targets = y
        self.features = X

        insignificant_improvement_counter = 0
        consecutive_epochs = 5
        bias = np.ones(1)
        epoch = 0
        max_accuracy = self.accuracy
        while epoch < self.deterministic:
            index = 0
            for row in self.features:
                pattern = np.append(row, bias)
                output = self.findOutput(pattern)

                new_weights = self.deltaWeights(self.targets[index], output, pattern)
                self.weights = np.add(self.weights, new_weights)
                index += 1

            if self.shuffle == True:
                self._shuffle_data(self.features, self.targets)

            # STOPPING CRITERIA
            current_accuracy = self.score(self.features, self.targets)
            if max_accuracy >= current_accuracy:
                insignificant_improvement_counter += 1

                if insignificant_improvement_counter == consecutive_epochs:
                    print(f"Stopped Training after {epoch} epochs.")
                    return self
            else:
                max_accuracy = current_accuracy
                insignificant_improvement_counter = 0

            epoch += 1

        return self

    def predict(self, X):
        """ Predict all classes for a dataset X

        Args:
            X (array-like): A 2D numpy array with the training data, excluding targets

        Returns:
            array, shape (n_samples,)
                Predicted target values per element in X.
        """
        predicted_targets = np.zeros(len(X))
        bias = np.ones(1)
        index = 0
        for row in X:
            pattern = np.append(row, bias)
            output = self.findOutput(pattern)

            predicted_targets[index] = output
            index += 1

        return predicted_targets

    def initialize_weights(self):
        """ Initialize weights for perceptron. Don't forget the bias!

        Returns:

        """
        bias = 1
        weights = np.zeros(self.numFeatures + bias)
        return weights

    def score(self, X, y):
        """ Return accuracy of model on a given dataset. Must implement own score function.

        Args:
            X (array-like): A 2D numpy array with data, excluding targets
            y (array-like): A 2D numpy array with targets

        Returns:
            score : float
                Mean accuracy of self.predict(X) wrt. y.
        """
        predictions = self.predict(X)

        index = 0
        correct_guesses = 0
        for target in y:

            if predictions[index] == target:
                correct_guesses += 1

            index += 1

        return correct_guesses / len(y)

    def _shuffle_data(self, X, y):
        """ Shuffle the data! This _ prefix suggests that this method should only be called internally.
            It might be easier to concatenate X & y and shuffle a single 2D array, rather than
             shuffling X and y exactly the same way, independently.
        """
        full_rows = np.hstack((X, y))
        np.random.shuffle(full_rows)

        columns = full_rows.shape[1]
        last_col = columns - 1
        first_col = 0
        new_targets = None
        new_features = None

        for column in range(0, columns):

            if column == first_col:
                new_features = np.hsplit(full_rows, columns)[first_col]
            elif column == last_col:
                new_targets = np.hsplit(full_rows, columns)[last_col]
            else:
                new_features = np.hstack((new_features, np.hsplit(full_rows, columns)[column]))

        self.features = new_features
        self.targets = new_targets
        pass

    ### Not required by sk-learn but required by us for grading. Returns the weights.
    def get_weights(self):
        return self.weights

    ### Helper functions
    def calculate_loss(self, actual_output, predicted_output):
        return (actual_output - predicted_output) ** 2

    def calculate_node_activation(self, net_value):
        return 1/(1 + math.exp( -net_value ))

    def calculate_activation_derivative(self, net_value):
        return self.calculate_activation_derivative(net_value) * (1 - self.calculate_activation_derivative(net_value))

    def deltaWeights(self, target, output, pattern):

        new_weights = np.zeros(len(pattern))

        ele_index = 0
        for element in pattern:
            curr_weight_val = self.lr * (target - output) * element
            new_weights[ele_index] = curr_weight_val
            ele_index += 1

        return new_weights

    def findOutput(self, pattern):

        temp = np.multiply(pattern, self.weights)
        net = np.sum(temp)
        output = 0

        if net > 0:
            output = 1

        return output

    def splitData(self, X, y):

        self._shuffle_data(X, y)

        seventyPercent = math.floor(self.features.shape[0] * .7)

        training_set = self.features[0:seventyPercent]
        training_labels = self.targets[0:seventyPercent]
        test_set = self.features[seventyPercent:]
        test_labels = self.targets[seventyPercent:]

        return training_set, test_set, training_labels, test_labels


mlp = MLPClassifier([3,3]) # This will create a model with two hidden layers, both 3 nodes wide
