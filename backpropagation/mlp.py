import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
import math
from arff import Arff


### NOTE: The only methods you are required to have are:
#   * predict
#   * fit
#   * score
#   * get_weights
#   They must take at least the parameters below, exactly as specified. The output of
#   get_weights must be in the same format as the example provided.

class MLPClassifier(BaseEstimator,ClassifierMixin):

    def __init__(self, input_layer_nodes, hidden_layers, output_layer_nodes, lr=.1, momentum=0, shuffle=True, deterministic=np.inf):
        """ Initialize class with chosen hyperparameters.

        Args:
            hidden_layers (list(int)): A list of integers which defines the width of each hidden layer
            lr (float): A learning rate / step size.
            shuffle: Whether to shuffle the training data each epoch. DO NOT SHUFFLE for evaluation / debug datasets.

        Example:
            mlp = MLPClassifier([3,3]),  <--- this will create a model with two hidden layers, both 3 nodes wide
        """
        self.lr = lr
        self.momentum = momentum
        self.shuffle = shuffle
        self.weights = None
        # self.deltaWeights = None
        self.deterministic = deterministic
        self.features = None
        self.targets = None
        self.accuracy = 0.01
        self.numFeatures = 0

        # mlp additions
        self.input_layer_size = input_layer_nodes
        self.hidden_layers = hidden_layers
        self.output_layer_size = output_layer_nodes
        self.model = self.create_model()


    def fit(self, X, y, initial_weights=None):
        """ Fit the data; run the algorithm and adjust the weights to find a good solution

        Args:
            X (array-like): A 2D numpy array with the training data, excluding targets
            y (array-like): A 2D numpy array with the training targets
            initial_weights (array-like): allows the user to provide initial weights

        Returns:
            self: this allows this to be chained, e.g. model.fit(X,y).predict(X_test)

        """
        self.initial_weights = self.initialize_weights() if not initial_weights else initial_weights
        self.weights = self.initial_weights
        layers_in_model = len(self.model)

        # self.features = X
        # self.targets = y
        observations = len(X)
        for observation in range(observations):

            for layer in range(layers_in_model):

                bias_node = [1]

                modified_observation = np.append(X[observation], bias_node)
                self.model[layer] = np.add(modified_observation,self.model[layer])
                if layer > 0:
                    pattern = np.zeros(len(self.model[layer]))
                    if 0 == layer:
                        pattern = np.add(pattern,self.model[layer])
                    else:
                        previous_layer = layer - 1
                        pattern = np.add(pattern, self.model[previous_layer])

                    self.get_layer_outputs(pattern, layer)

        # pattern = np.append(outputs, bias_node)
        # outputs.append(self.get_layer_outputs(pattern, layer))

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
            output = self.get_layer_outputs(pattern, index)

            predicted_targets[index] = output
            index += 1

        return predicted_targets

    def initialize_weights(self):
        """ Initialize weights for perceptron. Don't forget the bias!

        Returns:

        """
        model_weights = []
        delta_weights = []
        BIAS_NODE = 1
        hidden_layer_nodes = self.hidden_layers[1] - BIAS_NODE

        weights_between_in_and_hid = (self.input_layer_size, hidden_layer_nodes)
        weights_between_hid_and_hid = (self.hidden_layers[1], hidden_layer_nodes)
        weights_between_hid_and_out = (self.hidden_layers[1], self.output_layer_size)

        # TODO: weights = [np.random.normal(size=weights_between_in_and_hid),
        # TODO:            np.random.normal(size=weights_between_hid_and_out)]

        model_weights.append(np.ones(weights_between_in_and_hid))
        delta_weights.append(np.empty(weights_between_in_and_hid))

        if self.hidden_layers[0] > 1:

            for layer_num in range(self.hidden_layers[1]):
                model_weights.append(np.ones(weights_between_hid_and_hid))
                delta_weights.append(np.empty(weights_between_hid_and_hid))
        else:
            print("There is only 1 hidden layer in the model!")

        model_weights.append(np.ones(weights_between_hid_and_out))
        delta_weights.append(np.empty(weights_between_hid_and_out))

        self.deltaWeights = delta_weights
        return model_weights

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

    def update_weights(self, target, output, pattern):

        new_weights = np.zeros(len(pattern))

        ele_index = 0
        for element in pattern:
            curr_weight_val = self.lr * (target - output) * element
            new_weights[ele_index] = curr_weight_val
            ele_index += 1

        return new_weights

    def get_layer_outputs(self, pattern, layer):

        node_in_layer = 0

        for row in np.transpose(self.weights[layer]):

            temp = np.multiply(pattern, row)
            net = np.sum(temp)
            output = self.calculate_node_activation(net)
            self.model[layer][node_in_layer] = output
            node_in_layer += 1

        pass

    def create_model(self):

        model = []
        BIAS_NODE = 1
        self.input_layer_size = self.input_layer_size + BIAS_NODE
        self.hidden_layers[1] = self.hidden_layers[1] + BIAS_NODE

        # Add the amount of input layer nodes to model
        model.append(np.zeros(self.input_layer_size))

        # Add the amount of hidden layers and their nodes to model
        for layer_num in range(self.hidden_layers[0]):

            model.append(np.zeros(self.hidden_layers[1]))

        # Add the amount of input layer nodes to model
        model.append(np.zeros(output_layer_nodes))

        return model



    # def splitData(self, X, y):
    #
    #     self._shuffle_data(X, y)
    #
    #     seventyPercent = math.floor(self.features.shape[0] * .7)
    #
    #     training_set = self.features[0:seventyPercent]
    #     training_labels = self.targets[0:seventyPercent]
    #     test_set = self.features[seventyPercent:]
    #     test_labels = self.targets[seventyPercent:]
    #
    #     return training_set, test_set, training_labels, test_labels


# Testing with Homework Example
# input_layer_nodes = 2
# hidden_layers = [1, 2]
# output_layer_nodes = 1


# features = [0,1]
# np_features = np.asarray(features)
# classes = [0]
# np_classes = np.asarray(classes)
#
# fit_result = mlp.fit(np_features, np_classes)

# Testing with Debug Data
mat = Arff("backprophw.arff")
data = mat.data[:,0:-1]
labels = mat.data[:,-1].reshape(-1,1)
input_layer_nodes = len(data[0])
hidden_layers = [1, input_layer_nodes]
output_layer_nodes = len(labels[0])
MLPClass = MLPClassifier(input_layer_nodes=input_layer_nodes,
                    hidden_layers=hidden_layers,
                    output_layer_nodes=output_layer_nodes)
# MLPClass = MLPClassifier(lr=0.1,momentum=0.5,shuffle=False,deterministic=10)
MLPClass.fit(data,labels)
