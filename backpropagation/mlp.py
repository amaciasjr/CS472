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
        self.targets = y
        self.features = X

        layers_in_model = len(self.model)
        last_hidden_layer = layers_in_model - 1
        observations = len(X)
        epoch = 0

        while epoch < self.deterministic:

            for observation in range(observations):

                # Forward Propagation
                self.propagate_forward(layers_in_model, X, observation, last_hidden_layer)

                # GOING BACKWARDS: BACK-PROPAGATION OF ERROR
                target = y[observation]
                self.propagate_error_backwards(target)

                # Add delta weights to current weights. ("on-line/stochastic weight update")
                for layer in range(len(self.weights)):
                    self.weights[layer] = np.add(self.weights[layer], self.deltaWeights[layer])
                print(self.get_weights())

            # shuffle training set at each epoch
            if self.shuffle == True:
                self._shuffle_data(self.features, self.targets)


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
        layers_in_model = len(self.model)
        output_layer = layers_in_model - 1
        for observation_num in range(len(X)):
            self.propagate_forward(layers_in_model, X, observation_num, output_layer)
            output = self.model[output_layer]
            predicted_targets[observation_num] = output

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

        # random weight initialization (small random weights with 0 mean)
        model_weights.append(np.ones(weights_between_in_and_hid))
        # model_weights.append(np.random.normal(size=weights_between_in_and_hid))
        delta_weights.append(np.empty(weights_between_in_and_hid))

        if self.hidden_layers[0] > 1:

            for layer_num in range(self.hidden_layers[1]):
                model_weights.append(np.ones(weights_between_hid_and_hid))
                # model_weights.append(np.random.normal(size=weights_between_hid_and_hid))
                delta_weights.append(np.empty(weights_between_hid_and_hid))
        else:
            print("There is only 1 hidden layer in the model!")

        model_weights.append(np.ones(weights_between_hid_and_out))
        # model_weights.append(np.random.normal(size=weights_between_hid_and_out))
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

    def calculate_activation_derivative(self, value):
        return value * (1 - value)

    def update_weights(self, target, output, pattern):

        new_weights = np.zeros(len(pattern))

        ele_index = 0
        for element in pattern:
            curr_weight_val = self.lr * (target - output) * element
            new_weights[ele_index] = curr_weight_val
            ele_index += 1

        return new_weights

    def get_layer_outputs(self, pattern, previous_layer, is_output_layer = False):

        BIAS_NODE = 1
        current_layer = previous_layer + 1

        if not is_output_layer:
            nodes_in_layer = self.hidden_layers[1] - BIAS_NODE
        else:
            nodes_in_layer = self.output_layer_size

        for node_num in range(nodes_in_layer):
            layer_weights = np.transpose(self.weights[previous_layer])
            temp = np.multiply(pattern, layer_weights[node_num])
            net = np.sum(temp)
            output = self.calculate_node_activation(net)
            self.model[current_layer][node_num] = output

    def create_model(self):

        model = []
        BIAS_NODE = 1
        self.input_layer_size = self.input_layer_size + BIAS_NODE
        self.hidden_layers[1] = self.hidden_layers[1] + BIAS_NODE

        # Add the amount of input layer nodes to model
        model.append(np.zeros(self.input_layer_size))

        # Add the amount of hidden layers and their nodes to model
        # "ability to create a network structure with at least one hidden layer and an arbitrary number of nodes"
        for layer_num in range(self.hidden_layers[0]):
            model.append(np.zeros(self.hidden_layers[1]))

        # Add the amount of input layer nodes to model
        model.append(np.zeros(output_layer_nodes))

        return model

    def propagate_forward(self, layers_in_model, X, observation, output_layer):
        ONE_LAYER = 1
        INPUT_LAYER = 0

        for layer in range(layers_in_model):

            bias_node = [1]
            modified_observation = np.append(X[observation], bias_node)

            if layer != output_layer:
                self.model[layer] = modified_observation

            if layer > INPUT_LAYER:
                pattern = np.zeros(len(self.model[layer]))
                previous_layer_num = layer - ONE_LAYER
                previous_layer = self.model[previous_layer_num]
                pattern = np.add(pattern, previous_layer)
                if layer != output_layer:
                    self.get_layer_outputs(pattern, previous_layer_num)
                else:
                    is_output_layer = True
                    self.get_layer_outputs(pattern, previous_layer_num, is_output_layer)
                    self.calculate_output_activation(output_layer)

    def propagate_error_backwards(self, target):

        delta_weight_layer = len(self.deltaWeights) - 1
        is_output_layer = True
        last_layer_error = []
        for layer in reversed(self.model):
            errors = []
            if is_output_layer:

                for node_value in layer:
                    current_error = self.calculate_delta(target, node_value)
                    errors.append(current_error[0])

            elif delta_weight_layer > -1:
                layer_nodes_without_bias = len(layer) - 1
                for node_num in range(layer_nodes_without_bias):

                    errors_and_wights_sum = 0
                    for error_num in range(len(last_layer_error)):
                        weight_layer = delta_weight_layer + 1
                        weight = self.weights[weight_layer][error_num][0]
                        errors_and_wights_sum += last_layer_error[error_num] * weight
                    curr_val = layer[node_num]
                    new_error = curr_val * (1 - curr_val) * (errors_and_wights_sum)
                    errors.append(new_error)


            # for delta_weight_layer in range(delta_weight_layers, -1, -1):
            for delta_num in range(len(errors)):
                node_num = 0
                for node_value in self.model[delta_weight_layer]:
                    delta_weight_value = self.lr * errors[delta_num] * node_value
                    # an option to include a momentum term
                    if not is_output_layer:
                        last_delta_weight = self.deltaWeights[delta_weight_layer][node_num][delta_num]
                        self.deltaWeights[delta_weight_layer][node_num][delta_num] = delta_weight_value + self.momentum * last_delta_weight
                    else:
                        last_delta_weight = self.deltaWeights[delta_weight_layer][node_num]
                        self.deltaWeights[delta_weight_layer][node_num] = delta_weight_value + self.momentum * last_delta_weight

                    node_num += 1

            is_output_layer = False
            last_layer_error = errors
            delta_weight_layer -= 1

    def calculate_delta(self, target, node_value):

        return (target - node_value) * self.calculate_activation_derivative(node_value)

    def calculate_output_activation(self, output_layer):
        FIRST_NODE = 0
        ONE_NODE = 1
        ACTIVATE = 1
        DONT_ACTIVATE = 0
        output_layer_nodes = self.model[output_layer]

        # Assuming that Output layer nodes >= 1
        if len(output_layer_nodes) == ONE_NODE:
            if output_layer_nodes[FIRST_NODE] > DONT_ACTIVATE:
                output_layer_nodes[FIRST_NODE] = ACTIVATE
            else:
                output_layer_nodes[FIRST_NODE] = DONT_ACTIVATE
        else:
            largest_output = max(output_layer_nodes)
            for node_index in range(len(output_layer_nodes)):
                if output_layer_nodes[node_index] == largest_output:
                    output_layer_nodes[node_index] = ACTIVATE
                else:
                    output_layer_nodes[node_index] = DONT_ACTIVATE

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


# Testing with Debug Data
mat = Arff("backprophw.arff")
data = mat.data[:,0:-1]
labels = mat.data[:,-1].reshape(-1,1)
input_layer_nodes = len(data[0])
hidden_layers = [1, input_layer_nodes]
output_layer_nodes = len(labels[0])
MLPClass = MLPClassifier(input_layer_nodes=input_layer_nodes,
                    hidden_layers=hidden_layers,
                    output_layer_nodes=output_layer_nodes,
                    lr=1, deterministic=1)
# MLPClass = MLPClassifier(lr=0.1,momentum=0.5,shuffle=False,deterministic=10)
MLPClass.fit(data,labels).predict(data)
