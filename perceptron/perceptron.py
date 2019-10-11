import numpy as np
import math
import matplotlib.pyplot as plt
import arff
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import Perceptron

### NOTE: The only methods you are required to have are:
#   * predict
#   * fit
#   * score
#   * get_weights
#   They must take at least the parameters below, exactly as specified. The output of
#   get_weights must be in the same format as the example provided.



class PerceptronClassifier(BaseEstimator,ClassifierMixin):

    def __init__(self, lr=.1, shuffle=True, deterministic=np.inf):
        """ Initialize class with chosen hyperparameters.

        Args:
            lr (float): A learning rate / step size.
            shuffle: Whether to shuffle the training data each epoch. DO NOT SHUFFLE for evaluation / debug datasets.
            weights: Store the final weights after the Preceptron Trains. Used in get_weights() method.
            deterministic: # of epochs used to train Perceptron.
        """
        self.lr = lr
        self.shuffle = shuffle
        self.weights = None
        self.deterministic = deterministic
        self.features = None
        self.targets = None
        self.accuracy = 0.01

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

        self.initial_weights = self.initialize_weights(numFeatures) if not initial_weights else initial_weights
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
                self._shuffle_data(self.features,self.targets)

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

    def initialize_weights(self, numFeatures):
        """ Initialize weights for perceptron. Don't forget the bias!

        Returns:

        """
        bias = 1
        weights = np.zeros(numFeatures+bias)
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

        return correct_guesses/len(y)

    def _shuffle_data(self, X, y):
        """ Shuffle the data! This _ prefix suggests that this method should only be called internally.
            It might be easier to concatenate X & y and shuffle a single 2D array, rather than
             shuffling X and y exactly the same way, independently.
        """
        full_rows = np.hstack((X,y))
        np.random.shuffle(full_rows)

        columns = full_rows.shape[1]
        last_col = columns - 1
        first_col = 0
        new_targets = None
        new_features = None

        for column in range(0,columns):

            if column == first_col:
                new_features = np.hsplit(full_rows,columns)[first_col]
            elif column == last_col:
                new_targets = np.hsplit(full_rows,columns)[last_col]
            else:
                new_features = np.hstack((new_features,np.hsplit(full_rows,columns)[column]))

        self.features = new_features
        self.targets = new_targets
        pass

    ### Not required by sk-learn but required by us for grading. Returns the weights.
    def get_weights(self):
        return self.weights

    ### Helper Functions
    def deltaWeights(self, target, output, pattern ):

        new_weights = np.zeros(len(pattern))

        ele_index = 0
        for element in pattern:
            curr_weight_val = self.lr * ( target - output ) * element
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

        return training_set,test_set,training_labels,test_labels


# IMPORT DATA from *.arff file(s).
mat = arff.Arff(arff="lab1Voting.arff", label_count=1)
data = mat.data[:,0:-1]
labels = mat.data[:,-1].reshape(-1,1)
PClass = PerceptronClassifier(lr=1,shuffle=False)
trainingSet, testSet, trainingLabels, testLabels = PClass.splitData(data, labels)
PClass.fit(trainingSet,trainingLabels)
training_accuracy = PClass.score(trainingSet,trainingLabels)
test_accuracy = PClass.score(testSet,testLabels)
print("My Perceptron's Results")
print("Training Accuray = [{:.2f}]".format(training_accuracy))
print("Test Accuray = [{:.2f}]".format(test_accuracy))
print("Final Weights =",PClass.get_weights())

from sklearn.datasets import load_digits
from sklearn.linear_model import Perceptron
X, y = load_digits(return_X_y=True)
clf = Perceptron(tol=0.001, random_state=3, validation_fraction=0.9, early_stopping=True, warm_start=True)
clf.fit(trainingSet, trainingLabels.ravel())
print(clf)
print(clf.score(testSet, testLabels.ravel()) )
# PClass = PerceptronClassifier(lr=1,shuffle=False)
# trainingSet, testSet, trainingLabels, testLabels = PClass.splitData(X, y)
# PClass.fit(trainingSet,trainingLabels)
# training_accuracy = PClass.score(trainingSet,trainingLabels)
# test_accuracy = PClass.score(testSet,testLabels)


## Graph using matplotlib
# x = np.linspace(-1, 1, 100)
# f1 = np.hsplit(data,2)[0]
# f2 = np.hsplit(data,2)[1]
#
# index = 0
# for label in labels:
#     if label == 0:
#         plt.scatter(data[index][0], data[index][1], c="blue", label="Class 2" )
#     else:
#         plt.scatter(data[index][0], data[index][1], c="red", label="Class 1")
#     index += 1
#
# plt.xlabel('Feature 1')
# plt.ylabel('Feature 2')
# plt.title("Non-linearly Separable Dataset")
# plt.legend()
# plt.show()
