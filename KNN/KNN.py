import numpy as np
from math import sqrt
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, ClassifierMixin
from arff import Arff

class KNNClassifier(BaseEstimator,ClassifierMixin):


    def __init__(self, label_type='classification', weight_type='inverse_distance', k_neighbors = 3): ## add parameters here
        """
        Args:
            columntype for each column tells you if continues[real] or if nominal.
            weight_type: inverse_distance voting or if non distance weighting. Options = ["no_weight","inverse_distance"]
        """
        self.label_type = label_type
        self.weight_type = weight_type
        self.neighbors = []
        self.k_neighbors = k_neighbors
        self.observations = None
        self.labels = None


    def fit(self,data,labels):
        """ Fit the data; run the algorithm (for this lab really just saves the data :D)
        Args:
            X (array-like): A 2D numpy array with the training data, excluding targets
            y (array-like): A 2D numpy array with the training targets
        Returns:
            self: this allows this to be chained, e.g. model.fit(X,y).predict(X_test)
        """

        self.observations = data
        self.labels = labels

        return self


    def oldPredict(self, data):
        """ Predict all classes for a dataset X
        Args:
            X (array-like): A 2D numpy array with the training data, excluding targets
        Returns:
            array, shape (n_samples,)
                Predicted target values per element in X.
        """

        predictions = []

        if len(self.observations) < self.k_neighbors:
            print(f"Data length ({len(data)}) was too small.")

        for row in data:
            neighbors_info = {}

            for row_index in range(len(self.observations)):
                distance = self.calcualteEuclideanDistance(self.observations[row_index], row)
                if len(neighbors_info)  > self.k_neighbors - 1:
                    largest_distance = max(neighbors_info.keys())
                    if distance < largest_distance:
                        neighbors_info[distance] = self.labels[row_index]
                        del neighbors_info[largest_distance]
                else:
                    neighbors_info[distance] = self.labels[row_index]

            unique_values = set(neighbors_info.values())
            if len(unique_values) == 1:
                value = unique_values.pop()
                predictions.append(value)
            else:
                best_value = 0
                best_value_weight = 0
                for label in unique_values:
                    weight = 0
                    for distance in neighbors_info.keys():
                        if label == neighbors_info[distance]:
                            if 'inverse_distance' == self.weight_type:
                                weight += self.calulateWeightedVote(distance)
                            elif 'no_weight' == self.weight_type:
                                weight += 1
                            else:
                                print("Not a valid_weight_type.")

                    if weight > best_value_weight:
                        best_value_weight = weight
                        best_value = label

                predictions.append(best_value)
            # print(f"Neighbors Info: {neighbors_info}")

        return predictions


    def predict(self, data):

        predictions = []

        for row in data:
            temp_labels = self.labels
            neighbors_info = {}
            distance = (self.observations - row) ** 2
            sums = np.sum(distance, axis=1)
            sqrt_sums = np.sqrt(sums)
            for neighbor in range(self.k_neighbors):
                desired_ind = np.argmin(sqrt_sums)
                desired_distance = sqrt_sums[desired_ind]
                desired_label = temp_labels[desired_ind]
                neighbors_info[desired_distance] = desired_label
                temp_labels = np.delete(temp_labels,desired_ind)
                sqrt_sums = np.delete(sqrt_sums,desired_ind)

            if 'classification' == self.label_type:
                unique_values = set(neighbors_info.values())
                if len(unique_values) == 1:
                    value = unique_values.pop()
                    predictions.append(value)
                else:
                    best_value = 0
                    best_value_weight = 0
                    for label in unique_values:
                        weight = 0
                        for distance in neighbors_info.keys():
                            if label == neighbors_info[distance]:
                                if 'inverse_distance' == self.weight_type:
                                    weight += self.calulateWeightedVote(distance)
                                elif 'no_weight' == self.weight_type:
                                    weight += 1
                                else:
                                    print("Not a valid_weight_type.")

                        if weight > best_value_weight:
                            best_value_weight = weight
                            best_value = label

                    predictions.append(best_value)
            elif 'regression' == self.label_type:
                print('In regression!')
                labels = np.asarray(list(neighbors_info.values()))
                regression_output_value = np.mean(labels)
                predictions.append(regression_output_value)

        return np.asarray(predictions)


    #Returns the Mean score given input data and labels
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
        total_values = len(y)
        # if len(predictions) == total_values:
        #     for index in range(total_values):
        #         if predictions[index] == y[index]:
        #             correct_values += 1
        if 'classification' == self.label_type:
            correct_values = np.where(predictions == y)
            accuracy = correct_values[0].size / total_values
        elif 'regression' == self.label_type:
            sse = (y - predictions) ** 2
            sse_summed = np.sum(sse)
            accuracy = sse_summed / total_values

        return accuracy


    # Helper Functions
    def calcualteEuclideanDistance(self, row1, row2):
        assert len(row1) == len(row2)

        distance = 0
        for index in range(len(row2)):
            distance += (row1[index] - row2[index]) ** 2

        distance = sqrt(distance)
        return distance

    def calulateWeightedVote(self, distance):
        return 1/(distance ** 2)


def normalizeDataSets(train_data, test_data):
    assert np.shape(train_data) != (0,0) and np.shape(test_data) != (0,0)
    assert np.shape(train_data)[1] == np.shape(test_data)[1]


    for col_index in range(np.shape(train_data)[1]):
        col_max = np.amax(train_data[:,col_index])
        col_min = np.amin(train_data[:,col_index])
        train_data[:, col_index] = (train_data[:,col_index] - col_min) / (col_max - col_min)
        test_data[:, col_index] = (test_data[:, col_index] - col_min) / (col_max - col_min)

    return train_data, test_data

def part1():
    print('Running Part 1...')
    # Debug Data sets:
    # mat = Arff("../data/knn/debug/seismic-bumps_train.arff",label_count=1)
    # mat2 = Arff("../data/knn/debug/seismic-bumps_test.arff",label_count=1)

    # Evaluation Data sets:
    mat = Arff("../data/knn/evaluation/diabetes.arff",label_count=1)
    mat2 = Arff("../data/knn/evaluation/diabetes_test.arff",label_count=1)

    k_neighbors = 3
    raw_data = mat.data
    h, w = raw_data.shape
    train_data = raw_data[:, :-1]
    train_labels = raw_data[:, -1]

    raw_data2 = mat2.data
    h2, w2 = raw_data2.shape
    test_data = raw_data2[:, :-1]
    test_labels = raw_data2[:, -1]

    KNN = KNNClassifier(label_type='classification', weight_type='inverse_distance', k_neighbors=k_neighbors)
    print("Fitting data ...")
    KNN.fit(train_data, train_labels)
    print("Predict data ...")
    pred = KNN.predict(test_data)
    print("Scoring data ...")
    score = KNN.score(test_data, test_labels)
    np.savetxt("diabetes-prediction.csv",pred, delimiter=',',fmt="%i")
    print("Accuracy = [{:.2f}]\n".format(score * 100))


def part2():
    print('Running Part 2...')
    # Part 2 Data sets:
    mat = Arff("../data/knn/magic-telescope/mt_training.arff", label_count=1)
    mat2 = Arff("../data/knn/magic-telescope/mt_testing.arff", label_count=1)

    k_neighbors = 3
    raw_data = mat.data
    h, w = raw_data.shape
    train_data = raw_data[:, :-1]
    train_labels = raw_data[:, -1]

    raw_data2 = mat2.data
    h2, w2 = raw_data2.shape
    test_data = raw_data2[:, :-1]
    test_labels = raw_data2[:, -1]

    KNN = KNNClassifier(label_type='classification', weight_type='no_weight', k_neighbors=k_neighbors)
    print("Fitting data ...")
    KNN.fit(train_data, train_labels)
    print("Scoring data ...")
    score = KNN.score(test_data, test_labels)
    print("Accuracy = [{:.2f}]\n".format(score * 100))

    norm_train_data, norm_test_data = normalizeDataSets(train_data, test_data)

    print("Fitting normalized data ...")
    KNN.fit(norm_train_data, train_labels)
    print("Scoring normalized data ...")
    score = KNN.score(norm_test_data, test_labels)
    print("Accuracy = [{:.2f}]\n".format(score * 100))

    print('Running K Values tests...')
    k_values = [1, 3, 5, 7, 9, 11, 13, 15]
    scores = []
    for k_val in k_values:
        KNN = KNNClassifier(label_type='classification', weight_type='no_weight', k_neighbors=k_val)
        KNN.fit(norm_train_data, train_labels)
        score = KNN.score(norm_test_data, test_labels)
        scores.append(score * 100)

    print('Plotting K values vs Scores...')
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(k_values, scores, label='Accuracy')
    for xy in zip(k_values, scores):  # <--
        ax.annotate('(%s, %.1f)' % xy, xy=xy, textcoords='data')
    plt.title('Accuracy Using Different K Values')
    plt.xlabel('K Values')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.savefig('k-value-and-accuracy.png')
    plt.show()

def part3():
    print('Running Part 3...')
    # Part 3 Data sets:
    mat = Arff("../data/knn/housing-price/hp_training.arff", label_count=1)
    mat2 = Arff("../data/knn/housing-price/hp_testing.arff", label_count=1)

    k_neighbors = 3
    raw_data = mat.data
    h, w = raw_data.shape
    train_data = raw_data[:, :-1]
    train_labels = raw_data[:, -1]

    raw_data2 = mat2.data
    h2, w2 = raw_data2.shape
    test_data = raw_data2[:, :-1]
    test_labels = raw_data2[:, -1]

    # Normalize Data.
    train_data, test_data = normalizeDataSets(train_data, test_data)

    print('Running K Values tests...')
    k_values = [1, 3, 5, 7, 9, 11, 13, 15]
    mses = []
    for k_val in k_values:
        KNN = KNNClassifier(label_type='regression', weight_type='no_weight', k_neighbors=k_val)
        KNN.fit(train_data, train_labels)
        score = KNN.score(test_data, test_labels)
        mses.append(score)

    print('Plotting K values vs MSE Scores...')
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(k_values, mses, label='MSE')
    for xy in zip(k_values, mses):  # <--
        ax.annotate('(%s, %.2f)' % xy, xy=xy, textcoords='data')
    plt.title('MSE Using Different K Values')
    plt.xlabel('K Values')
    plt.ylabel('MSE')
    plt.legend()
    plt.savefig('k-value-and-mse.png')
    plt.show()

if __name__ == '__main__':

    # part1()
    # part2()
    part3()

