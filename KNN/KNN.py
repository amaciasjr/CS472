import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from arff import Arff

class KNNClassifier(BaseEstimator,ClassifierMixin):


    def __init__(self,labeltype=[],weight_type='inverse_distance'): ## add parameters here
        """
        Args:
            columntype for each column tells you if continues[real] or if nominal.
            weight_type: inverse_distance voting or if non distance weighting. Options = ["no_weight","inverse_distance"]
        """
        self.labeltype = labeltype
        self.weight_type = weight_type



    def fit(self,data,labels):
        """ Fit the data; run the algorithm (for this lab really just saves the data :D)
        Args:
            X (array-like): A 2D numpy array with the training data, excluding targets
            y (array-like): A 2D numpy array with the training targets
        Returns:
            self: this allows this to be chained, e.g. model.fit(X,y).predict(X_test)
        """
        return self
    def predict(self,data):
        """ Predict all classes for a dataset X
        Args:
            X (array-like): A 2D numpy array with the training data, excluding targets
        Returns:
            array, shape (n_samples,)
                Predicted target values per element in X.
        """
        pass

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

        return 0


mat = Arff("../data/knn/debug/seismic-bumps_train.arff",label_count=1)
mat2 = Arff("../data/knn/debug/seismic-bumps_test.arff",label_count=1)
raw_data = mat.data
h,w = raw_data.shape
train_data = raw_data[:,:-1]
train_labels = raw_data[:,-1]

raw_data2 = mat2.data
h2,w2 = raw_data2.shape
test_data = raw_data2[:,:-1]
test_labels = raw_data2[:,-1]

KNN = KNNClassifier(labeltype ='classification',weight_type='inverse_distance')
KNN.fit(train_data,train_labels)
pred = KNN.predict(test_data)
score = KNN.score(test_data,test_labels)
np.savetxt("seismic-bump-prediction.csv",pred,delimiter=',',fmt="%i")
