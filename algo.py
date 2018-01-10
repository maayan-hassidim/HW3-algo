from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.svm import SVC
from copy import deepcopy
import numpy as np

class OrdinalClassifier():
    """Ordinal Classifier"""

    def __init__(self, metaClassifier, ordinalList):
        """
        Called when initializing the classifier
        """
        self.metaClassifier = metaClassifier
        self.ordinalList = ordinalList
        self.numOfClasses = len(ordinalList)
        self.numOfClassifiers = self.numOfClasses - 1

    def fit(self, X, y=None):
        """
        This should fit classifier. All the "work" should be done here.

        Fit N-1 binary classifiers of the type of the Base Classifier where N is the number of values
        in the Target attribute
        """
        indexed_y = []
        for y_tag in y:
            indexed_y.append(self.ordinalList.index(y_tag))

        self.metaClassifiers = []

        for i in range(0,self.numOfClassifiers):
            self.metaClassifiers.append(deepcopy(self.metaClassifier))
            adj_y = []
            for j in range(0, len(indexed_y)):
                if indexed_y[j] > i:
                    adj_y.append(1)
                else:
                    adj_y.append(0)
            self.metaClassifiers[i].fit(X, adj_y)

        return self

    def predict(self, X, y=None):
        """
        Predict using N-1 classifiers
        1. Get the probability to get '1' value using Predict from each of the N-1 classifiers.
        2. For each instance in X
        2.1. Calculate probabilities vector to be in each Class.
        2.2. Return the class with the highest probability.
        """

        try:
            # STEP 1
            classifier_probabilities = []
            for i in range(0,len(self.metaClassifiers)):
                classifier_probabilities.append([])
                y = self.metaClassifiers[i].predict_proba(X)
                for test_case in y:
                    classifier_probabilities[i].append(test_case[1])

            # STEP 1.1 Arrange in to ease usage
            test_cases_probabilities = []
            for i in range(0,len(X)):
                proba_array = []
                for j in range(0,len(self.metaClassifiers)):
                    proba_array.append(classifier_probabilities[j][i])
                test_cases_probabilities.append(proba_array)
            # STEP 2
            result = []
            for test_case in test_cases_probabilities:

                # STEP 2.1
                probabilities = []
                for i in range(0, len(self.metaClassifiers)):
                    if i == 0:
                        probabilities.append(1-test_case[0])
                    elif i < len(self.metaClassifiers):
                        probabilities.append(test_case[i - 1] - test_case[i])
                probabilities.append(test_case[len(test_case) - 1])

                # STEP 2.2
                best_class = self.ordinalList[np.argmax(probabilities)]

                result.append(best_class)
            return result
        except AttributeError:
            raise RuntimeError("Error!")