import numpy as np

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import log_loss
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split


# Normalize inputs
#
# Assumes final column is the only output
def normalize_dataset_inputs(dataset):
    new_dataset = dataset.copy()

    inputs = new_dataset.iloc[:,:-1]
    new_dataset.iloc[:,:-1] = (inputs - inputs.mean()) / (2*inputs.std())

    return new_dataset


# Splits data into training, validation, and test sets
#
# test_size is from entire dataset, validation_size is from training set
def split_data(dataset, test_size=0.15):
    # Split data into training and testing
    training, test = train_test_split(dataset, test_size=test_size)

    return training, test


class DecisionTree(object):

    def __init__(self, alpha):
        self.dt = DecisionTreeClassifier(criterion='entropy', ccp_alpha=alpha)


    def train(self, dataset):
        self.dt = self.dt.fit(dataset.iloc[:, :-1], dataset.iloc[:, -1])
        return self


    def predict(self, inputs):
        return self.dt.predict(inputs)


    def score(self, dataset):
        outputs = self.predict(dataset.iloc[:, :-1])
        return (dataset.iloc[:, -1] == outputs).mean()


class NeuralNetwork(object):

    def __init__(self, hidden_size, hidden_depth):
        self.nn = MLPClassifier(hidden_layer_sizes=[hidden_size]*hidden_depth,
                                learning_rate_init=1e-2)


    def train(self, dataset):
        self.nn = self.nn.fit(dataset.iloc[:, :-1], dataset.iloc[:, -1])
        return self


    def predict(self, inputs):
        return self.nn.predict(inputs)


    def score(self, dataset):
        outputs = self.predict(dataset.iloc[:, :-1])
        return (dataset.iloc[:, -1] == outputs).mean()


class Boosting(object):

    def __init__(self, alpha):
        self.rf = RandomForestClassifier(ccp_alpha=alpha)


    def train(self, dataset):
        self.rf = self.rf.fit(dataset.iloc[:, :-1], dataset.iloc[:, -1])
        return self


    def predict(self, inputs):
        return self.rf.predict(inputs)


    def score(self, dataset):
        outputs = self.predict(dataset.iloc[:, :-1])
        return (dataset.iloc[:, -1] == outputs).mean()


class SupportVectorMachine(object):

    def __init__(self, kernel):
        self.svm = SVC(kernel=kernel)


    def train(self, dataset):
        self.svm = self.svm.fit(dataset.iloc[:, :-1], dataset.iloc[:, -1])
        return self


    def predict(self, inputs):
        return self.svm.predict(inputs)


    def score(self, dataset):
        outputs = self.predict(dataset.iloc[:, :-1])
        return (dataset.iloc[:, -1] == outputs).mean()


class NearestNeighbors(object):

    def __init__(self, k):
        self.knn = KNeighborsClassifier(n_neighbors=k)


    def train(self, dataset):
        self.knn = self.knn.fit(dataset.iloc[:, :-1], dataset.iloc[:, -1])
        return self


    def predict(self, inputs):
        return self.knn.predict(inputs)


    def score(self, dataset):
        outputs = self.predict(dataset.iloc[:, :-1])
        return (dataset.iloc[:, -1] == outputs).mean()
