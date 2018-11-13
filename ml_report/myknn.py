import numpy as np
from heapq import nlargest
from sklearn.base import BaseEstimator, ClassifierMixin

def minkowski(a, b, p):
    return sum(abs(a[i] - b[i]) ** p for i in range(len(a))) ** (1 / p)

def manhattan(a, b):
    return minkowski(a, b, 1)

def euclidean(a, b):
    return minkowski(a, b, 2)

def hamming(a, b):
    return sum(int(a[i] != b[i]) for i in len(a))

class MyKNN(BaseEstimator, ClassifierMixin):
    def __init__(self, k=1, metric=euclidean):
        self.k = k
        self.metric = metric
        
    def fit(self, X, y):
        self.classes = sorted(list(set(y)))
        self.X = X
        self.y = y
        
    def predict(self, X):
        return list(map(self.__predict, X))
        
    def __predict(self, x):
        nn_i = [e[0] for e in sorted([(i, x) for i, x in enumerate(self.X)],
                                     key=lambda e: self.metric(e[1], x))]
        knn_i = nn_i[:self.k]
        
        while True:
            last_nn_dist = self.metric(self.X[knn_i[-1]], x)
            next_n_dist = self.metric(self.X[nn_i[len(knn_i)]], x)
            if last_nn_dist == next_n_dist: # distance tie
                knn_i.append(nn_i[len(knn_i)])
                continue
                
            class_counts = [[self.y[i] for i in knn_i].count(c) for c in self.classes]
            top2 = nlargest(2, class_counts)
            if top2[0] == top2[1]: # count tie
                knn_i.append(nn_i[len(knn_i)])
                continue

            return self.classes[np.argmax(class_counts)]