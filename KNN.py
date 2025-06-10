import numpy as np
from collections import Counter

def euclidean_dist(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))

class KNN():
    
    def __init__(self, k = 3):
        self.k = k 
    
    def fit (self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train 
    
    def predict (self, X):
        
        predictions = [self._predict(x) for x in X] 
        
        return predictions 
    
    
    def _predict(self, x):
        
        all_nearest_dist = [euclidean_dist(x,X) for X in self.X_train]
        
        k_nearest_indices = np.argsort(all_nearest_dist)[:self.k] 
        
        k_nearest_labels = [ self.y_train[i] for i in k_nearest_indices]
        
        most_common_val = Counter(k_nearest_labels).most_common(1)
        
        return most_common_val[0][0] 
    
                 
    