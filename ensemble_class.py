# simple class to ensemble the results of multiple classifiers

import numpy as np
import pandas as pd


class Ensemble:
    def __init__(self, models, weights = None, n_classes = 2) -> None:
        self.models = models
        self.weights = [1] * len(models) if weights is None else weights
        self.n_models = len(models)
        self.n_classes = n_classes
    
    def predict_with_feat_perm(self, X, y):
        preds = np.zeros((X.shape[0], self.n_classes))
        for i, model in enumerate(self.models):
            cols = np.arange(0,X.shape[1])
            np.random.shuffle(cols)
            X = X[:, cols]
            preds[:, i] = model.predict_proba(X)
        
        preds = np.dot(preds, self.weights)
        preds = np.sum(preds, axis=1)
        preds = preds / np.sum(self.weights)

        return preds


