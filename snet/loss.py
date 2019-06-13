"""
    A loss function measures how accurate the model's prediction is.
    We can use this to alter the parameters of the model.
"""

import numpy as np

from snet.tensor import Tensor

class Loss:
    def loss(self, predicted, actual):
        raise NotImplementedError
    
    def grad(self, predicted, actual):
        raise NotImplementedError

class MSE(Loss):
    """
        Mean squared error loss.
    """
    def loss(self, predicted, actual):
        return np.sum(np.power(predicted - actual, 2))/len(predicted)

    def grad(self, predicted, actual):
        return (2/len(predicted)) * (predicted - actual)