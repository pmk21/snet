"""
    Neural nets will be made up of multiple layers.
    Each layer needs to pass it's inputs forward and
    propagate gradients backward.

    Example - 

    inputs -> Linear -> Activation -> Linear -> output
"""

import numpy as np

from snet.tensor import Tensor

class Layer:
    def init(self):
        self.params = {}
        self.grads = {}

    def forward(self, inputs):
        """
            Produce outputs corresponding to inputs
        """
        raise NotImplementedError
    
    def backward(self, grad):
        """
            Backpropagate this gradient through the layer
        """
        raise NotImplementedError
    

class Linear(Layer):
    """
        Computes output = inputs @ w + b
    """
    def __init__(self, input_size, output_size):
        super().__init__()
        self.params['w'] = np.random.randn(input_size, output_size)
        self.params['b'] = np.random.randn(output_size)

    def forward(self, inputs):
        return inputs @ self.params['w'] + self.params['b']
    
    def backward(self, grad):
        """
            if y = f(x) and x = a * b + c
            then dy/da = f'(x) * b
            and dy/db = f'(x) * a
            and dy/dc = f'(x)
            
            if y = f(x) and x = a @ b + c
            then dy/da = f'(x) @ b.T
            and dy/db = a.T @ f'(x)
            and dy/dc = f'(x)
        """
        self.grads["b"] = np.sum(grad, axis=0)
        self.grads["w"] = self.inputs.T @ grad
        return grad @ self.params["w"].T

class Activation(Layer):
    """
        An activation layer just applies a function
        elementwise to its inputs.
    """
    def __init__(self, f, f_comp):
        super().__init__()
        self.f = f
        self.f_comp = f_comp

    def forward(self, inputs):
        self.inputs = inputs
        return self.f(inputs)
    
    def backward(self):
        """
            if y = f(x) and x = g(z)
            then dy/dz = f'(x) * g'(z)
        """
        return self.f_comp(self.inputs) * grad
    
def tanh(x):
    return np.tanh(x)

def tanh_comp(x):
    y = tanh(x)
    return 1 - y ** 2

class Tanh(Activation):
    def __init__(self):
        super().__init__(tanh, tanh_comp)