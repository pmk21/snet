"""
    We'll feed inputs into our network in batches.
    So here are some tools for iterating over data in batches.
"""

import numpy as np

from snet.tensor import Tensor


class DataIterator:
    def __call__(self, inputs, targets):
        raise NotImplementedError
    
class BatchIterator(DataIterator):
    def __init__(self, batch_size=16, shuffle=True):
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __call__(self, inputs, targets):
        starts = np.arange(0, len(inputs), self.batch_size)
        if self.shuffle:
            np.random.shuffle(starts)
        
        for start in starts:
            end = start + self.batch_size
            batch_inputs = inputs[start:end]
            batch_targets = targets[start:end]
            yield [("inputs", batch_inputs), ("targets", batch_targets)]
