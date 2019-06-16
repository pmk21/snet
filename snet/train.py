from snet.tensor import Tensor
from snet.nn import NeuralNet
from snet.loss import Loss, MSE
from snet.optim import Optimizer, SGD
from snet.data import DataIterator, BatchIterator


def train(
    net,
    inputs,
    targets,
    num_epochs,
    iterator=BatchIterator,
    loss=MSE(),
    optimizer=SGD()):

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for batch in  iterator(inputs, targets):
            predicted = net.forward(batch["inputs"])
            epoch_loss += loss.loss(predicted, batch["targets"])
            grad = loss.grad(predicted, batch["targets"])
            net.backward(grad)
            optimizer.step(net)
        print(epoch, epoch_loss)
