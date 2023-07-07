from random import random
import numpy as np
from mlp.Debugger import Debugger


class BaseNeuron:
    # o ultimo peso faz referencia ao bias
    weights: np.core.multiarray

    output = 0
    error = 0
    deltas = []
    signals = []
    before_activation = 0

    def __init__(self, weights=[], next_layer_size=0):
        if next_layer_size != 0:
            self.weights = np.random.uniform(-1, 1, size=next_layer_size + 1)
            self.deltas = [ 0 for _ in range(next_layer_size + 1) ]
        else:
            self.deltas = [0 for _ in range(len(weights))]
            self.weights = weights


    def receive(self, signals: np.core.multiarray) -> float:
        self.signals = signals
        result = np.matrix.dot(self.weights[:-1], signals.T)
        result += self.weights[-1]
        return result

    def sigmoid(self, x):
        if x >= 10:
            return 1
        elif x <= -10:
            return 0
        return np.exp(x) / (np.exp(x) + 1)

    def activate(self, signals: list) -> float:
        self.before_activation = self.receive(signals)
        self.true_output = self.sigmoid(self.before_activation)
        output = (self.true_output - 0.5) * 2
        self.output = output

        return output

    def derivate_activation(self):
        derivative = self.true_output * ( 1 - self.true_output )
        return derivative

    def calculate_delta(self, rate, err):
        pass

    def update_weight(self, rate, signals):
        pass

    def __repr__(self) -> str:
        return f"{{ weights:{self.weights} output:{self.output} error" \
               f":{self.error} }}"


class HiddenNeuron(BaseNeuron):
    def calculate_deltas(self, rate, momentum, next_layer_error, weights, inputs):
        self.error = 0
        for index in range(len(next_layer_error)):
            self.error += next_layer_error[index] * weights[index]

        self.error_info = self.error * self.derivate_activation()

        for index, signal in enumerate(inputs):
            self.deltas[index] = (rate * self.error_info * signal) - (momentum * self.deltas[index])
        self.deltas[-1] = (rate * self.error_info) - (momentum * self.deltas[-1])

    def update_weight(self):
        for index, w in enumerate(self.signals):
            self.weights[index] += self.deltas[index]


class OutNeuron(BaseNeuron):
    def calculate_deltas(self, rate, momentum, error, inputs):
        self.error = error
        self.error_info = self.error * self.derivate_activation()
        for index, signal in enumerate(inputs):
            self.deltas[index] = (rate * self.error_info * signal) - (momentum * self.deltas[index])
        self.deltas[-1] = (rate * self.error_info) - (momentum * self.deltas[-1])

    def update_weight(self):
        for index, w in enumerate(self.signals):
            self.weights[index] += self.deltas[index]
