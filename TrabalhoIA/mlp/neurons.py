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
    after_activation = 0

    def __init__(self, next_layer_size):
        self.weights = np.array([(random() - 0.5) * 2 for i in range(next_layer_size + 1 )])
        self.deltas = [ 0 for _ in range(next_layer_size + 1) ]

    def receive(self, signals: np.core.multiarray) -> float:
        self.signals = signals
        result = np.matrix.dot(self.weights[:-1], signals.T)
        # print(self.weights[:-1].T, signals)
        # print(result)
        # result = np.matrix.dot(signals, self.weights)
        result += self.weights[-1]
        return result

    def sigmoid(self, x):
        if x >= 10:
            return 1
        elif x <= -10:
            return 0
        return np.exp(x) / (np.exp(x) + 1)
        # return 1/(1 + np.exp(-x))

    def activate(self, signals: list) -> float:
        self.after_activation = self.receive(signals)
        self.true_output = self.sigmoid(self.after_activation)
        output = (self.true_output - 0.5) * 2
        # output = self.true_output
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
