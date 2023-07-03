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
        self.weights = np.array([random() for i in range(next_layer_size
                                                              + 1 )])

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
            return -1
        return np.exp(x) / (np.exp(x) + 1)
        # return 1/(1 + np.exp(-x))

    def activate(self, signals: list) -> float:
        self.after_activation = self.receive(signals)
        output = self.sigmoid(self.after_activation)
        self.output = output

        return output

    def derivate_activation(self):
        derivative = self.output * ( 1 - self.output )
        # return derivative
        return (derivative - 0.5) * 2

    def calculate_delta(self, rate, err):
        pass

    def update_weight(self, rate, signals):
        pass

    def __repr__(self) -> str:
        return f"{{ weights:{self.weights} output:{self.output} error" \
               f":{self.error} }}"


class HiddenNeuron(BaseNeuron):
    def calculate_deltas(self, rate, next_layer_error, weights, total_weights):
        self.deltas = []
        self.error_info = 0
        for index in range(len(next_layer_error)):
            # print(next_layer_error[index])
            # print(weights[index])
            # print(in_err, next_layer_error[index], weights[index])
            self.error_info += next_layer_error[index] * (weights[index] / total_weights[index])

        # print(in_err)
        # self.error_info = in_err
        # print(self.error_info)
        in_err = self.error_info * self.derivate_activation()

        for index, signal in enumerate(self.signals):
            self.deltas.append(rate * in_err * signal)
        self.deltas.append(rate * in_err)

    def update_weight(self):
        for index, w in enumerate(self.weights):
            self.weights[index] += self.deltas[index]


class OutNeuron(BaseNeuron):
    def calculate_deltas(self, rate, error):
        self.deltas = []
        self.error_info = error
        in_err = self.error_info * self.derivate_activation()
        for index, signal in enumerate(self.signals):
            self.deltas.append(rate * in_err * signal)
        self.deltas.append(rate * in_err)

    def update_weight(self):
        for index, w in enumerate(self.weights):
            self.weights[index] += self.deltas[index]
