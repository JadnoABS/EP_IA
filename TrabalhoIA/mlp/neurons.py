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
        self.weights = np.array([random() - 0.5 for i in range(next_layer_size
                                                              + 1 )])

    def receive(self, signals: np.core.multiarray) -> float:
        self.signals = signals
        result = np.matrix.dot(signals, self.weights[:-1].T)
        # result = np.matrix.dot(signals, self.weights)
        result += self.weights[-1]
        return result

    def sigmoid(self, x):
        # if x >= 10:
            # return 1
        # elif x <= -10:
            # return 0
        return np.exp(x) / (np.exp(x) + 1)
        # return 1/(1 + np.exp(-x))

    def activate(self, signals: list) -> float:
        self.after_activation = self.receive(signals)
        output = self.sigmoid(self.after_activation)
        self.output = output

        return output

    def derivate_activation(self):
        derivative = self.sigmoid(self.after_activation) * ( 1 - self.sigmoid(self.after_activation) )
        return derivative

    def calculate_delta(self, rate, err):
        pass

    def update_weight(self, rate, signals):
        pass

    def __repr__(self) -> str:
        return f"{{ weights:{self.weights} output:{self.output} error" \
               f":{self.error} }}"


class HiddenNeuron(BaseNeuron):
    def calculate_deltas(self, rate, next_layer_error, weights):
        self.deltas = []
        in_err = 0
        for index in range(len(next_layer_error)):
            # print(next_layer_error[index])
            # print(weights[index])
            print(in_err, next_layer_error[index], weights[index])
            in_err += next_layer_error[index] * weights[index]

        # print(in_err)
        self.error_info = in_err * self.derivate_activation()
        # self.error_info = in_err
        # print(self.error_info)

        for index, signal in enumerate(self.signals):
            self.deltas.append(rate * self.error_info * signal)
        self.deltas.append(rate * self.error_info)

    def update_weight(self, rate):
        for index, w in enumerate(self.weights):
            self.weights[index] -= self.deltas[index]


class OutNeuron(BaseNeuron):
    def calculate_deltas(self, rate, error):
        self.deltas = []
        self.error_info = error * self.derivate_activation()
        for index, signal in enumerate(self.signals):
            self.deltas.append(rate * self.error_info * signal)
        self.deltas.append(rate * self.error_info)

    def update_weight(self, rate):
        for index, w in enumerate(self.weights):
            self.weights[index] += self.deltas[index]
