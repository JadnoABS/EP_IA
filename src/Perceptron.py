import pandas as pd
import numpy as np
from random import random

class InNeuron:
    def receive(self, value):
        self.value = value

class OutNeuron:
    input = 0
    output = 0

    def receive(self, values, weights, bias):
        r = 0
        for i in range(len(values)):
            r += values[i] * weights[i]

        r += bias

        self.input = r

    def activate(self):
        if(self.input > 0):
            self.output = 1
        else:
            self.output = -1
        return self.output


class Perceptron:
    def _init_(self, n_in, n_out):
        self.input_layer = [InNeuron() * n_in]

        self.output_layer = [OutNeuron() * n_out]
        self.bias = [0.2 for i in range(n_out)]

        self.weights = [ [ random() for i in range(n_in) ] for j in range(n_out) ]
        self.bias = [ 0.5 for i in range(n_out) ]

    def read_input(self, values):
        for value in values:
            self.input_layer.append(value)

    def train(self, n_cases, in_data, expected):
        cur_data = []
        cur_exp = []
        alpha = 0.5
        training = True

        while training:
            epoch = 0

            for i in range(n_cases):
                cur_data = in_data[i]
                cur_exp = expected[i]
                result = self.run(cur_data)

                for j in range(len(expected)):
                    if result[j] != expected[j]:
                        for x in range(len(self.weights[j])):
                            self.weights[j][x] += alpha * expected[j] * self.in_layer[x].value
                        epoch = 0

            epoch += 1
            if epoch >= 20:
                training = False

    def run(self, in_data):
        for i, in_neuron in enumerate(self.in_layer):
            in_neuron.receive(in_data[i])

        response = []

        for i, out_neuron in enumerate(self.out_layer):
            values = [ inp.value for inp in self.in_layer ]
            out_neuron.receive(values, self.weights[i], self.bias[i])
            response.append(out_neuron.activate())

        return response
###########################################################################33



perceptron = Perceptron(63, 7)

data = pd.read_csv('caracteres-limpo.csv', sep=',')






