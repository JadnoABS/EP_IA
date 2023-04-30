import pandas as pd
import numpy as np
import os
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
    def __init__(self, n_in, n_out):
        self.input_layer = [InNeuron() for i in range(n_in)]

        self.output_layer = [OutNeuron() for i in range(n_out)]
        self.bias = [0.2 for i in range(n_out)]

        self.weights = [ [ random() for i in range(n_in) ] for j in range(n_out) ]
        self.bias = [ 0.5 for i in range(n_out) ]

    def train(self, n_cases, in_data, expected):
        cur_data = []
        cur_exp = []
        alpha = 0.5
        training = True

        while training:
            epoch = 0

            # print("Pesos atuais:")
            # print(self.weights)

            for i in range(n_cases):
                cur_data = in_data[i]
                cur_exp = expected[i]
                result = self.run(cur_data)

                os.system('clear')

                print("Entrada:")
                for x in range(9):
                    for z in range(7):
                        if cur_data[7*x + z] == 1:
                            print("#", end="")
                        else:
                            print(".", end="")
                    print('\n')

                print("Resposta esperada: ")
                print(cur_exp)

                print("Resultado: ")
                print(result)

                print("EPOCA: ", epoch)

                for j in range(len(expected[i])):
                    if result[j] != expected[i][j]:
                        for x in range(len(self.weights[j])):
                            self.weights[j][x] += alpha * expected[i][j] * self.input_layer[x].value
                        epoch = 0

            epoch += 1
            if epoch >= 20:
                training = False

    def run(self, in_data):
        for i, in_neuron in enumerate(self.input_layer):
            in_neuron.receive(in_data[i])

        response = []

        for i, out_neuron in enumerate(self.output_layer):
            values = [ inp.value for inp in self.input_layer ]
            out_neuron.receive(values, self.weights[i], self.bias[i])
            response.append(out_neuron.activate())

        return response
###########################################################################33



perceptron = Perceptron(63, 7)

data = pd.read_csv('caracteres-limpo.csv', sep=',', header=None)
data = data.values

print(data)

in_data = [ elem[0:63] for elem in data ]
expected = [ elem[-7:] for elem in data ]


perceptron.train(len(data), in_data, expected)

##############################
# TESTES #
##############################

def test():
    n_cases = 7

    test_data = pd.read_csv('caracteres-ruido.csv', sep=',', header=None)

    in_data = [ elem[0:63] for elem in test_data ]
    expected = [ elem[-7:] for elem in test_data ]

    for i in range(n_cases):
    cur_data = in_data[i]
    cur_exp = expected[i]
    result = self.run(cur_data)

    print("Entrada:")
    for x in range(9):
        for z in range(7):
            if cur_data[7*x + z] == 1:
                print("#", end="")
            else:
                print(".", end="")
        print('\n')

    print("Resposta esperada: ")
    print(cur_exp)

    print("Resultado: ")
    print(result)

