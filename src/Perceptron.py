import os
import sys

import pandas as pd
import numpy as np
from random import random

class InNeuron:
    def receive(self, value):
        self.value = value

class OutNeuron:
    input = 0
    output = 0

    def receive(self, values, weights):
        r = 0
        for i in range(len(values)):
            r += values[i] * weights[i]

        r += weights[-1] # BIAS

        self.input = r

    def activate(self):
        if(self.input > 0):
            self.output = 1
        else:
            self.output = -1
        return self.output

class Perceptron:
    def __init__(self, n_in, n_out, random_weights):
        self.input_layer = [InNeuron() for i in range(n_in)]

        self.output_layer = [OutNeuron() for i in range(n_out)]

        if random_weights and os.path.isfile('./random_weights.csv'):
            self.weights = pd.read_csv('random_weights.csv', sep=',', header=None).values
        elif os.path.isfile('./trained_weights.csv'):
            self.weights = pd.read_csv('trained_weights.csv', sep=',', header=None).values
        else:
            self.weights = self.generate_random_weights(n_in, n_out)
            print(self.weights)
            np.savetxt('random_weights.csv', self.weights, delimiter=',')

    def generate_random_weights(self, n_in, n_out):
        return np.random.uniform(low=0.1, size=(n_out, n_in + 2))

    def train(self, n_cases, in_data, expected):
        cur_data = []
        cur_exp = []
        alpha = 0.5
        training = True

        epoch = 0
        while training:

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
                            if x < len(self.input_layer):
                                self.weights[j][x] += alpha * expected[i][j] * self.input_layer[x].value
                            else:
                                self.weights[j][x] += alpha * expected[i][j] # Atualizacao do peso do BIAS
                        epoch = 0

            epoch += 1
            if epoch >= 20:
                training = False

        np.savetxt('trained_weights.csv', self.weights, delimiter=',')

    def run(self, in_data):
        for i, in_neuron in enumerate(self.input_layer):
            in_neuron.receive(in_data[i])

        response = []

        for i, out_neuron in enumerate(self.output_layer):
            values = [ inp.value for inp in self.input_layer ]
            out_neuron.receive(values, self.weights[i])
            response.append(out_neuron.activate())

        return response
###########################################################################33

##############################
# TESTES #
##############################

def test(perceptron):
    n_cases = 7

    test_data = pd.read_csv('caracteres-ruido.csv', sep=',', header=None).values

    in_data = [ elem[0:63] for elem in test_data ]
    expected = [ elem[-7:] for elem in test_data ]

    for i in range(n_cases):
        cur_data = in_data[i]
        cur_exp = expected[i]
        result = perceptron.run(cur_data)

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


#########################################
# Execucao inicial
#########################################

if len(sys.argv) > 1:
    if sys.argv[1] == 'train':
        perceptron = Perceptron(63, 7, True)
        data = pd.read_csv('caracteres-limpo.csv', sep=',', header=None)
        data = data.values
        in_data = [ elem[0:63] for elem in data ]
        expected = [ elem[-7:] for elem in data ]
        perceptron.train(len(data), in_data, expected)

    elif sys.argv[1] == 'test':
        perceptron = Perceptron(63, 7, False)
        test(perceptron)
    else:
        print('Invalid Argument')
else:
    print('You need to define a argument (train or test)')











