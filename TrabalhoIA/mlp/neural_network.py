import numpy as np
from math import ceil

from mlp.neurons import HiddenNeuron, OutNeuron
from mlp.Debugger import Debugger


class NeuralNetwork:
    layers: list = []

    # rate = 0.2

    def __init__(self, n_layers, rate,
                 momentum, x_train=np.array([]), y_train=np.array([]),
                 x_test=np.array([]), y_test=np.array([]),
                 weights=[]):

        if len(weights) == 0:
            self.construct_layers(n_inputs=x_train.shape[1],
                                  n_outputs=y_train.shape[1],
                                  n_layers=n_layers)
        else:
            self.construct_layers(weights=weights)

        self.rate = rate
        self.momentum = momentum

    def construct_layers(self, n_inputs, n_outputs, n_layers, weights=[]):
        self.layers = []
        if len(weights) == 0:
            for index, layer in enumerate(weights):
                layer = []
                for neuron in weights:
                    if index == len(weights) - 1:
                        new_neuron = OutNeuron(weights=neuron)
                    else:
                        new_neuron = HiddenNeuron(weights=neuron)
                    layer.append(new_neuron)
                self.layers.append(layer)

        n_hidden = ceil((n_inputs / 3) * 2)
        self.layers.append(
            [HiddenNeuron(next_layer_size=n_inputs) for _ in range(n_hidden)])

        for i in range(n_layers - 2):
            self.layers.append([HiddenNeuron(next_layer_size=n_hidden) for _ in
                                range(n_hidden)])

        self.layers.append(
            [OutNeuron(next_layer_size=n_hidden) for _ in range(n_outputs)])

        # weights = []
        # for i, layer in enumerate(self.layers):
        # weights.append(self.get_weights_from_layer(i))
        # np.savetxt('initial_weights.csv', weights, delimiter=',')

    def propagate(self, inputs: list):
        # print(inputs)
        next_input = inputs
        for index, layer in enumerate(self.layers):
            input_arr = np.array(next_input)
            next_input = []
            for neuron in layer:
                output = neuron.activate(input_arr)
                next_input.append(output)
                # print(neuron.weights)
                # print(output)
                # Debugger.pause()

        # outputs = []
        # for n_index, neuron in enumerate(self.layers[-1]):
        # if next_input[n_index] > 0:
        # neuron.output = 1
        # outputs.append(1)
        # else:
        # neuron.output = -1
        # outputs.append(-1)

        # return outputs
        return next_input

    def calculate_initial_error(_, result, expected):
        err = 0
        for index, x in enumerate(expected):
            err += 0.5 * ((x - result[index]) ** 2)

        return err

    def calculate_errors(self, result, expected, input_array):
        # print(result, expected)
        # Debugger.pause()
        total_err = self.calculate_initial_error(result, expected)
        # print(initial_err)
        ik = self.get_outputs_from_layer(-2)
        for i, outNeuron in enumerate(self.layers[-1]):
            error = expected[i] - result[i]
            outNeuron.calculate_deltas(self.rate, self.momentum, error, ik)

        for index in range(len(self.layers) - 2, -1, -1):
            weights = np.transpose(self.get_weights_from_layer(index + 1))
            error_info = self.get_errors_from_layer(index + 1)
            total_weights = self.get_total_weight_from_layer(index + 1)

            outputs = []

            if index == 0:
                outputs = input_array.tolist()
                outputs.append(1)
            else:
                outputs = self.get_outputs_from_layer(index - 1)

            for pos, hiddenNeuron in enumerate(self.layers[index]):
                hiddenNeuron.calculate_deltas(self.rate, self.momentum,
                                              error_info, weights[pos], outputs)

        for layer in self.layers:
            for neuron in layer:
                neuron.update_weight()

    def get_outputs_from_layer(self, index):
        outputs = [x.output for x in self.layers[index]]
        outputs.append(1)
        return outputs

    def get_weights_from_layer(self, index):
        return [x.weights for x in self.layers[index]]

    def get_errors_from_layer(self, index):
        return [x.error_info for x in self.layers[index]]

    def get_total_weight_from_layer(self, index):
        result = 0
        for weight in self.get_weights_from_layer(index):
            result += weight
        return result

    def train(self, input_array, expected, x_test, y_test):
        epoch = 0
        stop = False
        while not stop:
            if epoch == 2000:
                stop = True
            for index in range(len(input_array)):
                result = self.propagate(input_array[index])
                # print(input_array[index], result, expected[index])
                self.calculate_errors(result, expected[index],
                                      input_array[index])
            epoch += 1
            if epoch % 25 == 0:
                print("Epoch: ", epoch)
                n_err = self.test(x_test, y_test)
                if n_err == 0:
                    stop = True

                # for neuron in self.layers[-1]:
                # print(neuron.error)

        # weights = []
        # for i, layer in enumerate(self.layers):
        # weights.append(self.get_weights_from_layer(i))
        # np.savetxt('trained_weights.csv', weights, delimiter=',')

    def test(self, x_test, y_test):
        err = 0
        total = 0.0
        for index, received in enumerate(x_test):
            results = self.propagate(x_test[index])
            received = np.array(results).argmax()
            expected = np.array(y_test[index]).argmax()
            if received != expected:
                err += 1
            total += 1

            # result = np.array(results).argmax()
            # expected = np.array(y_test[index]).argmax()
            # print(f"Input {index}: {result} -- "
            #       f"{expected}")
            # if result != expected:
            # err += 1

        print("Number of errors: ", err)
        print("precision: ", 1 - (err / total))
        return err
