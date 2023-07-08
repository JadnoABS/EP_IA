import numpy as np
from math import ceil

from mlp.neurons import HiddenNeuron, OutNeuron
from mlp.Debugger import Debugger


class NeuralNetwork:
    layers: list = []

    def __init__(self, n_layers, rate,
                 momentum, x_train=np.array([]), y_train=np.array([]),
                 cross_validation=False, k_folds = 4, anticapte_stop=True):
        self.anticapte_stop = anticapte_stop
        self.k_folds = k_folds;
        self.cross_validation = cross_validation

        self.construct_layers(n_inputs=x_train.shape[1],
                                n_outputs=y_train.shape[1],
                                n_layers=n_layers)
        self.rate = rate
        self.momentum = momentum

    def construct_layers(self, n_inputs, n_outputs, n_layers):
        self.layers = []

        n_hidden = ceil((n_inputs / 3) * 2)
        self.layers.append(
            [HiddenNeuron(next_layer_size=n_inputs) for _ in range(n_hidden)])

        for i in range(n_layers - 2):
            self.layers.append([HiddenNeuron(next_layer_size=n_hidden) for _ in
                                range(n_hidden)])

        self.layers.append(
            [OutNeuron(next_layer_size=n_hidden) for _ in range(n_outputs)])

    def propagate(self, inputs: list):
        next_input = inputs
        for index, layer in enumerate(self.layers):
            input_arr = np.array(next_input)
            next_input = []
            for neuron in layer:
                output = neuron.activate(input_arr)
                next_input.append(output)

        return next_input

    def calculate_errors(self, result, expected, input_array):

        ik = self.get_outputs_from_layer(-2)
        for i, outNeuron in enumerate(self.layers[-1]):
            error = expected[i] - result[i]
            outNeuron.calculate_deltas(self.rate, self.momentum, error, ik)

        for index in range(len(self.layers) - 2, -1, -1):
            weights = np.transpose(self.get_weights_from_layer(index + 1))
            error_info = self.get_errors_from_layer(index + 1)

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

    def train(self, input_array, expected):
        epoch = 0
        stop = False
        max_precision_epochs = 0
        current_fold = 0
        while not stop:
            last_index = current_fold * self.k_folds + len(input_array) // self.k_folds if (current_fold * self.k_folds + len(input_array) // self.k_folds) < len(input_array) else len(input_array)
            initial_index = current_fold * self.k_folds
            if epoch == 1000:
                stop = True
            training_errors = 0
            if self.cross_validation:
                for index in range(len(input_array)):
                    if index < last_index and index >= initial_index:
                        pass
                    else:
                        result = self.propagate(input_array[index])
                        # print(self.convert_to_alphabet(result), self.convert_to_alphabet(expected[index]))
                        self.calculate_errors(result, expected[index],
                                            input_array[index])
                        if np.array(result).argmax() != np.array(expected[index]).argmax():
                            training_errors += 1
            else:
                for index in range(len(input_array)):
                    result = self.propagate(input_array[index])
                    # print(self.convert_to_alphabet(result), self.convert_to_alphabet(expected[index]))
                    self.calculate_errors(result, expected[index],
                                        input_array[index])
                    if np.array(result).argmax() != np.array(expected[index]).argmax():
                        training_errors += 1
            epoch += 1
            current_fold = current_fold + 1 if current_fold < self.k_folds else 0
            # Debugger.pause()
            print('\n', end='')
            print("Epoch: ", epoch)
            if self.cross_validation:
                precision = self.test(input_array[initial_index:last_index], expected[initial_index:last_index])
            else:
                precision = self.test(input_array, expected)
            if precision >= 1:
                max_precision_epochs += 1
            else:
                max_precision_epochs = 0
            if self.anticapte_stop and max_precision_epochs >= (50 if self.cross_validation else 5):
                stop = True

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

        precision = 1 - (err / total)

        print("Number of errors: ", err)
        print("precision: ", precision)
        return precision

    def convert_to_alphabet(self, results):
        max_index = np.array(results).argmax()
        match max_index:
            case 0:
                return 'A'
            case 1:
                return 'B'
            case 2:
                return 'C'
            case 3:
                return 'D'
            case 4:
                return 'E'
            case 5:
                return 'J'
            case 6:
                return 'K'
