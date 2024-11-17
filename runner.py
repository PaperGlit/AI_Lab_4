import numpy as np

from GlobalVariables import letter1, letter2, noise


def sign(x):
    return np.where(x >= 0, 1, -1)

class HopfieldNetwork:
    def __init__(self, n_neurons):
        self.n_neurons = n_neurons
        self.weights = np.zeros((n_neurons, n_neurons))

    def train(self, patterns):
        for p in patterns:
            self.weights += np.outer(p, p)
        np.fill_diagonal(self.weights, 0)

    def recall(self, pattern, steps=5):
        state = pattern.copy()
        for _ in range(steps):
            state = sign(np.dot(self.weights, state))
        return state

if __name__ == "__main__":

    network = HopfieldNetwork(n_neurons=len(letter1))
    network.train([letter1, letter2])

    restored_pattern = network.recall(noise)

    print("Оригінальний патерн №1:")
    letter1_2d = np.reshape(letter1, (5, 3))
    print(letter1_2d)
    print("Оригінальний патерн №2:")
    letter2_2d = np.reshape(letter2, (5, 3))
    print(letter2_2d)
    print("Шумовий патерн:")
    noise_2d = np.reshape(noise, (5, 3))
    print(noise_2d)
    print("Відновлений патерн:")
    restored_pattern_2d = np.reshape(restored_pattern, (5, 3))
    print(restored_pattern_2d)
