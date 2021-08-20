from typing import List
import math

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

class Node:
    def __init__(self, weights: list[float], bias: float):
        self.weights = weights
        self.bias = bias
        self.degree = len(weights)

    def calculate(self, inputs: list[float]) -> float:
        list = [inputs[i] * self.weights[i] for i in range(self.degree)]
        s = sum(list) + self.bias
        return sigmoid(s)

class Layer:
    def __init__(self, nodes: list[Node]):
        self.nodes = nodes

    def predict(self, inputs: list[float]) -> list[float]:
        return [n.calculate(inputs) for n in self.nodes]

class Network:
    def __init__(self, layers: list[Layer]):
        self.layers = layers
        self.degree = len(layers)

    def predict(self, inputs: list[float]) -> list[float]:
        for l in self.layers:
            inputs = l.predict(inputs)
        return inputs


h1 = Node([0.15, 0.2], 0.35)
h2 = Node([0.25, 0.3], 0.35)

o1 = Node([0.40, 0.45], 0.6)
o2 = Node([0.50, 0.55], 0.6)

layer_h = Layer([h1, h2])
layer_o = Layer([o1, o2])

print(layer_h.predict([0.05, 0.1]))
print(layer_o.predict([0.05, 0.1]))

network = Network([layer_h, layer_o])

print(network.predict([0.05, 0.1]))
