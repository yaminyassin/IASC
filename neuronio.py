import numpy as np




class neuronio:
    def __init__(self, weights, bias=1, lr=0.15) -> None:
        self.weights = weights
        self.bias = bias
        self.lr = lr




n = neuronio([np.random.random_sample(), np.random.random_sample()])
print(n.bias)
print(n.weights)