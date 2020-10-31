import math

class axonio:
    def __init__(self):
        self.peso = None
        self.origem = None
        self.destino = None



class neuronio:
    def __init__(self, bias, tipo="sigmoid"):
        self.tipo = tipo
        self.valor = 0
        self.beta = 0
        self.bias = bias
        self.axonios_anteriores = []
        self.axonios_seguintes = []

    def funcao_ativacao(self, x):
        if self.tipo == "sigmoid":
            return 1/(1 + math.exp(-x))
        elif self.tipo == "tanh":
            return  (2 / (1 + math.exp(-2*x)) ) - 1
