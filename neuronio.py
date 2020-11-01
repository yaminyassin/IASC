import math

class axonio:
    def __init__(self):
        self.peso = None
        self.origem = None
        self.destino = None



class neuronio:
    def __init__(self, bias, tipo=0):
        self.tipo = tipo
        self.valor = 0
        self.beta = 0
        self.bias = bias
        self.axonios_anteriores = []
        self.axonios_seguintes = []


    def funcao_ativacao(self, x):
        if self.tipo == 0:
            return self.__sigmoid(x)
        elif self.tipo == 1:
            return  self.__tanh(x)

    def __sigmoid(self, x):
        return 1/(1 + math.exp(-x))

    def __tanh(self, x):
        return (2 / (1 + math.exp(-2*x)) ) - 1

    
    def calcular_beta(self, peso, destino_valor, destino_beta):
        self.beta += peso * destino_valor * ( 1- destino_valor) * destino_beta