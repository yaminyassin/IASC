import math

class axonio:
    def __init__(self):
        self.peso = None
        self.origem = None
        self.destino = None



class neuronio:
    def __init__(self, bias):
       
        self.valor = 0
        self.beta = 0
        self.bias = bias
        self.axonios_anteriores = []
        self.axonios_seguintes = []

    def funcao_ativacao(self, somatorio):
        return 1/(1+math.exp(-somatorio))
