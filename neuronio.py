import math

class axonio:
    def __init__(self):
        self.peso = None
        self.origem = None
        self.destinatario = None



class neuronio:
    def __init__(self, bias):
       
        self.valor = None
        self.beta = None
        self.bias = bias
        self.axonios_anteriores = []
        self.axonios_seguintes = []

    def funcao_ativacao(self, somatorio):

        return 1/(1+math.exp(-somatorio))
