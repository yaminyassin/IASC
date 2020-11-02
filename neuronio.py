import math

class axonio:
    def __init__(self):
        self.peso = None
        self.variacaoPeso = 0
        self.origem = None
        self.destino = None

    def atualizar_peso(self, lr, alpha=0):
        calculo = 0 
        if alpha == 0:
            if(self.destino.tipo == 0):
                self.peso += lr * self.origem.valor * self.destino.valor * (1 - self.destino.valor) * self.destino.beta
            else:
                self.peso += lr * self.origem.valor * (1 - self.destino.valor**2) * self.destino.beta

        else:
            if(self.destino.tipo == 0):
                calculo = alpha * self.variacaoPeso - lr * self.origem.valor * self.destino.valor * (1 - self.destino.valor) * self.destino.beta 
                self.peso += calculo
                self.variacaoPeso = calculo
                
            else:
                calculo = alpha * self.variacaoPeso - lr * self.origem.valor * (1 - self.destino.valor**2) * self.destino.beta 
                self.peso += calculo
                self.variacaoPeso = calculo

              
class neuronio:
    """
    classe que representa um neuronio. \n

    @tipo -> tipo de funcao de ativacao a usar \n
        0 -> sigmoid \n
        1 -> tangente Hiperbolica \n
    """
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
        if self.tipo == 0:
            self.beta += peso * (destino_valor * ( 1- destino_valor)) * destino_beta
        elif self.tipo == 1:
            self.beta += peso * (1 - destino_valor**2) *  destino_beta
    
    def atualizar_bias(self, lr):
        if(self.tipo == 0):
            self.bias += lr * self.valor * (1 - self.valor) * self.beta
        else:
            self.bias += lr * (1 - self.valor**2) * self.beta