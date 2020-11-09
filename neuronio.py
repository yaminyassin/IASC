import math
class axonio:
    def __init__(self):
        self.peso = None
        self.varPeso = 0
        self.origem = None
        self.destino = None

    def atualizar_peso(self, lr, alpha=0):
        """
        condicao de calculo dependendo do tipo de funcao de ativacao e alfa
        """
        incremento = 0
        momento = alpha * self.varPeso
        if(self.destino.tipo == 0):
            incremento = momento  + lr * self.origem.valor * self.destino.valor * (1 - self.destino.valor) * self.destino.beta 
        else:
            incremento = momento + lr *  self.origem.valor * (1 - self.destino.valor**2) * self.destino.beta 
        self.peso += incremento
        self.varPeso = incremento
              
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
        self.varBias = 0
        self.axonios_anteriores = []
        self.axonios_seguintes = []


    def funcao_ativacao(self, x):
        if self.tipo == 0:
            return self.__sigmoid(x)
        elif self.tipo == 1:
            return  self.__tanh(x)

    def __sigmoid(self, x):
        return 1/(1 + math.exp(x)) if x<0 else 1/(1 + math.exp(-x))

    def __tanh(self, x):
        return math.tanh(x)

    
    def calcular_beta(self, peso, destino_valor, destino_beta):
        if self.tipo == 0:
            self.beta += peso * destino_valor * ( 1- destino_valor) * destino_beta
        elif self.tipo == 1:
            self.beta += peso * (1 - destino_valor**2) *  destino_beta
    
    def atualizar_bias(self, lr, alpha):
        incremento = 0
        momento = alpha * self.varBias
        if(self.tipo == 0):
            incremento = momento + lr * self.valor * (1 - self.valor) * self.beta
        else:
            incremento = momento + lr * (1 - self.valor**2) * self.beta
        self.bias += incremento
        self.varBias = incremento