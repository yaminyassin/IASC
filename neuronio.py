class axonio:
    def __init__(self) -> None:
        self.peso = None
        self.origem = None
        self.destinatario = None



class neuronio:
    def __init__(self, bias) -> None:
       
        self.saida = None
        self.beta = None
        self.bias = bias
        self.axonios_anteriores = []
        self.axonios_seguintes = []


