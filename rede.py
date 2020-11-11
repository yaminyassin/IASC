from camada import camada
from neuronio import neuronio, axonio
import math, random
import numpy as np



class rede:
    """
    Classe rede que representa uma rede neuronal \n
    NOTA-SE: os pesos e bias sao inicializados com valores aleatorios dependendo do tipo de codificacao \n
    neur_por_camada -> tuplo que representa quantos neuronios e camadas irao ser instanciados na rede \n
    codificacao -> o tipo de codificacao a se usar:\n
        0 -> Binario \n
        1 -> Bipolar \n
    func_ativacao -> o tipo de funcao de ativacao que os neuronios irao usar: \n
        0 -> Sigmoid \n
        1 -> Tangente Hiperbolica \n
    """
    def __init__(self,*neur_por_camada, codificacao = 0, func_ativacao=0):
        self.camadas = self.criar_rede(neur_por_camada, codificacao, func_ativacao)
        self.erro = 1 
        self.iter = 0

    def criar_rede(self, neur_por_camada, codificacao, func_ativacao):
        camadas = camada()

        for i in range(len(neur_por_camada)):

            camadas.camadas.append([])
            camadas.axonios.append([])

            for n in range(neur_por_camada[i]):

                #novo_neuronio = neuronio(bias=random.uniform(0, 1) if codificacao == 0 else random.uniform(-1, 1), tipo=func_ativacao)

                novo_neuronio = neuronio(bias=1, tipo=func_ativacao)
                camadas.camadas[i].append(novo_neuronio)
                
                if i != 0: #  para criar ligacoes devemos ter pelo menos 2 camadas
                    for neur_anterior in camadas.camadas[i - 1]:
                        
                        novo_axonio = axonio()  #criar axonio
                        novo_axonio.peso = random.uniform(0, 1) if codificacao == 0 else random.uniform(-1, 1)

                        novo_axonio.origem = neur_anterior
                        novo_axonio.destino = novo_neuronio

                        #adicionar ligacao aos neuronios
                        neur_anterior.axonios_seguintes.append(novo_axonio)
                        novo_neuronio.axonios_anteriores.append(novo_axonio)
                        camadas.axonios[i].append(novo_axonio)
        return camadas


    def treinar(self, dados_treino, erro_minimo = 0.03, max_iter=20000, lr=0.15, alpha=0, isRandom=True):
        """
        dados_treino -> array com dados de treino, em que a ultima camada é o valor esperado \n
        erro_minimo -> erro \n
        max_iter -> iteracoes maximas da rede \n 
        lr -> taxa de aprendizagem \n
        alpha -> valor de momento \n
        """

        self.iter = 0
        while  self.iter < max_iter and self.erro >= erro_minimo: 
            if isRandom:
                treino = random.choice(dados_treino) 
            else:
                treino = dados_treino[self.iter % len(dados_treino)]
            entradas = treino[0]
            saidas = treino[1]
            self.propagar(entradas)
            self.calcular_erro_total(saidas)
            self.retropropagar(saidas)
            self.atualizar_pesos(lr, alpha)

            self.iter += 1
        print("---------------------------------------")
        print("iter = ", self.iter)
        #print("alpha = ", alpha)
        #print("erro = ", self.erro)
        #self.print_rede()
        #self.print_pesos()
        #self.print_betas()
        print("--------------------------------------")
    	    

    """
    @dados_treino é um tuplo de 3 posicoes (x, y, resultado)
    """
    def propagar(self, dados_treino):

        for i in range(len(self.camadas.camadas[0])):
            # neuronios da primeira camada tomam valor das entradas
            self.camadas.camadas[0][i].valor = dados_treino[i]

        for camada_escondida in self.camadas.camadas[1:]:
            for neuronio in camada_escondida:
                somatorio = 0
                for axonio in neuronio.axonios_anteriores:         #---somatorio
                    somatorio += axonio.origem.valor * axonio.peso
                somatorio += neuronio.bias
                neuronio.valor = neuronio.funcao_ativacao(somatorio)

    '''
    Calcula o erro quadratico medio da rede
    '''
    def calcular_erro_total(self, valor_esperado):
        
        self.erro = 0
        for count, neuronio_saida in enumerate(self.camadas.camadas[-1]):
            self.erro += (valor_esperado[count] - neuronio_saida.valor)**2 # ----erro quadratico medio 
        self.erro = math.sqrt(self.erro)


    '''
    calcula os betas dos neuronios
    '''
    def retropropagar(self, valor_esperado):

        for i in range(-1, -len(self.camadas.camadas), -1):
            for count, neur in enumerate(self.camadas.camadas[i]):
                if i == -1: #ultima camada 
                    neur.beta = valor_esperado[count] - neur.valor
                else: #resto das camadas
                    neur.beta = 0 
                    for axon in neur.axonios_seguintes:
                        neur.calcular_beta(axon.peso, axon.destino.valor, axon.destino.beta)
                         


    """
    atualiza os pesos nos axonios
    """
    def atualizar_pesos(self, lr,  alpha):
        for id_camada in range(len(self.camadas.camadas)-1):
            for id_neur in range(len(self.camadas.camadas[id_camada])):
                neuronio = self.camadas.camadas[id_camada][id_neur]
                for axonio in neuronio.axonios_seguintes:
                    axonio.atualizar_peso( lr, alpha)
                neuronio.atualizar_bias(lr, alpha)

    '''
    imprime a rede neuronal
    '''  
    def print_rede(self):
        print("Valor dos neuronios")
        for i in range(len(self.camadas.camadas)):
            print([''.join(str(x.valor)) for x in self.camadas.camadas[i]])
    
    def print_pesos(self):
        print("Pesos dos neuronios")
        for i in range(len(self.camadas.axonios)):
            print([''.join(str(x.peso)) for x in self.camadas.axonios[i]])

    def print_betas(self):
        print("Betas dos neuronios")
        for i in range(len(self.camadas.camadas)):
            print([''.join(str(x.beta)) for x in self.camadas.camadas[i]])


    def prever(self, dados_teste):
        dados = []
        self.propagar(dados_teste)
        for saidas in self.camadas.camadas[-1]:
            dados.append(round(saidas.valor))
        return dados
