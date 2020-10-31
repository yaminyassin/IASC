from camada import camada
from neuronio import neuronio, axonio
import math, random

class rede:
    def __init__(self,*neur_por_camada, codificacao = 0, tipo_ativacao=0):
        self.camadas = self.criar_rede(neur_por_camada, codificacao)
        self.erro = 1 


    def criar_rede(self, neur_por_camada, codificacao):
        camadas = camada()

        for i in range(len(neur_por_camada)):

            camadas.camadas.append([])
            camadas.axonios.append([])

            for n in range(neur_por_camada[i]):

                novo_neuronio = neuronio(bias=1)
                camadas.camadas[i].append(novo_neuronio)
                
                if i != 0: #  para criar ligacoes devemos ter pelo menos 2 camadas
                    for neur_anterior in camadas.camadas[i - 1]:
                        
                        novo_axonio = axonio()  #definir ligacao
                        novo_axonio.peso = random.uniform(0, 1) if codificacao == 0 else random.uniform(-1, 1)

                        novo_axonio.origem = neur_anterior
                        novo_axonio.destino = novo_neuronio

                        #adicionar ligacao aos neuronios
                        neur_anterior.axonios_seguintes.append(novo_axonio)
                        novo_neuronio.axonios_anteriores.append(novo_axonio)
                        camadas.axonios[i].append(novo_axonio)

        return camadas


    def treinar(self, dados_treino, erro_minimo = 0.01, max_iter=20000, lr=0.15):
        iter = 0
        
        while self.erro >= erro_minimo and iter < max_iter:

            treino = random.choice(dados_treino)

            self.propagar(treino)
            self.calcular_erro_total(treino[-1])
            self.retropropagar(treino[-1])
            self.atualizar_pesos(lr)
            iter += 1
            
            print("---------------------------------------")
            print("iter = ", iter)
            print("erro = ", self.erro)

            self.print_rede()
            self.print_pesos()
            self.print_betas()
            
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
        for neuronio_saida in self.camadas.camadas[-1]:
            self.erro += (valor_esperado - neuronio_saida.valor)**2 # ----erro quadratico medio 

        self.erro = math.sqrt(self.erro)


    '''
    calcula os betas dos neuronios
    '''
    def retropropagar(self, valor_esperado ):

        for i in range(-1, -len(self.camadas.camadas), -1):
            for neur in self.camadas.camadas[i]:
                if i == -1: #ultima camada 
                    neur.beta = valor_esperado - neur.valor
                else: #resto das camadas
                    neur.beta = 0 
                    for axon in neur.axonios_seguintes:
                        neur.beta +=  axon.peso * axon.destino.valor * ( 1- axon.destino.valor) * axon.destino.beta
                         


    """
    atualiza os pesos nos axonios
    """
    def atualizar_pesos(self, lr):
        for id_camada in range(len(self.camadas.camadas)-1):
            for id_neur in range(len(self.camadas.camadas[id_camada])):
                neuronio = self.camadas.camadas[id_camada][id_neur]
                for axonio in neuronio.axonios_seguintes:
                    axonio.peso += lr * axonio.origem.valor * axonio.destino.valor * (1 - axonio.destino.valor) * axonio.destino.beta
                neuronio.bias += lr * neuronio.valor * (1 - neuronio.valor) * neuronio.beta

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


if __name__ == '__main__':
    
    n = rede(2,2,1)
    
    dados_treino = [(0, 0, 0),
                    (0, 1, 1),
                    (1, 0, 1),
                    (1, 1, 0)]

    n.treinar(dados_treino, lr=0.15, max_iter=20000)

    print("previsao (0,0) = ", n.prever((0,0)))
    print("previsao (0,1) = ", n.prever((0,1)))
    print("previsao (1,0) = ", n.prever((1,0))) 
    print("previsao (1,1) = ", n.prever((1,1)))