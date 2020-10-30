from camada import camada
import math, random

class rede:
    def __init__(self,*neur_por_camada):
        self.total_camadas = camada(neur_por_camada)
        self.erro = 1 #raiz do erro quadratico medio


    def treinar(self, dados_treino, erro_minimo = 0.01):
        iter = 0
        while self.erro >= erro_minimo and iter < 20000:

            treino = random.choice(dados_treino)

            self.propagar(treino)
            self.calcular_erro_total(treino[-1])
            self.retropropagar(treino[-1])
            self.atualizar_pesos()
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

        for i in range(len(self.total_camadas.camadas[0])):
            # neuronios da primeira camada tomam valor das entradas
            self.total_camadas.camadas[0][i].valor = dados_treino[i]

        for camada_escondida in self.total_camadas.camadas[1:]:
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
        for neuronio_saida in self.total_camadas.camadas[-1]:
            self.erro += (valor_esperado - neuronio_saida.valor)**2 # ----erro quadratico medio 

        self.erro = math.sqrt(self.erro)


    '''
    calcula os betas dos neuronios
    '''
    def retropropagar(self, valor_esperado ):

        for i in range(-1, -len(self.total_camadas.camadas), -1):
            for neur in self.total_camadas.camadas[i]:
                if i == -1: #ultima camada 
                    neur.beta = valor_esperado - neur.valor
                else: #resto das camadas
                    neur.beta = 0 
                    for axon in neur.axonios_seguintes:
                        neur.beta +=  axon.peso * axon.destino.valor * ( 1- axon.destino.valor) * axon.destino.beta
                         


    """
    atualiza os pesos nos axonios
    """
    def atualizar_pesos(self, lr=0.15):
        for id_camada in range(len(self.total_camadas.camadas)-1):
            for id_neur in range(len(self.total_camadas.camadas[id_camada])):
                neuronio = self.total_camadas.camadas[id_camada][id_neur]
                for axonio in neuronio.axonios_seguintes:
                    axonio.peso += lr * axonio.origem.valor * axonio.destino.valor * (1- axonio.destino.valor) * axonio.destino.beta
                neuronio.bias += lr * neuronio.beta

    '''
    imprime a rede neuronal
    '''  
    def print_rede(self):
        print("Valor dos neuronios")
        for i in range(len(self.total_camadas.camadas)):
            print([''.join(str(x.valor)) for x in self.total_camadas.camadas[i]])
    
    def print_pesos(self):
        print("Pesos dos neuronios")
        for i in range(len(self.total_camadas.axonios)):
            print([''.join(str(x.peso)) for x in self.total_camadas.axonios[i]])

    def print_betas(self):
        print("Betas dos neuronios")
        for i in range(len(self.total_camadas.camadas)):
            print([''.join(str(x.beta)) for x in self.total_camadas.camadas[i]])




    def prever(self, dados_teste):
        dados = []
        self.propagar(dados_teste)
        for saidas in self.total_camadas.camadas[-1]:
            dados.append(round(saidas.valor))
        return dados




if __name__ == '__main__':
    
    n = rede(2,2,1)
    
    dados_treino = [(0, 0, 0),
                    (0, 1, 1),
                    (1, 0, 1),
                    (1, 1, 0)]


    n.treinar(dados_treino)

    print("previsao (0,1) = ", n.prever((0,1)))
    print("previsao (1,0) = ", n.prever((1,0)))
    print("previsao (0,0) = ", n.prever((0,0)))
    print("previsao (0,0) = ", n.prever((0,0)))
    print("previsao (1,1) = ", n.prever((1,1)))
    print("previsao (1,0) = ", n.prever((1,0)))
    print("previsao (0,0) = ", n.prever((0,0))) 
    print("previsao (1,1) = ", n.prever((1,1)))
    print("previsao (1,0) = ", n.prever((1,0)))