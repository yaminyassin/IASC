from camada import camada
import math, random

class rede:
    def __init__(self,*neur_por_camada):
        self.total_camadas = camada(neur_por_camada)
        self.erro = 1 #raiz do erro quadratico medio


    def treinar(self, dados_treino, erro_minimo = 0.015):
        iter = 0
        while self.erro >= erro_minimo:
            treino = random.choice(dados_treino)
            self.propagar(treino)

            self.calcular_erro_total(treino[-1])
            self.retropropagar(treino[-1])
            self.atualizar_pesos(lr = 0.1)
            iter += 1
            print("iter = ", iter)
            print("entradas = ", treino[:-1])
            print("esperado = ", treino[-1])
            print("atual = ", self.total_camadas.camadas[-1][0].valor)
            print("erro = ", self.erro)
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
                for axonio in neuronio.axonios_anteriores:
                    somatorio += axonio.origem.valor * axonio.peso
                somatorio += neuronio.bias
                print(somatorio)
                neuronio.valor = math.round(neuronio.funcao_ativacao(somatorio))

    '''
    Calcula o erro quadratico medio da rede
    '''
    def calcular_erro_total(self, valor_esperado):
        
        self.erro = 0
        for neuronio_saida in self.total_camadas.camadas[-1]:
            self.erro += 0.5 * (valor_esperado - neuronio_saida.valor)**2
            
        self.erro = math.sqrt(self.erro)


    '''
    calcula os betas dos neuronios
    '''
    def retropropagar(self, valor_esperado ):

        for i in range(-1, -len(self.total_camadas.camadas)-1, -1):
            for neur in self.total_camadas.camadas[i]:
                if i == -1: #ultima camada 
                    neur.beta = valor_esperado - neur.valor
                else: #resto das camadas
                    for axon in neur.axonios_seguintes:
                        neur.beta +=  axon.peso * axon.destino.valor * ( 1- axon.destino.valor) * axon.destino.beta
                                

    """
    atualiza os pesos nos axonios
    """
    def atualizar_pesos(self, lr=0.15):
        for id_camada in range(len(self.total_camadas.camadas)-1, 0, -1):
            for id_neur in range(len(self.total_camadas.camadas[id_camada])):
                neuronio = self.total_camadas.camadas[id_camada][id_neur]
                for axonio in neuronio.axonios_anteriores:
                    axonio.peso += lr * axonio.origem.valor * neuronio.valor * (1- neuronio.valor) * neuronio.beta
                neuronio.bias += lr * neuronio.beta

    '''
    imprime a rede neuronal com o valor de cada neuronio
    '''  
    def print_rede(self):
        for i in range(len(self.total_camadas.camadas)):
            print([''.join(str(x.valor)) for x in self.total_camadas.camadas[i]])


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