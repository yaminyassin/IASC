from neuronio import neuronio, axonio
from camada import camada


class rede:
    def __init__(self,*neur_por_camada):
        self.total_camadas = camada(neur_por_camada)



    '''
    @dados_treino é um array de um tuplo de 3 posicoes (x, y, resultado)
    '''
    def retropropagacao(self, dados_treino):
        
        for i in range(len(self.total_camadas.camadas[0])):
            #neuronios da primeira camada tomam valor das entradas
            self.total_camadas.camadas[0][i].valor = dados_treino[0][i]

        for camada_escondida in self.total_camadas.camadas[1:]:
            for neuronio in camada_escondida:
                somatorio = 0
                for axonio in neuronio.axonios_anteriores:
                    somatorio += axonio.origem.valor * axonio.peso
                somatorio += neuronio.bias

                neuronio.valor = neuronio.funcao_ativacao(somatorio)

    def print_rede(self):
        for i in range(len(self.total_camadas.camadas)):
            print([''.join(str(x.valor)) for x in self.total_camadas.camadas[i]])


if __name__ == '__main__':

    n = rede(2,2,1)

    dados_treino = [(0, 0, 0),
                    (0, 1, 1),
                    (1, 0, 1),
                    (1, 1, 0)]

    n.retropropagacao(dados_treino)

    n.print_rede()

