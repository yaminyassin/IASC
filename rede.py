from neuronio import neuronio, axonio
from camada import camada


class rede:
    def __init__(self,*neur_por_camada) -> None:
        self.total_camadas = camada(neur_por_camada)




    def retropropagacao(self, dados_treino):
        pass

if __name__ == '__main__':

    n = rede(2,2,1)

