import random
from neuronio import neuronio, axonio



class camada:
    def __init__(self, neur_por_camada):
        self.camadas = []
        
        for i in range(len(neur_por_camada)):
            self.camadas.append([])

            for n in range(neur_por_camada[i]):
                novo_neuronio = neuronio(bias=random.uniform(0, 1) if i != 0 else 0)
                self.camadas[i].append(novo_neuronio)
                
                """
                para criar ligacoes devemos ter pelo menos 2 camadas
                """
                if i != 0:
                    for neur_anterior in self.camadas[i - 1]:

                        #definir ligacao
                        novo_axonio = axonio()
                        novo_axonio.origem = neur_anterior
                        novo_axonio.destino = novo_neuronio
                        novo_axonio.peso = random.uniform(0, 1)

                        #adicionar ligacao aos neuronios
                        neur_anterior.axonios_seguintes.append(novo_axonio)
                        novo_neuronio.axonios_anteriores.append(novo_axonio)
                        
       