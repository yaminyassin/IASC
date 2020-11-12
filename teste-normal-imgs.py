import numpy as np 
import matplotlib.pyplot as plt
from rede import rede

if __name__ == '__main__':

    dados_treino = [
                        [[1, 1, 1, 1,
                          1, 0, 0, 1,
                          1, 0, 0, 1,
                          1, 1, 1, 1], [1, 0]],
                        [[1, 0, 0, 1,
                          0, 1, 1, 0,
                          0, 1, 1, 0,
                          1, 0, 0, 1], [0, 1]]
                    ]

    teste1 = [[1, 1, 1, 1],
              [1, 0, 0, 1], #resultado esperado [1, 0]
              [0, 0, 0, 0],
              [0, 0, 0, 0]]

    teste2 = [0, 0, 0, 0,
              0, 0, 0, 0, #resultado esperado [0, 1]
              0, 1, 1, 0, 
              1, 0, 0, 1]

    teste3 = [1, 1, 0, 0,
              1, 0, 0, 0, #resultado esperado [1, 0]
              1, 0, 0, 0,
              1, 1, 0, 0]

    teste4 = [1, 0, 0, 0,
              0, 1, 0, 0,  #resultado esperado [0, 1]
              0, 0, 1, 0,
              0, 0, 0, 1]
                    
    codificacao = funcao_ativacao = 1
    max_iter = 20000
    lr = [0.1, 0.15, 0.2]
    alpha = [0.2]
    erro = 0.1
    num_treinos = 10
    isRandom = True

    for a in range(len(alpha)):
        resultado = np.zeros((num_treinos, len(lr)))
        resultado2 = np.zeros((num_treinos, len(lr)))
        for i in range(len(resultado)):
            res = np.zeros(len(lr))
            res2 = np.zeros(len(lr))
            for j in range(len(lr)):
                print("lr = {}, alpha = {}".format(lr[j], alpha[a]))
                n = rede(16, 2, 2, codificacao=codificacao, func_ativacao=funcao_ativacao)

                n.treinar(dados_treino, max_iter=max_iter, lr=lr[j], alpha=alpha[a], erro_minimo=erro, isRandom=isRandom)

                previsao1 = n.prever(teste1)
                previsao2 = n.prever(teste2)
                previsao3 = n.prever(teste3) 
                previsao4 = n.prever(teste4)

                acertos_por_iter = 0
                acertos_por_iter += 1 if [1,0] == previsao1 else 0
                acertos_por_iter += 1 if [0,1] == previsao2 else 0
                acertos_por_iter += 1 if [1,0] == previsao3 else 0
                acertos_por_iter += 1 if [0,1] == previsao4 else 0
                acertos_por_iter /= 4

                res2[j] = acertos_por_iter  # PERCENTAGEM DE ACERTOS 
                res[j] = n.iter  #NUMERO DE ITERACOES
            
            resultado[i] = res
            resultado2[i] = res2
        
        indices = np.array([lr])

        media = np.array([np.mean(resultado, axis=0)])
        resultado = np.append(resultado, media, axis=0)
        final = np.append(indices, resultado, axis=0)
        final = np.insert(final, 0, range(num_treinos+2), axis=1)
        np.savetxt("imgs_bipolar_iter_{}.csv".format(alpha[a]), final, delimiter=",", fmt='%1.2f')
        
        media2 = np.array([np.mean(resultado2, axis=0)])
        resultado2 = np.append(resultado2, media2, axis=0)
        final2 = np.append(indices, resultado2, axis=0)
        final2 = np.insert(final2, 0, range(num_treinos+2), axis=1)
        np.savetxt("imgs_bipolar_freq_{}.csv".format(alpha[a]), final2, delimiter=",", fmt='%1.2f')
        


    print("---------teste 1--------")
    print([''.join(str(x)) for x in teste1]) 
    print('resultado esperado [1, 0]')

    print("----------teste 2----------")
    print([''.join(str(x)) for x in teste2]) 
    print('resultado esperado [0, 1]')

    print("----------teste 3----------")
    print([''.join(str(x)) for x in teste3]) 
    print('resultado esperado [1, 0]')

    print("----------teste 4----------")
    print([''.join(str(x)) for x in teste4]) 
    print('resultado esperado [0, 1]')