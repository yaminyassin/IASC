import numpy as np 
import matplotlib.pyplot as plt
from rede import rede

if __name__ == '__main__':
    dados_treino = [[[0, 0], [0]],
                    [[0, 1], [1]],
                    [[1, 0], [1]],
                    [[1, 1], [0]]]
    
    codificacao = funcao_ativacao = 1
    max_iter = 10000
    lr = [0.01]
    alpha = [0.9]
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
                n = rede(2,2,1, codificacao=codificacao, func_ativacao=funcao_ativacao)

                n.treinar(dados_treino, max_iter=max_iter, lr=lr[j], alpha=alpha[a], erro_minimo=erro, isRandom=isRandom)
                
                
                acertos_por_iter = 0
                acertos_por_iter += 1 if dados_treino[0][1] == n.prever([0,0]) else 0
                acertos_por_iter += 1 if dados_treino[1][1] == n.prever([0,1]) else 0
                acertos_por_iter += 1 if dados_treino[2][1] == n.prever([1,0]) else 0
                acertos_por_iter += 1 if dados_treino[3][1] == n.prever([1,1]) else 0
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
        np.savetxt("xor_iter_alfa={}_codif={}.csv".format(alpha[a], codificacao), final, delimiter=",", fmt='%1.2f')
        
        media2 = np.array([np.mean(resultado2, axis=0)])
        
        resultado2 = np.append(resultado2, media2, axis=0)
        final2 = np.append(indices, resultado2, axis=0)
        final2 = np.insert(final2, 0, range(num_treinos+2), axis=1)
        np.savetxt("xor_freq2_alfa={}_codif={}.csv".format(alpha[a], codificacao), final2, delimiter=",", fmt='%1.2f')
        


    dados_treino2 = [[ [1, 1, 1, 1,
                        1, 0, 0, 1,
                        1, 0, 0, 1,
                        1, 1, 1, 1], [1, 0]],
                    [   [1, 0, 0, 1,
                        0, 1, 1, 0,
                        0, 1, 1, 0,
                        1, 0, 0, 1], [0, 1]]]


    '''
    n2 = rede(16, 1, 2, codificacao=codificacao, func_ativacao=funcao_ativacao)

    n2.treinar(dados_treino2, lr=lr, alpha=alpha, max_iter=max_iter)
    

    teste1 = [1, 1, 1, 1,
             1, 0, 0, 1, #resultado esperado [1, 0]
             1, 0, 0, 1,
             1, 1, 1, 1]

    teste2 = [1, 0, 0, 1,
              0, 1, 1, 0, #resultado esperado [0, 1]
              0, 1, 1, 0,
              1, 0, 0, 1]

    teste3 = [1, 1, 1, 1,
              1, 0, 0, 1, #resultado esperado [1, 0]
              0, 0, 0, 0,
              0, 0, 0, 0]

    teste4 = [0, 0, 0, 0,
              0, 0, 0, 0, #resultado esperado [0, 1]
              0, 1, 1, 0, 
              1, 0, 0, 1]

    teste5 = [1, 1, 0, 0,
              1, 0, 0, 0, #resultado esperado [1, 0]
              1, 0, 0, 0,
              1, 1, 0, 0]

    teste6 = [1, 0, 0, 0,
              0, 1, 0, 0,  #resultado esperado [0, 1]
              0, 0, 1, 0,
              0, 0, 0, 1]

    previsao1 = n2.prever(teste1)
    previsao2 = n2.prever(teste2)
    previsao3 = n2.prever(teste3) 
    previsao4 = n2.prever(teste4)
    previsao5 = n2.prever(teste5)
    previsao6 = n2.prever(teste6)

    print("---------teste 1--------")
    print([''.join(str(x)) for x in previsao1])

    print("----------teste 2----------")
    print([''.join(str(x)) for x in previsao2])

    print("----------teste 3----------")
    print([''.join(str(x)) for x in previsao3])

    print("---------teste 4--------")
    print([''.join(str(x)) for x in previsao4])

    print("----------teste 5----------")
    print([''.join(str(x)) for x in previsao5])

    print("----------teste 6----------")
    print([''.join(str(x)) for x in previsao6])

    '''