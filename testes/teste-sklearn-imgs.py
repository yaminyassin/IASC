import numpy as np
from sklearn.neural_network import MLPClassifier

dados_treino = np.array([
    [1, 1, 1, 1,
    1, 0, 0, 1,
    1, 0, 0, 1,
    1, 1, 1, 1],
    
    [1, 0, 0, 1,
    0, 1, 1, 0,
    0, 1, 1, 0,
    1, 0, 0, 1]
])

output_treino = np.array([
    [1,0],
    [0,1]
])


teste1 = [1, 1, 1, 1,
        1, 0, 0, 1, #resultado esperado [1, 0]
        0, 0, 0, 0,
        0, 0, 0, 0]

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


dados_teste = np.array([teste1, teste2, teste3, teste4])

output_teste = np.array([[1,0],[0,1],[1,0],[0,1]])


max_iter = 20000
lr = [0.1, 0.15, 0.2]
alpha = [0.1]
num_treinos = 10
isRandom = True

for a in range(len(alpha)):
    resultado = np.zeros((num_treinos, len(lr)))
    resultado2 = np.zeros((num_treinos, len(lr)))
    
    for i in range(len(resultado)):
        res = np.zeros(len(lr))
        res2 = np.zeros(len(lr))
        
        for j in range(len(lr)):
            n = MLPClassifier(hidden_layer_sizes=(1),
                    solver='sgd',
                    activation='tanh',
                    learning_rate_init=lr[j],
                    momentum=alpha[a],
                    max_iter=max_iter
            )

            n.fit(dados_treino, output_treino)

            print(n.predict(dados_teste))

            res2[j] = n.score(dados_teste, output_teste)  # freq de acerto 
            res[j] = n.n_iter_  #num de iteracoes
        
        resultado[i] = res
        resultado2[i] = res2
    
    indices = np.array([lr])

    media = np.array([np.mean(resultado, axis=0)])
    resultado = np.append(resultado, media, axis=0)
    final = np.append(indices, resultado, axis=0)
    final = np.insert(final, 0, range(num_treinos+2), axis=1)
    np.savetxt("sklearn_imgs_iter_alfa={}.csv".format(alpha[a]), final, delimiter=",", fmt='%1.2f')
    
    media2 = np.array([np.mean(resultado2, axis=0)])
    resultado2 = np.append(resultado2, media2, axis=0)
    final2 = np.append(indices, resultado2, axis=0)
    final2 = np.insert(final2, 0, range(num_treinos+2), axis=1)
    np.savetxt("sklearn_imgs_score_alfa={}.csv".format(alpha[a]), final2, delimiter=",", fmt='%1.2f')
