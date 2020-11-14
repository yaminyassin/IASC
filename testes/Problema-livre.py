import numpy as np
import pandas as p
from sklearn.datasets import load_iris
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

iris = load_iris()


data = p.DataFrame(
    data=np.c_[iris['data'], iris['target']],
    columns= iris['feature_names'] + ['target']
) 


iris_data = iris.data
iris_target = iris.target

dados_treino, dados_teste, output_treino, output_teste = train_test_split(iris_data, iris_target, train_size=0.45)


print(len(dados_treino))
print(len(dados_teste))

max_iter = 20000
lr = [0.1, 0.15, 0.2]
alpha = [0.1, 0.2]
num_treinos = 10
isRandom = True
func_ativacao ="tanh"
hidden_layers =(1)

for a in range(len(alpha)):
    resultado = np.zeros((num_treinos, len(lr)))
    resultado2 = np.zeros((num_treinos, len(lr)))
    
    for i in range(len(resultado)):
        res = np.zeros(len(lr))
        res2 = np.zeros(len(lr))
        
        for j in range(len(lr)):
            n = MLPClassifier(hidden_layer_sizes=hidden_layers,
                    solver='sgd',
                    activation=func_ativacao,
                    learning_rate_init=lr[j],
                    momentum=alpha[a],
                    max_iter=max_iter
            )

            n.fit(dados_treino, output_treino)

            res2[j] = n.score(dados_teste, output_teste)  # freq de acerto 
            res[j] = n.n_iter_  #num de iteracoes
        
        resultado[i] = res
        resultado2[i] = res2
    
    indices = np.array([lr])

    media = np.array([np.mean(resultado, axis=0)])
    resultado = np.append(resultado, media, axis=0)
    final = np.append(indices, resultado, axis=0)
    final = np.insert(final, 0, range(num_treinos+2), axis=1)
    
    media2 = np.array([np.mean(resultado2, axis=0)])
    resultado2 = np.append(resultado2, media2, axis=0)
    final2 = np.append(indices, resultado2, axis=0)
    final2 = np.insert(final2, 0, range(num_treinos+2), axis=1)

    final = np.append(final, final2, axis=0)

    np.savetxt("{}/{}iris_alfa{}.csv".format(func_ativacao, hidden_layers, alpha[a]), final, delimiter=",", fmt='%1.2f')
