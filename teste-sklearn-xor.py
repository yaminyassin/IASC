import numpy as np
from sklearn.neural_network import MLPClassifier

inputs = np.array([[0,0],[0,1],[1,0],[1,1]])
expected_output = np.array([0,1,1,0])

model = MLPClassifier(
    hidden_layer_sizes=(2),
    activation='tanh',
    solver='sgd',
    learning_rate_init=0.0015,
    max_iter=20000,
    momentum=0.9
    )

model.fit(inputs, expected_output)

print("n layers = ", model.n_layers_)
print("n_iter = ", model.n_iter_)

print("predictions: ", model.predict(inputs))
print("score = ", model.score(inputs, expected_output))
