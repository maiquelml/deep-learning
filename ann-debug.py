import numpy as np

class Network(object):
  
  def __init__(self, sizes):
    self.num_layers = len(sizes)
    self.sizes = sizes

    # Gera uma matriz de ordem ix1 com biases (w0) randomicos para cada camada, com exceção da primeira.
    # Executado com randint para melhor entendimento (rand de 0 a 2).
    self.biases = [np.random.randint(3, size=rows) for rows in sizes[1:]]

    # Gera uma matriz de ordem equivalente ao par ordenado invertido com pesos sinapticos (wn) randomicos para cada par ordenado.
    # Exemplo: sizes = [2,3,1] gera as matrizes A e B, tal que A possuiordem 3x2 e B ordem 1x3
    # Executado com randint para melhor entendimento (rand de 0 a 2).
    self.weights = [np.random.randint(3, size=(rows, columns)) for columns, rows in zip(sizes[:-1], sizes[1:])]

    # Sem random para um examplo [2,3,1]
    self.biases = [
                    [ 0,
                      1,
                      2 ],

                    [ 1 ]
                  ]

    self.weights =  [
                      [ [0,1],
                        [1,1],
                        [0,2] ],

                      [ [0,1,0] ]
                    ]

  def feedforward(self, x):
    y = []

    for bias, weight in zip(self.biases, self.weights):
      if len(y) == 0:
        u = np.dot(weight, x) + bias
      else:
        u = np.dot(weight, y) + bias
      y = u

    return y

layers = [2,3,1]
inputs = [2,1]

n = Network(layers)
print('\nFINAL:', n.feedforward(inputs))
