import numpy as np

class Network(object):
  
  # sizes parameter contains the number of neurons in the respective layers
  def __init__(self, sizes):
    self.num_layers = len(sizes)
    self.sizes = sizes
    self.biases = [np.random.randn(rows) for rows in sizes[1:]]
    self.weights = [np.random.randn(rows, columns) for columns, rows in zip(sizes[:-1], sizes[1:])]
  
  # sigmoid activation function
  def sigmoid(self, z):
    return 1/(1 + np.exp(-z))

  def feedforward(self, x):
    if self.sizes[0] != len(x):
      return print("Parâmetro 'x' deve possuir o mesmo número de sinais de entrada da rede.")

    y = []

    for bias, weight in zip(self.biases, self.weights):
      if len(y) == 0:
        u = np.dot(weight, x) + bias
      else:
        u = np.dot(weight, y) + bias
      y = self.sigmoid(u)

    return y