import random

class Perceptron:

	def __init__(self, inputs, outputs, learning_rate=0.05, epochs=1000):
		self.inputs = inputs
		self.outputs = outputs
		self.learning_rate = learning_rate
		self.epochs = epochs
		self.n_inputs = len(inputs)
		self.n_inputs_values = len(inputs[0])
		self.weights = [random.uniform(-1, 1) for i in range(self.n_inputs_values)]

		self.__bias = -1

	def step(self, u):
		if u >= 0:
			return 1
		return 0

	def train(self):
  
		# Configure the threshold and w0
		for input in self.inputs:
  			input.insert(0, self.__bias)
		self.weights.insert(0, random.uniform(-1, 0))
		self.n_inputs_values += 1

		for e in range(self.epochs):
				error = False

				for current_input, current_output in zip(self.inputs, self.outputs):
					u = 0
					for x, w in zip(current_input, self.weights):
						u += x * w
					y = self.step(u)

					if y != current_output:
						delta_error = current_output - y
						for i, x in enumerate(current_input):
							self.weights[i] = self.weights[i] + self.learning_rate * delta_error * x
						error = True

				if not error:
						print('Found epoch :', e)
						break

	def test(self, inputs):
		inputs.insert(0, self.__bias)
		u = 0
		for i in range(self.n_inputs_values):
			u += self.weights[i] * inputs[i]
		y = self.step(u)
		return y

inputs = [[0, 0], [0, 1], [1, 0], [1, 1]]
outputs = [0, 1, 1, 1]

n = Perceptron(inputs, outputs)
n.train()

x = [0,0]
print('Class:', n.test(x))
