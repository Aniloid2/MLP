
# Question 3 CODE

from matplotlib import pyplot as plt
import numpy as np


class Perceptron:
	"""docstring for Perceptron"""
	def __init__(self,which):
		self.which = which
		self.limH = 1.1
		self.limL =-0.1


	def plot(self,matrix,weights=None,title="Prediction Matrix"):
		fig,ax = plt.subplots()
		ax.set_title(title)
		ax.set_xlabel("x")
		ax.set_ylabel("d")

		if weights:


			x = np.linspace(-5, 5, 1000)
			y = []
			print (weights)
			for i in range(len(x)):
				if self.which == 'COMP':
					x_i = -(weights[0]/weights[1])
					y.append(x_i)
				else:
					y_i = -(weights[1]/weights[2])*x[i] - weights[0]/weights[2]
					y.append(y_i)
			print (len(y))
			print (len(x))

			if self.which == 'COMP':
				plt.plot(y,x,c='r',label='regression')
			else:
				plt.plot(x,y,c='r',label='regression')



			plot_matrix_x = []
			plot_matrix_y = []
			for i in range(len(matrix)):
				plot_matrix_x.append(matrix[i][1])
				plot_matrix_y.append(matrix[i][2])
				print (matrix[i][-1])
				if self.which == 'COMP':
					if matrix[i][-1] == 0:
						plt.scatter(matrix[i][1],0, c='r', marker='o' )
					else:
						plt.scatter(matrix[i][1], 0, c='b', marker='x')
				else:
					if matrix[i][-1] == 0:
						plt.scatter(matrix[i][1],matrix[i][2], c='r', marker='o' )
					else:
						plt.scatter(matrix[i][1], matrix[i][2], c='b', marker='x')

	def plot_weights(self,learning, weights, title=''):
		fig,ax = plt.subplots()
		ax.set_title(title)
		ax.set_xlabel("learning steps")
		ax.set_ylabel("w")
		w = np.array(weights)
		print (w)

		x = np.linspace(0, learning, learning)


		plt.plot(x,w[:,:1], label='b')
		plt.plot(x,w[:,1:], label='w')


		plt.legend(fontsize=15,loc=1)

		plt.show()

	def predict (self,inputs, weights): #2
		threshold = 0
		total_activation = 0 

		for input, weight in zip(inputs, weights):
			total_activation += input*weight
		return 1 if total_activation >= threshold else 0

	def accuracy(self,matrix,weights): #3
		num_correct = 0
		preds = []
		for i in range(len(matrix)):
			prediction = self.predict (matrix[i][:-1], weights)
			preds.append(prediction)
			if prediction==matrix[i][-1]: num_correct += 1
		print ('Predictions', preds)

		return num_correct//float(len(matrix))

	def train_weights(self,matrix,weights, epoch = 10,	
					 learning_rate=0.01,p=False, stop = True):

		weights_n = []
		learning_steps = 0
		error_n = []
		for epoch in range(epoch):
			acc = self.accuracy(matrix, weights) 
			

			# if acc ==1 and stop: break
			if p: 
				self.plot(matrix,weights, title="Epoch {}".format(epoch))
				plt.ylim(self.limL,self.limH)
				plt.xlim(self.limL,self.limH)
			for entry in range(len(matrix)):
				prediction = self.predict(matrix[entry][:-1], weights)
				error =matrix[entry][-1] - prediction
				learning_steps+=1


				weights_n.append(weights)
				temp_weights = []
				for update in range(len(weights)):
					temp_weights.append(weights[update] + (learning_rate*error*matrix[entry][update]))
				weights =temp_weights



		self.plot(matrix,weights,title='%s, Final epoch, learning rate: %.4f, No epochs %d' % (self.which ,learning_rate, epoch + 1))
		plt.ylim(self.limL,self.limH)
		plt.xlim(self.limL,self.limH)
		self.plot_weights(learning_steps,weights_n, title='Weights versus learning steps, initial weights {} \n {}, final weights {},  learning rate: {}'.format( weights_n[0], self.which,weights, learning_rate))
		return weights

	def main(self): #1

		if self.which == 'AND':
			# AND
			data = [	[1,0,0,0],
						[1,0,1,0],
						[1,1,0,0],
						[1,1,1,1],
						]
			weights = [0.2,0.4,0.8]
		elif self.which == 'NAND':
			# #NAND
			data = [	[1,0,0,1],
						[1,0,1,1],
						[1,1,0,1],
						[1,1,1,0],]
			weights = [0.9,0.1,0.2]

		elif self.which == 'OR':
			# #NAND
			data = [	[1,0,0,0],
						[1,0,1,1],
						[1,1,0,1],
						[1,1,1,1],]
			weights = [0.7,0.4,0.9]
		elif self.which == 'COMP':
			#complememt in 2D
			data = [	[1,0,1],
						[1,1,0],]
			weights = [0.9,-0.4]
		elif self.which == 'EXOR':
			#ExOR
			data = [	[1,0,0,0],
						[1,0,1,1],
						[1,1,0,1],
						[1,1,1,0]]
			weights = [0.7,0.1,0.8]


		self.train_weights(data,weights = weights,p=False,epoch = 30,learning_rate=1)
		plt.show()



Perceptron('EXOR').main()
