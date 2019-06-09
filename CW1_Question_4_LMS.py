#Question 4 LMS
from matplotlib import pyplot as plt
import numpy as np

def plot(matrix,weights=None,title=""):

	fig,ax = plt.subplots()
	ax.set_title(title)
	ax.set_xlabel("x")
	ax.set_ylabel("d")

	if weights:


		x = np.linspace(0, 5, 1000)
		y = []
		for i in range(len(x)):
			y.append(x[i]*weights[-1] + weights[0])
		print (len(y))
		print (len(x))

		plt.plot(x,y,c='r',label='regression')
		plot_matrix_x = []
		plot_matrix_y = []
		for i in range(len(matrix)):
			plot_matrix_x.append(matrix[i][1])
			plot_matrix_y.append(matrix[i][2])


		plt.scatter(plot_matrix_x,plot_matrix_y)


def plot_weights(learning, weights, title=''):
	fig,ax = plt.subplots()
	ax.set_title(title)
	ax.set_xlabel("learning steps")
	ax.set_ylabel("w")
	w = np.array(weights)

	x = np.linspace(0, learning, learning)


	plt.plot(x,w[:,:1], label='b')
	plt.plot(x,w[:,1:], label='w')


	plt.legend(fontsize=15,loc=1)

	plt.show()


def pred (inputs, weights): #2
	Mat_mult = 0
	for input, weight in zip(inputs, weights):
		Mat_mult += input*weight
	return Mat_mult




def train_weights(matrix,weights, nb_e = 100,	
				 learning_rate=0.5):
	
	weights_n = []
	learning_steps = 0
	error_n = []
	for epoch in range(nb_e):
		for entry in range(len(matrix)):
			prediction = pred(matrix[entry][:-1], weights)
			error = matrix[entry][-1] - prediction
			
			learning_steps+=1
			error_n.append(error)

			print (weights)
			weights_n.append(weights)
			temp_weights = []
			for update in range(len(weights)):
				temp_weights.append(weights[update] + (learning_rate*error*matrix[entry][update]))
			weights = temp_weights
			

	print ('Final weights: %.4f, %.4f' %( weights[0] , weights[1]))
	plot(matrix,weights,title='LMS learning epoch:%d, learning rate %.3f,\n Final weights: %.4f, %.4f' % (nb_e, learning_rate,weights[0] , weights[1]))
	plot_weights(learning_steps,weights_n, title='weights versus learning steps')
	return weights

def main(): #1


	data =[	[1, 0 ,  0.5],
			[1, 0.8, 1  ],
			[1, 1.6, 4  ],
			[1, 3.0, 5  ],
			[1, 4.0, 6  ],
			[1, 5.0, 8  ]]

	weights = [1,1]


	Ow = train_weights(data,weights = weights)

	print ('Final weights: %.4f, %.4f' %( Ow[0] , Ow[1]))


main()