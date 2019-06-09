#Question 4 LLS

from matplotlib import pyplot as plt
import numpy as np

def plot(matrix,weights=None,title=""):

	fig,ax = plt.subplots()
	ax.set_title(title)
	ax.set_xlabel("x")
	ax.set_ylabel("d")

	if weights:
		x = np.linspace(0, 5, 100)
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
		plt.show()
def predict (inputs, weights): #2
	total_activation = 0
	for input, weight in zip(inputs, weights):
		total_activation += input*weight
	return total_activation
def train_weights(matrix,weights = None, nb_epoch = 150,	
				 l_rate=0.1,do_plot=False, stop_early = True, verbose=True):
	X = np.zeros((len(matrix),len(matrix[0])-1))
	d = np.zeros((len(matrix),1))
	for i in range(len(matrix)):
		row = matrix[i][:-1]
		X[i,:] = row
		d[i] = matrix[i][-1]
	w =  np.matmul(np.matmul(np.linalg.inv(np.matmul(X.transpose(),X)),X.transpose()),d)
	print (w)
	w = [i for i in w]
	return w
def main(): 

	data =[	[1, 0 ,  0.5],
			[1, 0.8, 1  ],
			[1, 1.6, 4  ],
			[1, 3.0, 5  ],
			[1, 4.0, 6  ],
			[1, 5.0, 8  ]]




	Oweights = train_weights(data,do_plot=True)

	plot(data,Oweights,title='LLS Final weights: %.4f, %.4f' % (Oweights[0], Oweights[1]))


main()