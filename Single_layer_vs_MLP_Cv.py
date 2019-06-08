import sys
import os
from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier
from sklearn.utils import shuffle
from PIL import Image
import numpy as np
class Computer_vision:
	def __init__(self, folders):
		self.folders =folders
		self.Categories = []
		self.mean_vector = 0
		self.variance_vector = 0

	def unpack_folder(self):
		for i in range(len(self.folders)):
			self.Categories.append(Categ( self.folders[i]))
			self.Categories[i].label_set(i)
			self.Categories[i].initialise_image_size()
			self.Categories[i].train_validation_split()

	def Single_perceptron(self, average):
		[train_data, train_lables, validation_data, validation_lables ] = self.load_data(average)
		clf = Perceptron(tol=None, random_state=0,  max_iter=100) # 8 epochs best result when generalising
		# tol = The stopping criterion. If it is not None, the iterations will stop when (loss > previous_loss - tol). 
		#Defaults to None. Defaults to 1e-3 from 0.21.
		clf.fit(train_data, train_lables)
		train_score = clf.score(train_data, train_lables)
		validation_score = clf.score(validation_data, validation_lables)
		print ('Perceptron Train score:',train_score,', Validation score:',validation_score)

	def MLP(self, training_method, average, L2 = 0.1):
		[train_data, train_lables, validation_data, validation_lables ] = self.load_data(average)
		if training_method == 'batch':
			clf = MLPClassifier(solver='sgd', activation='logistic', alpha=L2,batch_size= len(train_data),verbose=True, \
				max_iter=300,tol=-1, hidden_layer_sizes=(100), random_state=1,  learning_rate='constant', learning_rate_init=0.001)
			clf.fit(train_data, train_lables)
		elif training_method == 'online':

			clf = MLPClassifier(solver='sgd', activation='logistic', alpha=0.1, verbose=True, batch_size = 1 ,\
				max_iter=30, warm_start=True,tol=-1, hidden_layer_sizes=(100), random_state=1,  learning_rate='constant', learning_rate_init=0.001)	
			clf.fit(train_data, train_lables)

		else:
			print ('error select eather online or batch')
			sys.exit()
		print ('clf created')
		
		train_score = clf.score(train_data, train_lables)
		validation_score = clf.score(validation_data, validation_lables)
		print ('Perceptron Train score:',train_score,'Validation score:',validation_score)

	def load_data(self, averaged = True):
		# when averaged = False it takes longer to generalize
		# while when averaged = True it takes less to generalised. futhermore it gives better results for less epocs
		# it then starts to overfit, for higher number of epocs not averaging it helps
		# therefore averaging is better.
		data = np.zeros((len(self.Categories)*500 - 50*len(self.Categories), self.Categories[0].vector_shape[0]))
		pred = np.zeros((50*len(self.Categories), self.Categories[0].vector_shape[0]))
		lables_train = np.zeros((len(self.Categories)*500 - 50*len(self.Categories)))
		lables_validation = np.zeros((50*len(self.Categories)))

		for i in range(len(self.Categories)):
			for j in range(len(self.Categories[i].train_vector)):
				data[j + i*450] = self.Categories[i].train_vector[j]
				lables_train[j + i*450] = self.Categories[i].label_id
			for j in range(len(self.Categories[i].validation_vector)):	
				pred[j + i*50] = self.Categories[i].validation_vector[j]
				lables_validation[j + i*50] = self.Categories[i].label_id
		if not averaged:
			return (data, lables_train, pred, lables_validation)
		else:

			mean_value = np.mean(data)
			variance_flat = data.reshape(data.shape[0]*self.Categories[0].vector_shape[0])
			variance_value = 0
			variance_value = np.std(data)
			for i in range(data.shape[0]):
				for j in range(self.Categories[0].vector_shape[0]):
					data[i,j] = (data[i,j] - mean_value)/variance_value

			for i in range(pred.shape[0]):
				for j in range(self.Categories[0].vector_shape[0]):
					pred[i,j] = (pred[i,j] - mean_value)/variance_value

			return (data, lables_train, pred, lables_validation)
class Categ:

	def __init__(self,folder):
		self.folder = folder
		self.Name = folder.split('/')[-1]
		self.images_name_list = os.listdir(folder)

		self.image_shape = 0
		self.vector_shape = 0

		self.train_vector = 0
		self.validation_vector = 0

		self.mean_vector = 0
		self.variance_vector = 0

	def label_set(self, label_id):
		self.label_id = label_id

	def initialise_image_size(self):
		img = Image.open(self.folder +'/'+ self.images_name_list[0]).convert('L')
		img_array = np.asarray(img)
		self.image_shape = img_array.shape
		vector = self.image_to_vector(img_array)
		image = self.vector_to_image(vector)
		self.vector_shape = vector.shape
		
	def train_validation_split(self):
		print (len(self.images_name_list), self.vector_shape[0])
		self.all_images = np.zeros((len(self.images_name_list), self.vector_shape[0]))
		self.train_vector = np.zeros((450, self.vector_shape[0]))
		self.validation_vector = np.zeros((50, self.vector_shape[0]))
		for i in range(len(self.images_name_list)):
			img = Image.open(self.folder +'/'+ self.images_name_list[i]).convert('L')
			img_array = np.asarray(img)
			vector = self.image_to_vector(img_array)
			if i < 450:
				self.all_images[i] = vector
				self.train_vector[i] = vector		
			if i >= 450:
				i = i - 450
				self.all_images[i] = vector
				self.validation_vector[i] = vector

	def image_to_vector(self, image):
		image_vector = np.reshape(image,image.size)
		return image_vector

	def vector_to_image(self, vector):
		img_array =vector.reshape((32, 32))
		im = Image.fromarray(img_array)
		return im

A = Computer_vision(['group_0/airplane', 'group_0/cat'])
A.unpack_folder()
A.Single_perceptron(False)
A.Single_perceptron(True) 
A.MLP('batch', False)
A.MLP('batch', False, L2 = 0.1)
A.MLP('online', False)

		

