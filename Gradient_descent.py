import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm


class RB:
	def __init__(self):
		self.w = [-0.50,-0.75]
		self.f = (1 - self.w[0])**2 + 100*(self.w[1] - self.w[0]**2)**2
		self.n = 0.001
		self.X = np.arange(-1.3, 1.3, 0.01)
		self.Y = np.arange(-1.3, 1.3, 0.01)
		self.X, self.Y = np.meshgrid(self.X, self.Y)
		self.Z = (1 - self.X)**2 + 100*(self.Y - self.X**2)**2
	def gradient_descend(self):
		iterations = 0
		tragectory_x = []
		tragectory_y = []
		tragectory_z = []
		print (self.w)
		while self.f > 0.001:
			w_old = self.w
			tragectory_x.append(self.w[0])
			tragectory_y.append(self.w[1])
			tragectory_z.append(self.f)
			dx = 2*(200*(self.w[0]**3) - 200*self.w[0]*self.w[1] + self.w[0] - 1)
			dy = 200*(self.w[1]- self.w[0]**2)
			self.w[0] = w_old[0] - self.n*dx
			self.w[1] = w_old[1] - self.n*dy
			iterations +=1
			self.f = (1 - self.w[0])**2 + 100*(self.w[1] - self.w[0]**2)**2

		tragectory_x.append(self.w[0])
		tragectory_y.append(self.w[1])
		tragectory_z.append(self.f)
		method = 'SD'
		self.plot_3d(tragectory_x,tragectory_y,tragectory_z, iterations, method)
		self.plot_contour(tragectory_x,tragectory_y,tragectory_z, iterations,method)
		self.plot_f(tragectory_z, iterations, method)


	def Newton(self):
		iterations = 0
		tragectory_x = []
		tragectory_y = []
		tragectory_z = []
		while self.f > 0.001:
			w_old = self.w
			print ('weights',self.w)
			tragectory_x.append(self.w[0])
			tragectory_y.append(self.w[1])
			tragectory_z.append(self.f)
			dx = 2*(200*(self.w[0]**3) - 200*self.w[0]*self.w[1] + self.w[0] - 1)
			dy = 200*(self.w[1]- self.w[0]**2)
			dxx = 1200*self.w[0]**2-400*self.w[1]+2
			dxy = -400*self.w[0]
			dyx = -400*self.w[0]
			dyy = 200
			gv = np.array([dx,dy])
			H = np.array([[dxx, dxy], [dyx, dyy]])
			Hmin1 = np.linalg.inv(H)
			Dw = np.matmul(gv,Hmin1)
			self.w[0] = w_old[0] - Hmin1[0,0]*gv[0] - Hmin1[0,1]*gv[1] 
			self.w[1] = w_old[1] - Hmin1[1,0]*gv[0] - Hmin1[1,1]*gv[1] 
			iterations +=1
			self.f = (1 - self.w[0])**2 + 100*(self.w[1] - self.w[0]**2)**2
		tragectory_x.append(self.w[0])
		tragectory_y.append(self.w[1])
		tragectory_z.append(self.f)
		method = 'Newton'
		self.plot_3d(tragectory_x,tragectory_y,tragectory_z, iterations, method)
		self.plot_contour(tragectory_x,tragectory_y,tragectory_z, iterations, method)
		self.plot_f(tragectory_z, iterations, method)
	def plot_3d(self,x,y,z, iterations, method):

		fig = plt.figure()
		ax = fig.add_subplot(111, projection='3d')
		surf = ax.plot_surface(self.X, self.Y, self.Z,linewidth=0,cmap=cm.coolwarm, antialiased=False)
		ax.text(x[0], y[0], z[0],'Start', color='white')
		ax.text(x[-1], y[-1], z[-1],'End', color='white')
		ax.plot(x, y, z, c='r')
		Title = 'Rosenbrock\'s Valley Problem 3D \n X: {0:.3f}, Y: {0:.3f}'.format(self.w[0], self.w[1]) + '\n Final f: {0:.7f}'.format(z[-1]) + '\n Iterations: {}, Method: {}'.format(iterations, method)
		plt.title(Title)
		ax.set_xlabel('X',fontsize=12)
		ax.set_ylabel('Y',fontsize=12)
		ax.set_zlabel('f(X,Y)',fontsize=12)

	def plot_contour(self,x,y,z, iterations, method):
		fig, ax = plt.subplots()
		CS = ax.contour(self.X, self.Y, self.Z,20,cmap=cm.coolwarm)
		ax.plot(x, y, c='r')
		ax.text(x[0], y[0],'Start', fontsize=12)
		ax.text(x[-1], y[-1], 'End',fontsize=12)
		Title = 'Rosenbrock\'s Valley Problem contour \n method: {}'.format(method)
		plt.title(Title)
		ax.set_xlabel('X',fontsize=12)
		ax.set_ylabel('Y',fontsize=12)
	def plot_f(self,f, iterations, method):
		fig, ax = plt.subplots()
		ax.plot([i for i in range(iterations + 1)],f, c='r')
		Title = 'f(X,Y) over iterations\n method: {}'.format(method)
		plt.title(Title)
		ax.set_xlabel('Iterations',fontsize=12)
		ax.set_ylabel('f(X,Y)',fontsize=12)


A = RB()

A.gradient_descend()
# A.Newton()

plt.show()