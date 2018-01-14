import os
import numpy as np 
from matplotlib.lines import Line2D
# import sklearn.datasets
import matplotlib.pyplot as plt
from neuralnet import NeuralNet
import imageio

def createGif(images):
	imageio.mimsave('1-ddata-tranformation.gif', images, duration=0.1)


def generateOneDData():
	np.random.seed(3)
	A = np.random.uniform(low=-1/3, high=1/3, size=(500,1))
	Ya = np.zeros((500,1)) # Class A label [0]
	b1 = np.random.uniform(low=-1, high=-2/3, size=(250,1))
	b2 = np.random.uniform(low=2/3, high=1, size=(250,1))
	B = np.vstack((b1,b2))
	Yb = np.ones((500,1)) # Class B label [1]
	X = np.vstack((A,B)).T
	Y = np.vstack((Ya, Yb)).T

	return X, Y

def plotOneDData(X, Y):
	A = X[Y==0]
	B = X[Y==1]
	plt.scatter(A, np.zeros_like(A), c='r', label="Class A - [-1/3, 1/3]")
	plt.scatter(B, np.zeros_like(B), c='b', label="Class B - [-1, -2/3] U [2/3, 1]")
	plt.legend()
	plt.xlabel('Feature Vector')
	plt.title('One Dimensional data')
	plt.savefig(os.path.join('Plots','One-Dimensional-data.png'))
	plt.show(block=False)

def plotTransformedData(A, W, b,Y):
	plt.ion()
	# images = []
	for count,(w,b, data) in enumerate(zip(W,b, A)):
		point1 = [-1 , (-b+w[0,0])/w[0,1]]
		point2 = [(-b-w[0,1])/w[0,0], 1]
		line = Line2D(point1, point2, c=(0,0,0) , label='Decision Boundary')
		ax = plt.axes()
		ax.add_line(line)
		X1 = data[0,:].reshape(1,-1)
		X2 = data[1,:].reshape(1,-1)
		plt.scatter(X1[Y==0],X2[Y==0], s=3, c='r', label='Class A')
		plt.scatter(X1[Y==1],X2[Y==1], s=3, c='b', label='Class B')
		plt.title('One Dimensional data', fontweight='bold')
		plt.xlabel('Output from first hidden unit')
		plt.ylabel('Output from second hidden unit')
		plt.legend()
		# plt.savefig(os.path.join('Plots','forgif',str(count)+'.png'))
		# images.append(imageio.imread(os.path.join('Plots','forgif',str(count)+'.png')))
		plt.pause(0.1)
		plt.clf()

	# createGif(images)

def main():
	X, Y = generateOneDData()
	# plotOneDData(X, Y)
	noOfLayers = 2 # Hidden and Output layer (Excluding the input layer)
	layerDimensions = [1, 2, 1] # No of units in Input, Hidden, Output layer
	noOfIterations = 5000
	learningRate = 0.6
	N = NeuralNet(noOfLayers, layerDimensions) # Create a object of Neural Net
	AL, WL, bL = N.gradientDescent(X, noOfIterations, learningRate, Y, printCost=True)
	plotTransformedData(AL, WL, bL, Y)


if __name__ == '__main__':
	main()
