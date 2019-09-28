import numpy as np
import matplotlib.pyplot as plt
import model

def plot(trainList, testList, parameters, classPriories):
	#trainList: list of training data of different classes
	fig = plt.figure()

	xmax = -1e9
	xmin = 1e9
	ymax = -1e9
	ymin = 1e9

	for val in trainList:
		xmax = max(xmax, val[:,0].max())
		xmin = min(xmin, val[:,0].min())
		ymax = max(ymax, val[:,1].max())
		ymin = min(ymin, val[:,1].min())

	for test in testList:
		plt.scatter(test[:,0],test[:,1])

		#Length of each step for colouring purposes.
	hx = (xmax-xmin)/200
	hy = (ymax-ymin)/200

		#Dimensions of plotting area
	x_max, x_min = xmax+10*hx, xmin-10*hx
	y_max, y_min = ymax+10*hy, ymin-10*hy

		#Gives 2d arrays with X values and Y values at given poitn respectively
	xx, yy = np.meshgrid(np.arange(x_min, x_max, hx), np.arange(y_min, y_max, hy))

	Z = np.array([model.predict(x, parameters, classPriories) for x in np.c_[xx.ravel(), yy.ravel()]])
	Z = Z.reshape(xx.shape)
	plt.contourf(xx, yy, Z, alpha=0.25)
	plt.contour(xx, yy, Z, colors='k', linewidths=0.5)
	plt.show()