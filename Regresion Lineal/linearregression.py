import numpy as np 
import matplotlib.pyplot as plt


def estimate_m_b(x, y):
	n = np.size(x)
    #Average x and y 
	mx, my = np.mean(x), np.mean(y)
    #Sumatory E
	sumatory_xy = np.sum((x-mx)*(y-my))
	sumatory_xx = np.sum(x*(x-mx))
	#Regression coef
	m = sumatory_xy/sumatory_xx
	b = my - m*mx

	return b, m


def graph_regression(x, y, b):
	plt.scatter(x, y, color='g', marker='o', s=30) # disperssion graph
	y_pred = b[0] + b[1]*x #regression ecu
	plt.plot(x, y_pred, color='r')	
	plt.xlabel('x values')
	plt.ylabel('y values')
	plt.show()


def run():
	x = np.array([1,2,3,4,5])
	y = np.array([2,3,5,6,5])

	b = estimate_m_b(x, y)
	print(f'B Values: {b[0]} \n\rM Values: {b[1]}')
	graph_regression(x, y, b)


if __name__ == '__main__':
	run()
