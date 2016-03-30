import numpy as np
import matplotlib.pyplot as plt

def fun(x):
	x1 = x[0]
	x2 = x[1]
	x3 = x[2]
	f = x1**2 + 2*x2**2 + 3*x3**2
	c = x1 + x2 + x3 - 3.5
	return f, c

if __name__ == '__main__':
	sigma_x = np.array([0.0, 0.06, 0.2])
	x_mean = np.array([1., 1., 1.])
	f_array = np.array([])
	c_array = np.array([])
	n = 10**4
	for i in range(n):
		x = sigma_x * np.random.randn(3) + x_mean
		f = fun(x)[0]
		f_array = np.append(f_array, f)
		c = fun(x)[1]
		c_array = np.append(c_array, c)
	f_mean = np.mean(f_array)
	print "f mean", f_mean
	sigma_f = np.std(f_array)
	print "f std", sigma_f
	plt.figure(1)
	plt.hist(f_array, bins = 25)
	plt.title('Function Value')
	plt.figure(2)
	plt.hist(c_array, bins = 25)
	plt.title('Constraint Value')
	plt.show()
		
		
	

