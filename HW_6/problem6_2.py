import numpy as np
from pyoptwrapper import optimize
from pyoptsparse import NSGA2, SNOPT, ALPSO

def func(x):
	f = x[0]**2 + 2*x[1]**2 + 3*x[2]**2
	c1 = 6 - 2*x[0] - x[1] - 2*x[2] #less than or equal to zero
	c2 = 10 - 5*x[0] + x[1] + 3*x[2] #less than or equal to zero
	c = np.array([c1, c2])
	return f, c
	
def func_b(x):
	f = x[0]**2 + 2*x[1]**2 + 3*x[2]**2
	delta_c1 = 0.4
	delta_c2 = 0.75
	c1 = 6 - 2*x[0] - x[1] - 2*x[2] + delta_c1 #less than or equal to zero
	c2 = 10 - 5*x[0] + x[1] + 3*x[2] + delta_c2 #less than or equal to zero
	c = np.array([c1, c2])
	return f, c
	
if __name__ == '__main__':
	x1 = 0.
	x2 = 0.
	x3 = 0.
	lb = -1000000.
	ub = 1000000.
	x_orig = np.array([x1, x2, x3])
	optimizer = SNOPT()
	xopt, fopt, info = optimize(func, x_orig, lb, ub, optimizer)
	
	print "---------------Results Part (a)----------------------"
	print "Original X: ", x_orig
	print "Orig Func Val: ", func(x_orig)
	print "Optimal X: ", xopt
	print "Opt Func Val:, ", fopt
	
	xopt, fopt, info = optimize(func_b, x_orig, lb, ub, optimizer)
	
	print "---------------Results Part (b)----------------------"
	print "Original X: ", x_orig
	print "Orig Func Val: ", func(x_orig)
	print "Optimal X: ", xopt
	print "Opt Func Val:, ", fopt
	
	
