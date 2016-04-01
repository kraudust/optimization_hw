import numpy as np
from pyoptwrapper import optimize
from pyoptsparse import NSGA2, SNOPT, ALPSO
from scipy.stats import norm

def func(x):
	f = x[0]**2. + 2*x[1]**2. + 3.*x[2]**2.
	c1 = 6. - 2.*x[0] - x[1] - 2.*x[2] #less than or equal to zero
	c2 = 10. - 5.*x[0] + x[1] + 3.*x[2] #less than or equal to zero
	c = np.array([c1, c2])
	return f, c

def func_b(x):
	f = x[0]**2. + 2.*x[1]**2. + 3.*x[2]**2.
	delta_c1 = 0.4
	delta_c2 = 0.75
	c1 = 6. - 2.*x[0] - x[1] - 2.*x[2] + delta_c1 #less than or equal to zero
	c2 = 10. - 5.*x[0] + x[1] + 3.*x[2] + delta_c2 #less than or equal to zero
	c = np.array([c1, c2])
	return f, c

def func_c(x):
	f = x[0]**2. + 2*x[1]**2. + 3.*x[2]**2.
	sig_x1 = 0.1/3.
	sig_x2 = 0.1/3.
	sig_x3 = 0.05/3.
	sig_c1 = np.sqrt((-2.*sig_x1)**2. + (-1.*sig_x2)**2. + (-2.*sig_x3)**2.)
	sig_c2 = np.sqrt((-5.*sig_x1)**2. + (1.*sig_x2)**2. + (3.*sig_x3)**2.)
	k = norm.ppf(0.99865) #i.e. 99.865% reliability for each constraint indiv.
	c1 = 6. - 2.*x[0] - x[1] - 2.*x[2] + k*sig_c1 #less than or equal to zero
	c2 = 10. - 5.*x[0] + x[1] + 3.*x[2] + k*sig_c2 #less than or equal to zero
	c = np.array([c1, c2])
	return f, c

if __name__ == '__main__':
	x1 = 1.
	x2 = 2.
	x3 = 3.
	lb = -1000000.
	ub = 1000000.
	x_orig = np.array([x1, x2, x3])
	f_orig = func(x_orig)[0]
	optimizer = SNOPT()
	##################### Part (a) #########################################
	xopt, fopt, info = optimize(func, x_orig, lb, ub, optimizer)
	print '\n', "--------------Results Part (a) Deterministic-----------------"
	print "Original X: ", x_orig
	print "Orig Func Val: ", f_orig
	print "Optimal X: ", xopt
	print "Opt Func Val: ", fopt, '\n'

	##################### Part (b) #########################################
	xopt, fopt, info = optimize(func_b, x_orig, lb, ub, optimizer)
	print "-----------Results Part (b) Worst Case Tolerances------------------"
	print "Original X: ", x_orig
	print "Orig Func Val: ", f_orig
	print "Optimal X: ", xopt
	print "Opt Func Val: ", fopt, '\n'

	##################### Part (c) #########################################
	xopt, fopt, info = optimize(func_c, x_orig, lb, ub, optimizer)
	print "-----------Results Part (c) Transmitted Variance------------------"
	print "Original X: ", x_orig
	print "Orig Func Val: ", f_orig
	print "Optimal X: ", xopt
	print "Opt Func Val: ", fopt, '\n'
