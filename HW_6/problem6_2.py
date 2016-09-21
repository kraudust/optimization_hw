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

def stats_func(n, function, x_opt):
	f = np.zeros(n)
	sig_x1 = 0.1/3.
	sig_x2 = 0.1/3.
	sig_x3 = 0.05/3.
	sigma = np.array([sig_x1, sig_x2, sig_x3])
	counter = 0
	for i in range(n):
		x1 = x_opt[0] + np.random.randn(1)*sigma[0]
		x2 = x_opt[1] + np.random.randn(1)*sigma[1]
		x3 = x_opt[2] + np.random.randn(1)*sigma[2]
		x = np.array([x1, x2, x3])

		f[i] = function(x)[0]
		c = function(x)[1]
		if np.all(c <= 0.):
			counter += 1
	# mean
	mu = np.average(f)

	# standard deviation
	std = np.std(f, ddof=1)  #ddof=1 gives an unbiased estimate (np.sqrt(1.0/(n-1)*(np.sum(f**2) - n*mu**2)))
	reliability = counter/float(n)
	return mu, std, f, c, reliability

if __name__ == '__main__':
	x1 = 6.
	x2 = 2.
	x3 = 7.
	lb = -100.
	ub = 100.
	x_orig = np.array([x1, x2, x3])
	f_orig = func(x_orig)[0]
	optimizer = SNOPT()
	##################### Part (a) #########################################
	xopt_a, fopt, info = optimize(func, x_orig, lb, ub, optimizer)
	print '\n', "--------------Results Part (a) Deterministic-----------------"
	print "Original X: ", x_orig
	print "Orig Func Val: ", f_orig
	print "Optimal X: ", xopt_a
	print "Opt Func Val: ", fopt, '\n'

	##################### Part (b) #########################################
	xopt_b, fopt, info = optimize(func_b, x_orig, lb, ub, optimizer)
	print "-----------Results Part (b) Worst Case Tolerances------------------"
	print "Original X: ", x_orig
	print "Orig Func Val: ", f_orig
	print "Optimal X: ", xopt_b
	print "Opt Func Val: ", fopt, '\n'

	##################### Part (c) #########################################
	xopt_c, fopt, info = optimize(func_c, x_orig, lb, ub, optimizer)
	print "-----------Results Part (c) Transmitted Variance------------------"
	print "Original X: ", x_orig
	print "Orig Func Val: ", f_orig
	print "Optimal X: ", xopt_c
	print "Opt Func Val: ", fopt, '\n'

	##################### Part (d) #########################################
	n = 1000000
	print "Reliability(a): ", stats_func(n, func, xopt_a)[4]
	print "Reliability(b): ", stats_func(n, func, xopt_b)[4]
	print "Reliability(c): ", stats_func(n, func, xopt_c)[4]
