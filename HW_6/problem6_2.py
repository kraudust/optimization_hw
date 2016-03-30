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
    ##################### Part (a) #########################################
    xopt, fopt, info = optimize(func, x_orig, lb, ub, optimizer)

    print '\n', "---------------Results Part (a) Deterministic----------------------"
    print "Original X: ", x_orig
    print "Orig Func Val: ", func(x_orig)[0]
    print "Optimal X: ", xopt
    print "Opt Func Val: ", fopt, '\n'

    xopt_b, fopt_b, info = optimize(func_b, xopt, lb, ub, optimizer)

    ##################### Part (b) #########################################	
    print "-----------Results Part (b) Worst Case Tolerances------------------"
    print "Original X: ", xopt
    print "Orig Func Val: ", fopt
    print "Optimal X: ", xopt_b
    print "Opt Func Val: ", fopt_b, '\n'

    ##################### Part (c) #########################################
    #sig_x1 = 0.033
	
	
