import numpy as np
from pyoptwrapper import optimize
from pyoptsparse import SNOPT, NSGA2

def p3(x):

    f = -(x[1] + 47)*np.sin(np.sqrt(np.abs(x[0]/2.0 + x[1] + 47))) - x[0]*np.sin(np.sqrt(np.abs(x[0] - x[1] - 47))) + (x[0]/100 - x[1]/100 + x[2]/100)**2
    return f/10.0, []

if __name__ == '__main__':
    x0 = np.array([0.0, 0.0, 0.0])
    #x0 = np.array([250.0, 250.0, 250.0])
    #x0 = np.array([-250.0, -250.0, -250.0])
    #x0 = np.array([500.0, -500.0, 0.0])
    f0 = p3(x0)[0]
    lb = np.array([-512.0, -512.0, -512.0])
    ub = np.array([512.0, 512.0, 512.0])

    #Run Optimization
    optimizer = NSGA2()
    xopt, fopt, info = optimize(p3, x0, lb, ub, optimizer)

    print "x0: ", x0
    print "f0: ", f0*10
    print "xopt: ", xopt
    print "fopt: ", fopt*10

#------------------------------Answers-----------------------------------------
#1. Answer: xopt = [512.0    404.2318    -107.7733]
#           fopt = -959.641
#2. Description of the how and why of your approach:
#       First I scaled the object function by 10 so that it was of order 1.
#       Then I ran the optimizer with SNOPT using different starting points,
#       and I found that the answer I got was very dependent on my starting
#       points, meaning that there were likely multiple local minimum. Also,
#       the objective function has an absolute value in it which makes for
#       discontinuous derivatives. So, I ended up using a non-gradient based
#       optimizer NSGA2. Using this I got much better results than any of
#       my starting values using the gradient based optimzer.
#------------------------------------------------------------------------------
