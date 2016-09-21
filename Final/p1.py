#importing from cmath instead of math
from cmath import sin, cos, acos, exp, sqrt, pi
import numpy as np


def p1(x):
    a = 1.0/2*pi*sqrt(x[1])/x[0]
    b = x[2]*cos(x[4]) + x[3]*sin(x[4])
    c = x[3]*sin(x[4]) - x[2]*cos(x[4])
    d = x[0]*abs(sin(x[4]))
    e = pi*acos(exp(-d))
    f = a/b/e
    g = f/(x[1] + f)
    h = c/g - b/f
    return h

if __name__ == '__main__': #main code goes here
    x0 = np.array([0.5, 0.1, 0.1, 1.0, 0.2], dtype = complex)
    x = np.zeros(5, dtype = complex)
    x_FD = np.zeros(5)
    grad_h_x_CS = np.zeros(5)
    grad_h_x_FD = np.zeros(5)
    step_CS = 1.0e-20 # Step size for complex step
    step_FD = 1.0e-6 # Step size for finite difference
    #complex step
    for i in range(0, 5):
        x[:] = x0[:]
        x[i] = x[i] + complex(0., step_CS)
        grad_h_x_CS[i] = p1(x).imag/step_CS
    #finite difference
    for i in range(0, 5):
        x[:] = x0[:].real
        x[i] = x[i] + step_FD
        grad_h_x_FD[i] = (p1(x).real - p1(x0).real)/step_FD
    print "Gradient CS: ", grad_h_x_CS
    print "Gradient FD: ", grad_h_x_FD

#------------------------------Answers-------------------------------------
#1. Answer: [-0.34970161  0.63078121 -1.81285698  0.04619601  0.26464051]
#2. Description of the how and why of your approach:
#      I used a step size of 1.0e-20 because this will result in minimal
#      error. Complex step has an advantage over finite difference in that
#      it does not have subtractive cancellation, so the answers are
#      exact. To convert this problem to complex, I simply switched from
#      importing math to importing cmath (complex math), and made sure all
#      my variables were complex. I then used the complex step formula to
#      find the derivitives as shown in the code. I also calculated the
#      derivitives using finite difference to compare the complex step
#      derivitives to, and got comparable results.
#--------------------------------------------------------------------------
