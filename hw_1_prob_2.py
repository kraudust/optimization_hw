import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True}) # makes it so that the plot windows don't cut off the labels

def f_of_x(x, gamma):
    return -x + gamma*x**2
def g_of_x(x, gamma):
    return -1 + 2.0*gamma*x
def golden_section_search(f, a, b, tol, args = [], i = 0,):
    """ This function returns the minimum of a function using the golden section search method
    :param f: function handle
    :param a: lower bound of interval containing minimum
    :param b: upper bound of interval containing minimum
    :param tol: tolerance on final solution
    :param i: counter to keep track of function calls
    :param args: if f is a function of more than 1 variable, put the variables here
    :return: input value that yields the minimum of the function, and iterations
    """
    tau = (5**0.5 - 1)/2  # inverse of golden ratio
    c = b-tau*(b-a)
    d = a + tau*(b-a)
    if not args:
        if np.sqrt(np.sum((c - d)**2)) <= tol:
            return (a + b)/2, i
        if f(d) < f(c):
            a = c
        else:
            b = d
        i = i + 1
        return golden_section_search(f, a, b, tol, args, i)
    else:
        if np.sqrt(np.sum((c - d)**2)) <= tol:
            return (a + b)/2, i
        if f(d, args) < f(c, args):
            a = c
        else:
            b = d
        i = i + 1
        return golden_section_search(f, a, b, tol, args, i)
def simple_back_line(x, p, alpha, rho, mu, f, g, args_f = [], args_g = []):
    """ This function performs a simple backtracking line search
    :param x: initial starting point
    :param p: direction gradient is to be taken in
    :param alpha: initial step length
    :param rho: reduction parameter
    :param mu: sufficient decrease parameter
    :param f: function handle
    :param g: directional gradient handle
    :param args_f: if f is a function of more than 1 variable, put the variables here
    :param args_g: if g is a function of more than 1 variable, put the variables here
    :return:
    """
    i = 1
    if not args_f and not args_g: #if f and g are not function of any additional arguments
        while f(x + alpha*p) > (f(x) + mu*alpha*np.dot(g(x),p)):
            alpha = rho*alpha
            i = i + 1
        return alpha, i
    if args_f and not args_g: #if g is not a function of any additional arguments but f is
        while f((x + alpha*p), args_f) > (f(x, args_f) + mu*alpha*np.dot(g(x),p)):
            alpha = rho*alpha
            i = i + 1
        return alpha, i
    if not args_f and args_g: #if f is not a function of any additional arguments but g is
        while f(x + alpha*p) > (f(x) + mu*alpha*np.dot(g(x,args_g),p)):
            alpha = rho*alpha
            i = i + 1
        return alpha, i
    if args_f and args_g: #if both f and g are functions of additional arguments
        while f((x + alpha*p),args_f) > (f(x, args_f) + mu*alpha*g(x, args_g)):
            alpha = rho*alpha
            i = i + 1
        return alpha, i

#################################################################################################################
if __name__ == '__main__': # this keeps the whole script from running when I import it's functions elsewhere
    gamma = [0.5, 10.0, 10.0**4]
    z = raw_input('For problem 2 part a type a, for part b type b')
    if z == "a":
        for j in range(0,3):
            minimum = golden_section_search(f_of_x, -1, 1.5, 10**-6, gamma[j])
            print minimum[0], f_of_x(minimum[0],gamma[j])
            print 'For gammma = %g, the x value that minimizes the function is: %f in %d iterations' % \
              (gamma[j],minimum[0], minimum[1])
            x = np.linspace(-10,10,num = 100)
            y = f_of_x(x,gamma[j])
            plt.figure(j)
            plt.plot(x,y)
            plt.plot(minimum[0],f_of_x(minimum[0],gamma[j]),'ro')
            plt.xlabel('x')
            plt.ylabel('f(x)')
            plt.title('Gamma = %g' % (gamma[j]))
            p = j
            plt.savefig('Pictures/HW_1/hw_1_prob_2 figure %d' %(p)+'.pdf', format = 'pdf')
        plt.show()
    if z == "b":
        x_0 = 0
        alpha_0 = 1
        rho = 0.5
        mu = 10e-6
        for j in range(0,3):
            print simple_back_line(x_0, 1, alpha_0, rho, mu, f_of_x, g_of_x, gamma[j],gamma[j])
