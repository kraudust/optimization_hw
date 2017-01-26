import numpy as np
from math import sqrt
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import hw_1_prob_2

def simple_back_line(x, p, alpha, rho, mu, f, f_of_x_k, g_of_x_k, args_f = []):
    """ This function performs a simple backtracking line search
    :param x: initial starting point
    :param p: direction gradient is to be taken in
    :param alpha: initial step length
    :param rho: reduction parameter
    :param mu: sufficient decrease parameter
    :param f: function handle in the form f, g = f(x)
    :param f_of_x_k: function value at current x
    :param g_of_x_k: gradient value at current x
    :param args_f: if f is a function of more than the design variables, put the variables here
    :return:
    """
    i = 1
    if not args_f: #if f and g are not function of any additional arguments
        while f(x + alpha*p)[0] > (f_of_x_k + mu*alpha*np.dot(g_of_x_k,p)):
            alpha = rho*alpha
        return alpha
    if args_f: #if g is not a function of any additional arguments but f is
        while f((x + alpha*p), args_f)[0] > (f_of_x_k + mu*alpha*np.dot(g_of_x_k,p)):
            alpha = rho*alpha
        return alpha
def make_contour(x1, x2, y1, y2, func, x_label, y_label, title):
    # --- setup grid for contour plot ---
    nx = 200  # number of points in x-direction
    ny = 200  # number of points in y-direction
    x = np.linspace(x1, x2, nx)  # nx points equally spaced between x1, x2
    y = np.linspace(y1, y2, ny)  # ny points equally spaced between y1, y2
    X, Y = np.meshgrid(x, y, indexing='ij')  # 2D array (matrix) of points across x and y
    Z = np.zeros((nx, ny))  # initialize output of size (nx, ny)
    # --- evaluate across grid ---
    for i in range(nx):
        for j in range(ny):
            t = (X[i, j], Y[i, j])
            Z[i, j] = func(t)[0]
    plt.figure()  # start a new figure
    plt.contour(X, Y, Z, 100)  # using 100 contour lines.
    plt.colorbar()  # add a color bar
    plt.xlabel(x_label)  # labels for axes
    plt.ylabel(y_label)
    plt.title(title)
def matyas_fun(x):
    """
    :param x: parameters for Matyas function in order [x1, x2]
    :return: the value of the Matyas function (f) and its gradient (g)
    """
    global matyas_func_calls
    x1 = x[0]
    x2 = x[1]
    f = 0.26*(x1**2 + x2**2) - 0.48*x1*x2
    g1 = 0.52*x1 - 0.48*x2
    g2 = 0.52*x2 - 0.48*x1
    g = np.array([g1, g2])
    matyas_func_calls = matyas_func_calls + 1
    return f, g
def rosenbrock_fun(x):
    """
    :param x: parameters for Matyas function in order [x1, x2]
    :param x2: parameter of Rosenbrock function
    :return: value of the Rosenbrock function (f) and its gradient (g)
    """
    global rosen_func_calls
    x1 = x[0]
    x2 = x[1]
    f = (1 - x1)**2 + 100.0*(x2 - x1**2)**2
    g1 = 400*x1**3 - 400*x1*x2 + 2*x1 -2
    g2 = -200*x1**2 + 200*x2
    g = np.array([g1,g2])
    rosen_func_calls += 1
    return f, g
def brachistochrone(yint):
    """brachistochrone problem.

    Parameters
    ----------
    yint : a vector of y location for all the interior points

    Outputs
    -------
    J : scalar proportion to the total time it takes the bead to traverse
        the wire
    g : dJ/dyint the derivatives of J w.r.t. each yint.
    """

    # fill in details
    global brach_func_calls
    y = np.insert(yint, 0, 1) #add the initial value of H = 1
    y = np.append(y, 0.) #add the final value of H = 0
    x = np.linspace(0, 1, y.size)
    H = 1.0
    mu_k = 0.3
    J = 0
    for i in range(0, y.size-1):
        num = sqrt((x[i+1]-x[i])**2 + (y[i+1]-y[i])**2)
        den = sqrt(H - y[i+1] - mu_k*x[i+1]) + sqrt(H - y[i] - mu_k*x[i])
        J_temp = num/den
        J = J + J_temp
    g = grad_brach(x, y, mu_k, H)  # note y is not the same as yint.  y should include the end points
    brach_func_calls = brach_func_calls + 1
    return J, g
def grad_brach(x, y, mu_k, H):
    """gradients of the brachistochrone function.  This function accepts
    as input the full x, and y vectors, but returns gradients only for the
    interior points.

    Parameters
    ----------
    x : array of length n
        an array of x locations including the end points
    y : array of length n
        corresponding heights including the end points
    mu_k : float
        coefficient of kinetic friction
    H : float
        initial height of bead

    Outputs
    -------
    g : array of length n-2
        dJ/dy for all interior points.  Note that the end points are
        fixed and thus are not design variables and so there gradients
        are not included.

    """

    n = len(x)
    g = np.zeros(n-2)

    for i in range(n-1):

        ds = sqrt((x[i+1] - x[i])**2 + (y[i+1] - y[i])**2)
        vbar = sqrt(H - y[i+1] - mu_k*x[i+1]) + sqrt(H - y[i] - mu_k*x[i])

        if i > 0:
            dsdyi = -(y[i+1] - y[i])/ds
            dvdyi = -0.5/sqrt(H - y[i] - mu_k*x[i])
            dtdyi = (vbar*dsdyi - ds*dvdyi)/(vbar**2)
            g[i-1] += dtdyi
        if i < n-2:
            dsdyip = (y[i+1] - y[i])/ds
            dvdyip = -0.5/sqrt(H - y[i+1] - mu_k*x[i+1])
            dtdyip = (vbar*dsdyip - ds*dvdyip)/(vbar**2)
            g[i] += dtdyip

    return g
def uncon(func, x0, epsilon_g, options=None):
    """An algorithm for unconstrained optimization.
    Parameters
    ----------
    func : function handle
        function handle to a function of the form: f, g = func(x)
        where f is the function value and g is a numpy array containing
        the gradient. x are design variables only.
    x0 : ndarray
        starting point
    epsilon_g : float
        convergence tolerance.  you should terminate when
        np.max(np.abs(g)) <= epsilon_g.  (the infinity norm of the gradient)
    options : dict
        a dictionary containing options.  You can use this to try out different
        algorithm choices.  I will not pass anything in, so if the input is None
        you should setup some defaults.

    Outputs
    -------
    xopt : ndarray
        the optimal solution
    fopt : float
        the corresponding function value
    outputs : list
        a list  of numpy arrays containing: [function values, convergence criteria, iteration count, x value]
    """

    if options is None:
        iter = 0
        iterations = [iter]
        f_0, g_0 = func(x0)
        func_val = [f_0]
        conv_crit = [max(np.abs(g_0))]
        x_val = [x0]
        I = np.identity(x0.size)
        V_0 = I #initial hessian inverse of size n where n = num design variables
        p_0 = np.dot(-V_0, g_0) #performs matrix multiplication instead of element wise multiplication
        alpha = 1.
        rho = 0.25 #parameter for simple backtracking line search
        mu = 1e-4 #parameter for simple backtracking line search
        alpha = simple_back_line(x0,p_0,alpha,rho,mu,func,f_0,g_0,args_f = [])
        #x_k = hw_1_prob_2.golden_section_search(func,x0,x0+2*p_0,0.01)[0]
        x_k = x0 + alpha*p_0
        s_k = x_k - x0 # same thing as alpha*p
        f_k, g_k = func(x_k)
        y_k = g_k - g_0
        iter = iter + 1
        iterations.append(iter)
        conv_crit.append(max(np.abs(g_k)))
        func_val.append(f_k)
        x_val.append(x_k)
        denom = np.dot(s_k, y_k) #denominator in V_k+1 equation
        num_1 = np.outer(s_k,y_k) #numerator in first term of V_k+1 equation
        num_2 = np.outer(y_k,s_k) #numerator in second term of V_k+1 equation
        num_3 = np.outer(s_k,s_k) #numerator in third term of V_k+1 equation
        V_k = np.dot(np.dot((I-num_1/denom),V_0),(I-num_2/denom)) + num_3/denom
        while np.max(np.fabs(g_k)) > epsilon_g:
            p_k = np.dot(-V_k, g_k)
            alpha = 1. #initially, alpha is 1 i.e. step to minimum of local quadratic approximation
            #if minimum of local quadratic doesn't satisfy sufficient decrease, calculate new alpha
            alpha = simple_back_line(x_k, p_k, alpha, rho, mu, func,f_k,g_k, args_f = [])
            x_k_pl_1 = x_k + alpha*p_k
            #x_k_pl_1 = hw_1_prob_2.golden_section_search(func,x_k,x_k+2*p_k,0.01)[0]
            s_k = x_k_pl_1 - x_k
            f_k_pl_1, g_k_pl_1 = func(x_k_pl_1)
            y_k = g_k_pl_1 - g_k
            iter = iter + 1
            denom = np.dot(s_k, y_k) #denominator in V_k+1 equation sometimes goes to zero with poorly chosen x0
            num_1 = np.outer(s_k,y_k) #numerator in first term of V_k+1 equation
            num_2 = np.outer(y_k,s_k) #numerator in second term of V_k+1 equation
            num_3 = np.outer(s_k,s_k) #numerator in third term of V_k+1 equation
            V_k = np.dot(np.dot((I-num_1/denom),V_k),(I-num_2/denom)) + num_3/denom
            x_k = x_k_pl_1
            g_k = g_k_pl_1
            f_k = f_k_pl_1
            conv_crit.append(max(np.abs(g_k)))
            iterations.append(iter)
            func_val.append(f_k)
            x_val.append(x_k)
        conv_crit = np.asarray(conv_crit) #convert convergence criteria to a numpy array
        iterations = np.asarray(iterations) #convert iterations list to a numpy array
        func_val = np.asarray(func_val) #convert function values list to a numpy array
        x_val = np.asarray(x_val)
        outputs = [func_val, conv_crit, iterations, x_val]
        xopt = x_k
        fopt = f_k
        return xopt, fopt, outputs
if __name__ == '__main__': #main code goes here
    #variables to keep track of function calls
    brach_func_calls = 0
    rosen_func_calls = 0
    matyas_func_calls = 0

    epsilon_g = 1e-6 #convergence criteria
    #----------------------------------------Rosenbrock Function-------------------------------------------------
    print "---------------------------------------------Rosenbrock-----------------------------------------------"
    x0 = np.array([-1.5, 0.]) #initial point

    #run my optimizer
    opt = uncon(rosenbrock_fun,x0,epsilon_g)
    print "My optimizer:"
    print "         Minimum xf: ", opt[0]
    print "         f(xf): ", opt[1]
    print "         Norm g(xf): ", opt[2][1][np.size(opt[2][2])-1]
    print "         Major Iterations: ", opt[2][2][np.size(opt[2][2])-1]
    print "         Function Calls: ", rosen_func_calls

    #run the built in optimizer
    options = {'disp': True}
    res = minimize(rosenbrock_fun, x0, method='BFGS', jac=True, tol=epsilon_g, options=options)
    print "         Minimum xf:", res.x

    #plot for convergence metric and major iterations for my optimizer
    plt.figure()
    plt.plot(opt[2][2],opt[2][1], '-o')
    plt.yscale('log')
    plt.xlabel('Major Iterations')
    plt.ylabel('Infinity norm of gradient')
    plt.title('Rosenbrock Function with $x_0 = (-1.5,0)$')
    plt.savefig('Pictures/HW_2/conv_metric_vs_iterations_Rosenbrock.pdf', format = 'pdf')

    #Make contour plot and plot iterations on it for my optimizer
    make_contour(-2,2,-1,3,rosenbrock_fun,"x1", "x2", "Rosenbrock Function")
    x_val = opt[2][3]
    x1 = np.zeros(np.shape(x_val)[0])
    x2 = np.zeros(np.shape(x_val)[0])
    for i in range(0,np.shape(x_val)[0]):
        x1[i] = x_val[i][0]
        x2[i] = x_val[i][1]
    plt.plot(x1, x2, 'k-o',label = 'Iteration History')
    plt.legend()
    plt.savefig('Pictures/HW_2/iterations_contour_Rosenbrock.pdf', format = 'pdf')

    #----------------------------------------------Matyas Function--------------------------------------------------
    print "--------------------------------------------Matyas--------------------------------------------------"
    x0 = np.array([9., 7.]) # initial point

    #run my optimizer
    opt = uncon(matyas_fun, x0,epsilon_g)
    print "My optimizer:"
    print "         Minimum xf: ", opt[0]
    print "         f(xf): ", opt[1]
    print "         Norm g(xf): ", opt[2][1][np.size(opt[2][2])-1]
    print "         Major Iterations: ", opt[2][2][np.size(opt[2][2])-1]
    print "         Function Calls: ", matyas_func_calls

    #run the built in optimizer
    options = {'disp': True}
    res = minimize(matyas_fun, x0, method='BFGS', jac=True, tol=epsilon_g, options=options)
    print "         Minimum xf:", res.x

    #plot for convergence metric and major iterations for my optimizer
    plt.figure()
    plt.plot(opt[2][2],opt[2][1], '-o')
    plt.yscale('log')
    plt.xlabel('Major Iterations')
    plt.ylabel('Infinity norm of gradient')
    plt.title('Matyas Function with $x_0 = (9,7)$')
    plt.savefig('Pictures/HW_2/conv_metric_vs_iterations_Matyas.pdf', format = 'pdf')

    #Make contour plot and plot iterations on it for my optimizer
    make_contour(-10,10,-10,10, matyas_fun, "x1", "x2", "Matyas Function")
    x_val = opt[2][3]
    x1 = np.zeros(np.shape(x_val)[0])
    x2 = np.zeros(np.shape(x_val)[0])
    for i in range(0,np.shape(x_val)[0]):
        x1[i] = x_val[i][0]
        x2[i] = x_val[i][1]
    plt.plot(x1, x2, 'k-o', label = 'Iteration History')
    plt.legend()
    plt.savefig('Pictures/HW_2/iterations_contour_Matyas.pdf', format = 'pdf')

    #-------------------------------------------Brachistochrone Function--------------------------------------------
    print "-----------------------------------------Brachistochrone------------------------------------------------"
    n = 32. #number of points to put on brachistochrone function
    del_y = 1./(n+1)
    yint = np.linspace(1-del_y, 0+del_y, n) #seed the function initially with a line

    #run my optimizer
    opt = uncon(brachistochrone,yint,epsilon_g)
    iteration_count = np.size(opt[2][2]) - 1
    print "My optimizer:"
    print "         Minimum xf: ", opt[0]
    print "         f(xf): ", opt[1]
    print "         Norm g(xf): ", opt[2][1][np.size(opt[2][2])-1]
    print "         Major Iterations: ", iteration_count
    print "         Function Calls: ", brach_func_calls

    #run the built in optimizer
    options = {'disp': True}
    res = minimize(brachistochrone, yint, method='BFGS', jac=True, tol=epsilon_g, options=options)
    print "         Minimum xf:", res.x

    #plot for convergence metric and major iterations
    plt.figure()
    plt.plot(opt[2][2],opt[2][1], '-o')
    plt.yscale('log')
    plt.xlabel('Major Iterations')
    plt.ylabel('Infinity norm of gradient')
    plt.title('Brachistochrone Function with n = 32')
    plt.savefig('Pictures/HW_2/conv_metric_vs_iterations_Brachistochrone.pdf', format = 'pdf')

    #plot a few configurations of the track every n iterations
    n = 4
    y_int_values = opt[2][3]
    y_values_0 = y_int_values[0]
    y_values_0 = np.append(y_values_0,0)
    y_values_0 = np.insert(y_values_0,0,1)
    num = (iteration_count)/n #number of iterations between plots
    x = np.linspace(0, 1, y_values_0.size)
    plt.figure()
    for i in range(0,iteration_count,num):
        y_value = y_int_values[i]
        y_value = np.append(y_value,0) #insert the final point
        y_value = np.insert(y_value,0,1) #insert the first point
        plt.plot(x,y_value,'-o',label = "iteration %d" %(i))
    #plot final point
    y_value_f = opt[0]
    y_value_f = np.append(y_value_f,0) #insert the final point
    y_value_f = np.insert(y_value_f,0,1) #insert the first point
    plt.plot(x,y_value_f,'-o',label = "iteration %d" %(iteration_count))
    plt.legend()
    plt.title('Brachistochrone Function')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.savefig('Pictures/HW_2/iterations_history_Brachistochrone.pdf', format = 'pdf')

    #Explore effect of increased problem dimensionality.... with plots... and linear initial seeding
    n = [4, 8, 16, 32, 64, 128, 256, 512] #number of points to put on brachistochrone function
    #n = [4, 8, 16]
    iterations = []
    func_calls = []
    for i in range(0,len(n)):
        brach_func_calls = 0
        del_y = 1./(n[i] + 1)
        yint = np.linspace(1-del_y, 0+del_y, n[i]) #seed the function initially with a line
        opt = uncon(brachistochrone,yint,epsilon_g)
        iteration_count = np.size(opt[2][2]) - 1
        iterations.append(iteration_count)
        func_calls.append(brach_func_calls)
    plt.figure()
    plt.plot(n,func_calls,'bo-',label = 'Function Calls')
    plt.plot(n,iterations,'ro-', label = 'Major Iterations')
    plt.legend(loc = 'upper left')
    plt.xlabel('Dimensionality (n)')
    plt.ylabel('# of function calls/# of major iterations')
    plt.title('Brachistrochrone Increased Dimensionality Plot with Linear Initial Seeding')
    plt.savefig('Pictures/HW_2/increased_dimensionality_Brachistochrone.pdf', format = 'pdf')

    #Explore effect of increased problem dimensionality.... with plots... and seeding from previous n

    #Explore effect of increased problem dimensionality.... with plots... and linear initial seeding
    n = [4, 8, 16] #number of points to put on brachistochrone function
    iterations = []
    func_calls = []
    for i in range(0,len(n)):
        brach_func_calls = 0
        del_y = 1./(n[i] + 1)
        yint = np.linspace(1-del_y, 0+del_y, n[i]) #seed the function initially with a line
        opt = uncon(brachistochrone,yint,epsilon_g)
        iteration_count = np.size(opt[2][2]) - 1
        iterations.append(iteration_count)
        func_calls.append(brach_func_calls)
    plt.figure()
    plt.subplot(1,2,1)
    plt.plot(n,func_calls,'bo-',label = 'Function Calls')
    plt.plot(n,iterations,'ro-', label = 'Major Iterations')
    axes = plt.gca()
    axes.set_ylim([0,140])
    plt.legend(loc = 'upper left')
    plt.xlabel('Dimensionality (n)')
    plt.ylabel('# of function calls/# of major iterations')
    plt.title('Linear Initial Seeding')

    #n = [4, 8, 16, 32, 64, 128, 256] #number of points to put on brachistochrone function
    n = [4, 8, 16]
    iterations = []
    func_calls = []
    del_y = 1./(n[0] + 1)
    yint = np.linspace(1-del_y, 0+del_y, n[0]) #seed the function initially with a line
    for i in range(0,len(n)):
        brach_func_calls = 0
        opt = uncon(brachistochrone,yint,epsilon_g)
        iteration_count = np.size(opt[2][2]) - 1
        iterations.append(iteration_count)
        func_calls.append(brach_func_calls)
        y = np.insert(opt[0], 0, 1) #add the initial value of H = 1
        y = np.append(y, 0.) #add the final value of H = 0
        x = np.linspace(0, 1, y.size) #the x vector
        xvals = np.linspace(0,1,y.size*2 - 2) #make double the x vector length
        yvals = np.interp(xvals,x,y) #linearly interpolate the y values
        yvals = np.delete(yvals,0) #delete the 1
        yvals = np.delete(yvals,np.size(yvals)-1) #delete the 0
        yint = yvals
    plt.subplot(1,2,2)
    plt.plot(n,func_calls,'bo-',label = 'Function Calls')
    plt.plot(n,iterations,'ro-', label = 'Major Iterations')
    axes = plt.gca()
    axes.set_ylim([0,140])
    plt.legend(loc = 'upper left')
    plt.xlabel('Dimensionality (n)')
    plt.ylabel('# of function calls/# of major iterations')
    plt.title('Seeding from previous n')
    plt.savefig('Pictures/HW_2/increased_dimensionality_Brachistochrone2.pdf', format = 'pdf')
    plt.show() #show all plots
