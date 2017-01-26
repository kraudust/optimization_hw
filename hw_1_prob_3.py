import numpy as np
import math
import matplotlib.pyplot as plt
import hw_1_prob_2

def endurance(x):
    """
    :param x: vector of arguments to the function in the order x = (b, V) where b is wingspan and V is flight velocity
    :return: endurance cost, dJ_db, dJ_dV
    """
    b = x[0]
    V = x[1]
    rho = 1.23
    mu = 1.8e-5
    k = 1.2
    C_L = 0.4
    e = 0.96
    W = 1000
    eta_max = 0.8
    V_bar = 20.0
    sigma = 5.0
    L = W
    q = 0.5*rho*V**2
    S = L/(C_L*q)
    S_wet = 2.05*S
    c = S/b
    Re = (rho*V*c)/mu
    C_f = 0.074/(Re**0.2)
    D_f = k*C_f*S_wet*q
    D_i = (L**2)/(q*np.pi*(b**2)*e)
    D = D_i + D_f
    eta = eta_max*math.exp((-(V-V_bar)**2)/(2.0*sigma**2))
    J = (-eta/(D*V))*(1e4)
    return J
def endurance_grad(x):
    """
    :param x: vector of arguments to the function in the order x = (b, V) where b is wingspan and V is flight velocity
    :return: dJ_db, dJ_dV
    """
    b = x[0]
    V = x[1]
    rho = 1.23
    mu = 1.8e-5
    k = 1.2
    C_L = 0.4
    e = 0.96
    W = 1000
    eta_max = 0.8
    V_bar = 20.0
    sigma = 5.0
    L = W
    q = 0.5*rho*V**2
    S = L/(C_L*q)
    S_wet = 2.05*S
    c = S/b
    Re = (rho*V*c)/mu
    C_f = 0.074/(Re**0.2)
    D_f = k*C_f*S_wet*q
    D_i = (L**2)/(q*np.pi*(b**2)*e)
    D = D_i + D_f
    eta = eta_max*math.exp((-(V-V_bar)**2)/(2.0*sigma**2))
    J = (-eta/(D*V))*(1e4)

    dJ_db = -(J/(D*b))*(0.2*D_f - 2.0*D_i)
    dJ_dV = -(J*((1.0/V) + ((V - V_bar)/(sigma**2.0)) + (0.2*D_f - 2.0*D_i)/(D*V)))
    z = np.array([dJ_db,dJ_dV])
    return z
def dir_grad(x, p, grad, args = []):
    """
    :param x: location to evaluation gradient
    :param p: direction you are line searching
    :param grad: gradient function handle
    :param args: arguments to the gradient function
    :return: directional derivative
    """
    if not args:
        return np.dot(grad(x),p)
    else:
        return np.dot(grad(x,args),p)
#############################################################################################################
if __name__ == '__main__':
    b_V = [np.array([15.0, 15.0]), np.array([25.0, 16.0]), np.array([20.0, 24.0])]
    p_arr = [np.array([3.0, 3.0]), np.array([3.0, 1.0]), np.array([0.1, -0.5])]
    # --- setup grid for contour plot ---
    nx = 400  # number of points in x-direction
    ny = 400  # number of points in y-direction
    x = np.linspace(5, 35, nx)  # nx points equally spaced between 5...35
    y = np.linspace(5, 30, ny)  # ny points equally spaced between 5...30
    X, Y = np.meshgrid(x, y, indexing='ij')  # 2D array (matrix) of points across x and y
    Z = np.zeros((nx, ny))  # initialize output of size (nx, ny)
    # --- evaluate across grid ---
    for i in range(nx):
        for j in range(ny):
            t = (X[i, j], Y[i, j])
            Z[i, j] = endurance(t)
    inp = raw_input('Enter "b" for a simple backtracking line search or "e" for an exact line search:')

    #Inexact line search using backtracking
    if inp == "b":
        for q in range(0,3):
            x_0 = b_V[q]
            p = p_arr[q]
            alpha_0 = 1.0
            rho = 0.5
            mu = 10e-6
            z = hw_1_prob_2.simple_back_line(x_0, p, alpha_0, rho, mu, endurance, endurance_grad, [], [])
            alpha = z[0]
            i = z[1]
            x_f = x_0 + alpha*p
            print 'Alpha:', alpha, '   p:', p, '   x_0, ', x_0, '   x_f:', x_f, '   J-Value:', endurance(x_f), \
                '   Iterations:', i
            # --- contour plot ---
            plt.figure(q)  # start a new figure
            plt.contour(X, Y, Z, 300)  # using 300 contour lines.
            plt.plot(x_0[0],x_0[1], 'ko', x_f[0], x_f[1], 'ro')
            x = np.linspace(x_0[0], x_f[0], 100)
            y = np.linspace(x_0[1], x_f[1], 100)
            plt.plot(x, y, 'k')
            plt.colorbar()  # add a color bar
            plt.xlabel('Wingspan (b)(m)')  # labels for axes
            plt.ylabel('Flight Speed (V) (m/s)')
            plt.legend(['Initial Point', 'Final Point'])
            plt.title('$x_0$ = %s   p = %s' %(x_0, p))
            plt.savefig('Pictures/HW_1/hw_1_prob_3_inexact figure %d' %(q)+'.pdf', format = 'pdf')
        plt.show()  # show plot

    # Exact line search using golden section search
    if inp == 'e':
        for q in range(0,3):
            x_0 = b_V[q]
            p = p_arr[q]
            solution = hw_1_prob_2.golden_section_search(endurance,x_0,x_0+13.0*p,10e-6)
            x_f = solution[0]
            i = solution[1]
            print 'p:', p, '   x_0, ', x_0, '   x_f:', x_f, '   J-Value:', endurance(x_f), '   Iterations:', i
            # --- contour plot ---
            plt.figure(q)  # start a new figure
            plt.contour(X, Y, Z, 300)  # using 300 contour lines.
            plt.plot(x_0[0],x_0[1], 'ko', x_f[0], x_f[1], 'ro')
            x = np.linspace(x_0[0], x_f[0], 100)
            y = np.linspace(x_0[1], x_f[1], 100)
            plt.plot(x, y, 'k')
            plt.colorbar()  # add a color bar
            plt.xlabel('Wingspan (b)(m)')  # labels for axes
            plt.ylabel('Flight Speed (V) (m/s)')
            plt.legend(['Initial Point', 'Final Point'])
            plt.title('$x_0$ = %s   p = %s' %(x_0, p))
            plt.savefig('Pictures/HW_1/hw_1_prob_3_exact figure %d' %(q)+'.pdf', format = 'pdf')
        #plt.show()  # show plot