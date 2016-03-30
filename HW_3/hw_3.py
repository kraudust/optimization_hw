import numpy as np
from cmath import sin, cos, sqrt, pi
from scipy.optimize import minimize
# from matplotlib import pyplot as plt

Alast = []
csave = []
dcsave = []
truss_func_calls = 0

def bar(E, A, L, phi):
    """Computes the stiffness and stress matrix for one element

    Parameters
    ----------
    E : float
        modulus of elasticity
    A : float
        cross-sectional area
    L : float
        length of element
    phi : float
        orientation of element

    Outputs
    -------
    K : 4 x 4 ndarray
        stiffness matrix
    S : 1 x 4 ndarray
        stress matrix

    """

    # rename
    c = cos(phi)
    s = sin(phi)

    # stiffness matrix
    k0 = np.array([[c**2, c*s], [c*s, s**2]],dtype = complex)
    k1 = np.hstack([k0, -k0])
    K = E*A/L*np.vstack([k1, -k1])

    # stress matrix
    S = E/L*np.array([[-c, -s, c, s]])

    return K, S

def node2idx(node, DOF):
    """Computes the appropriate indices in the global matrix for
    the corresponding node numbers.  You pass in the number of the node
    (either as a scalar or an array of locations), and the degrees of
    freedom per node and it returns the corresponding indices in
    the global matrices

    """

    idx = np.array([], dtype=np.int)

    for i in range(len(node)):

        n = node[i]
        start = DOF*(n-1)
        finish = DOF*n

        idx = np.concatenate((idx, np.arange(start, finish, dtype=np.int)))

    return idx

def truss(start, finish, phi, A, L, E, rho, Fx, Fy, rigid):
    """Computes mass and stress for an arbitrary truss structure

    Parameters
    ----------
    start : ndarray of length nbar
        index of start of bar (1-based indexing) start and finish can be in any order as long as consistent with phi
    finish : ndarray of length nbar
        index of other end of bar (1-based indexing)
    phi : ndarray of length nbar (radians)
        defines orientation or bar
    A : ndarray of length nbar
        cross-sectional areas of each bar
    L : ndarray of length nbar
        length of each bar
    E : ndarray of length nbar
        modulus of elasticity of each bar
    rho : ndarray of length nbar
        material density of each bar
    Fx : ndarray of length nnode
        force in the x-direction at each node
    Fy : ndarray of length nnode
        force in the y-direction at each node
    rigid : list(boolean) of length nnode
        True if node_i is rigidly constrained

    Outputs
    -------
    mass : float
        mass of the entire structure
    stress : ndarray of length nbar
        stress of each bar

    """
    global truss_func_calls
    n = len(Fx)  # number of nodes
    DOF = 2  # number of degrees of freedom
    nbar = len(A)  # number of bars

    # mass
    mass = np.sum(rho*A*L)

    # stiffness and stress matrices
    K = np.zeros((DOF*n, DOF*n), dtype = complex)
    S = np.zeros((nbar, DOF*n), dtype = complex)

    for i in range(nbar):  # loop through each bar

        # compute submatrix for each element
        Ksub, Ssub = bar(E[i], A[i], L[i], phi[i])

        # insert submatrix into global matrix
        idx = node2idx([start[i], finish[i]], DOF)  # pass in the starting and ending node number for this element
        K[np.ix_(idx, idx)] += Ksub
        S[i, idx] = Ssub

    # applied loads
    F = np.zeros((n*DOF, 1))

    for i in range(n):
        idx = node2idx([i+1], DOF)  # add 1 b.c. made indexing 1-based for convenience
        F[idx[0]] = Fx[i]
        F[idx[1]] = Fy[i]


    # boundary condition
    idx = np.squeeze(np.where(rigid))
    remove = node2idx(idx+1, DOF)  # add 1 b.c. made indexing 1-based for convenience

    K = np.delete(K, remove, axis=0)
    K = np.delete(K, remove, axis=1)
    F = np.delete(F, remove, axis=0)
    S = np.delete(S, remove, axis=1)

    # solve for deflections
    d = np.linalg.solve(K, F)

    # compute stress
    stress = np.dot(S, d).reshape(nbar)
    truss_func_calls = truss_func_calls + 1
    return mass, stress, d, K, S

def tenbartruss(A, grad_type='FD'):
    """This is the subroutine for the 10-bar truss.  You will need to complete it.

    Parameters
    ----------
    A : ndarray of length 10
        cross-sectional areas of all the bars
    grad_type : string (optional)
        gradient type.  'FD' for finite difference, 'CS' for complex step,
        'AJ' for adjoint

    Outputs
    -------
    mass : float
        mass of the entire structure
    stress : ndarray of length 10
        stress of each bar
    dmass_dA : ndarray of length 10
        derivative of mass w.r.t. each A
    dstress_dA : 10 x 10 ndarray
        dstress_dA[i, j] is derivative of stress[i] w.r.t. A[j]

    """
    # --- setup 10 bar truss ----
    L0 = 360.0 #length of square sides (in.)
    Ld = sqrt(2.)*L0 #length of the diagonal beams
    P = 100000.0 #applied load (lb)
    rho = 0.1 #material density (lb/in^3)
    E = 1.e7 #modulus of elasticity (psi)
    L = np.array([L0, L0, L0, L0, L0, L0, Ld, Ld, Ld, Ld]) #an array of lengths from bar 1 to 10
    E = np.ones(10)*E #the modulus of elasticity for each truss element
    rho = np.ones(10)*rho #density of each truss element
    rigid = [False, False, False, False, True, True] #True = rigidly constrained nodes
    start = np.array([ 5, 3, 6, 4, 4, 2, 5, 6, 3, 4])#start of each truss element with base 1 indexing
    finish = np.array([3, 1, 4, 2, 3, 1, 4, 3, 2, 1])#end of each truss element with base 1 indexing
    phi = np.array([0., 0., 0., 0., pi/2., pi/2., -pi/4., pi/4., -pi/4., pi/4.]) #truss angles measured from start
    Fx = np.zeros(6)
    Fy = np.array([0., -P, 0., -P, 0., 0.])

    # --- call truss function ----
    mass, stress, d, K, S = truss(start,finish,phi,A,L,E,rho,Fx,Fy,rigid)

    # --- compute derivatives for provided grad_type ----
    dmass_dA = np.zeros(np.size(L), dtype = complex)
    dstress_dA = np.empty((np.size(L),np.size(L)), dtype = complex)
    #----------------------------Finite Difference--------------------------------------
    if grad_type == 'FD':
        B = np.zeros(np.size(L), dtype = complex)
        for i in range(0,np.size(L)):
            h = 1.e-6
            B[:] = A[:]
            B[i] = B[i] + h
            mass_h, stress_h, _, _, _ = truss(start, finish, phi, B, L, E, rho, Fx, Fy, rigid)
            dmass_dA[i] = (mass_h-mass)/h #forward difference approximation
            dstress_dA[i] = (stress_h - stress)/h

    #-----------------------------Complex Step-------------------------------------------
    if grad_type == 'CS':
        B = np.zeros(np.size(L), dtype = complex)
        h = 1.e-20
        h = complex(0.,h)
        for i in range(0,np.size(L)):
            B[:] = A[:]
            B[i] = B[i] + h
            mass_h, stress_h, _, _, _ = truss(start, finish, phi, B, L, E, rho, Fx, Fy, rigid)
            dmass_dA[i] = mass_h/h
            dstress_dA[i] = stress_h/h

    #----------------------------Adjoint-------------------------------------------------
    if grad_type == 'AJ':
        dK_dA = np.zeros((10,8,8), dtype = complex)
        dstress_dA = np.empty([np.size(L), np.size(L), 1], dtype = complex)
        for j in range(0,10):
            n = len(Fx) #number of nodes
            DOF = 2 # number of degrees of freedom
            nbar = len(A) # number of bars
            #mass
            mass = np.sum(rho*A*L)
            dK_dA_sub = np.zeros((DOF*n, DOF*n), dtype = complex)

            Ksub, _ = bar(E[j], A[j], L[j], phi[j])
            idx = node2idx([start[j], finish[j]], DOF)
            dK_dA_sub[np.ix_(idx, idx)] += Ksub

            for i in range(n):
                idx = node2idx([i+1], DOF)
            idx = np.squeeze(np.where(rigid))
            remove = node2idx(idx+1, DOF)
            dK_dA_sub = np.delete(dK_dA_sub, remove, axis = 0)
            dK_dA_sub = np.delete(dK_dA_sub, remove, axis = 1)
            dK_dA[j] = dK_dA_sub/A[j]
            dstress_dA[j] = np.dot(np.dot(np.dot(-S,np.linalg.inv(K)),dK_dA[j]),d)
            dmass_dA[j] = rho[j]*L[j]
    dstress_dA = np.reshape(dstress_dA,(10,10),order = 'C')
    return mass.real, stress.real, dmass_dA.real, dstress_dA.real

def obj(A):
    mass, stress, dmass_dA, dstress_dA = tenbartruss(A)
    global Alast, csave, dcsave
    Alast = A
    csave = stress
    dcsave = dstress_dA
    return mass, dmass_dA

def con(A):
    global Alast, csave, dcsave
    yield_stress1 = 25.e3
    yield_stress2 = 75.e3
    yield_stress = np.ones(10)
    yield_stress = yield_stress*yield_stress1
    yield_stress[8] = yield_stress2
    c = np.zeros(20)
    if not np.all(Alast == A):
        mass, csave, dmass_dA, dcsave = tenbartruss(A)
        Alast = A
    for i in range(0, 10):
        c[i] = yield_stress[i] - csave[i]
        c[i+10] = yield_stress[i] + csave[i]
    return c

def congrad(A):
    global Alast, csave, dcsave
    if not np.all(Alast == A):
        mass, csave, dmass_dA, dcsave = tenbartruss(A)
        Alast = A
    grad = np.empty((20,10))
    grad1 = np.zeros((10,10))
    grad2 = np.zeros((10,10))
    for i in range(0, 10):
        grad1[i] = -dcsave[i]
        grad2[i] = dcsave[i]
    grad1 = np.transpose(grad1)
    grad2 = np.transpose(grad2)
    grad = np.vstack((grad1, grad2))
    return grad

if __name__ == '__main__': #main code goes here
    #global truss_func_calls
    A0 = 2.0 #initial cross sectional area in inches^2
    A = np.ones(10, dtype = complex)*A0
    #A = np.ones(10)*A0
    # mass, stress, dmass_dA, dstress_dA = tenbartruss(A, grad_type = 'AJ')
    # print "mass: "
    # print mass
    # print "stress: "
    # print stress
    # print "dmass_dA: "
    # print dmass_dA
    # print "dstress_dA: "
    # print dstress_dA

    #-----------------------------Optimization------------------------------
    #convert bounds to list of tuples in format scipy wants
    bounds = np.zeros((10,2))
    bounds[:] = (0.1,100)
    constraints = {'type': 'ineq', 'fun': con, 'jac': congrad}

    options = {'disp': True, 'iprint': 2, 'maxiter': 600}
    res = minimize(obj, A, method = 'SLSQP', jac = True, bounds = bounds, tol = 1e-6,constraints = constraints, options = options)
    A_opt =  res.x
    print A_opt
    print truss_func_calls
    print tenbartruss(A_opt,grad_type = 'AJ')[1]

    #Adjoint iterations vs. convergence criteria plot
    major_iter = np.zeros(18)
    fun_diff = np.zeros(18)
    for i in range(0,18):
        major_iter[i] = i+1
    obj_fun = np.array([8.392935E+02, 8.773779E+02, 1.108012E+03, 1.358784E+03, 1.475279E+03, 1.497036E+03,
                        1.497599E+03, 1.497600E+03, 1.497600E+03, 1.497600E+03, 1.497600E+03, 1.497600E+03,
                        1.497600E+03, 1.497600E+03, 1.497600E+03, 1.497600E+03, 1.497600E+03, 1.497600E+03])
    for i in range(0, 17):
        fun_diff[i] = obj_fun[i+1] - obj_fun[i]
    fun_diff[17] = 0
    #
    #
    # # plt.figure()
    # # plt.plot(major_iter, fun_diff)
    # # plt.yscale('log')
    # # plt.ylabel('Delta Mass')
    # # plt.xlabel('Major Iterations')
    # # plt.title('Convergence Plot for Adjoint Method')
    # # plt.xlim((1, 8))
    # # plt.show()
