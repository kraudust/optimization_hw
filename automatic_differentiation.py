import numpy as np
import scipy as sp
from algopy import UTPM


def jensen_power(xin,params):
    r_0, rho, U_velocity = params
    #loss in one turbine
    loss = xin
    a = 1./3.
    Cp = 4.*a*(1-a)**2.
    A = r_0**2*np.pi
    V = (1-loss)*U_velocity
    "Calculate Power from a single turbine"
    P = 0.5*rho*A*Cp*V**3
    return P

if __name__ == '__main__':

    rho = 1.1716
    U_velocity = 8.
    r_0 = 40
    loss = 0.5
    params = np.array([r_0, rho, U_velocity])
    xin = loss
    x_algopy = UTPM.init_jacobian(xin)
    power = jensen_power(x_algopy,params)
    derivative_automatic = UTPM.extract_jacobian(power)
    print "Automatic Differentiation Derivative: ", derivative_automatic
    h = 1e-6
    xin_h = xin + h
    power_h = jensen_power(xin_h,params)
    power_normal = jensen_power(loss,params)
    derivative_finite = (power_h - power_normal)/h
    print derivative_finite




