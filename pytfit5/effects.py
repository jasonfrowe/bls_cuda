import numpy as np
from numba import njit

@njit
def albedoMod(phi, ag):
    phi = phi + np.pi
    if phi > 2*np.pi:
        phi = phi - 2*np.pi

    alpha = abs(phi)      
    alpha = alpha - 2*np.pi * int(alpha/(2.0*np.pi))
    if alpha > np.pi:
        alpha = abs(alpha - 2*np.pi)

    phase = (np.sin(alpha) + (np.pi - alpha)*np.cos(alpha)) / np.pi
      
    return ag * phase

@njit
def ttv_lininterp(x, y, ntt, xin, npl):
    """
    Calculates a linear interpolation of xin.

    x: 2D array containing times of ttv. shape=(nb_planet, nb_ttv)
    y: 2D array of o-c. shape=(nb_planet, nb_ttv)
    ntt: 1D array containing nb of ttv. shape=(nb_planet,)
    """

    if ntt[npl] == 0:
        yout = 0
    
    elif xin <= x[npl, 0]:
        yout = y[npl,0] + (xin - x[npl,0]) / (x[npl,1] - x[npl,0]) * (y[npl,1] - y[npl,0])

    elif xin >= x[npl, ntt[npl] - 1]:
        n_ttv = ntt[npl] - 1
        yout = y[npl,n_ttv] + (xin - x[npl,n_ttv]) / (x[npl,n_ttv] - x[npl,n_ttv-1]) * (y[npl,n_ttv] - y[npl,n_ttv-1])

    else:
        for i in range(ntt[npl] - 1):
            if x[npl,i] < xin <= x[npl,i+1]:
                yout = y[npl,i] + (xin - x[npl,i]) / (x[npl,i+1] - x[npl,i]) * (y[npl,i+1] - y[npl,i])
                break
    
    return yout