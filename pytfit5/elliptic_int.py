import numpy as np
from numba import njit

@njit
def ellK(k):
    """
    Computes the elliptic integral of the first kind (Hasting's approximation)
    """
    m1 = 1 - k*k
    a0 = 1.38629436112
    a1 = 0.09666344259
    a2 = 0.03590092383
    a3 = 0.03742563713
    a4 = 0.01451196212
    b0 = 0.5
    b1 = 0.12498593597
    b2 = 0.06880248576
    b3 = 0.03328355346
    b4 = 0.00441787012

    ek1 = a0+m1*(a1+m1*(a2+m1*(a3+m1*a4)))
    ek2 = (b0+m1*(b1+m1*(b2+m1*(b3+m1*b4)))) * np.log(m1)

    return ek1 - ek2

@njit
def ellE(k):
    """
    Computes the elliptic integral of the second kind (Hasting's approximation)
    """

    m1 = 1 - k*k
    a1 = 0.44325141463
    a2 = 0.06260601220
    a3 = 0.04757383546
    a4 = 0.01736506451
    b1 = 0.24998368310
    b2 = 0.09200180037
    b3 = 0.04069697526
    b4 = 0.00526449639
    
    ee1 = 1 + m1*(a1+m1*(a2+m1*(a3+m1*a4)))
    ee2 = m1*(b1+m1*(b2+m1*(b3+m1*b4))) * np.log(m1)
    
    return ee1 - ee2

@njit
def ellPi(n, k):
    """
    Calculates the elliptic integral of the third kind taken from Bulirsch (1965)
    """

    kc = np.sqrt(1 - k*k)
    p = np.sqrt(1 - n)
    m0 = 1
    c = 1
    d = 1/p
    e = kc

    nit = 0
    diff = 1

    while (nit < 10000 and diff > 1e-3):
        f = c
        c = d/p + c
        g = e/p
        d = 2 * (f*g + d)
        p = g + p
        g = m0
        m0 = kc + m0

        diff = abs(1 - kc/g)
        kc = 2*np.sqrt(e)
        e = kc*m0
            
        nit += 1

    return 0.5*np.pi*(c*m0+d)/(m0*(m0+p))

# import matplotlib.pyplot as plt
# import mpmath as mm

# x = np.linspace(0, 0.9, 100)
# y = np.zeros(len(x))
# y_test = np.zeros(len(x))
# for i, x_ in enumerate(x):
#     y[i] = mm.ellippi(-999, x_*x_)
#     y_test[i] = ellPi(-999, x_)

# print(np.max(np.abs(y - y_test)))

# plt.plot(x, y)
# plt.plot(x, y_test)
# plt.plot(x, y - y_test)
# plt.show()