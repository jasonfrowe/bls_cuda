import numpy as np
from numba import njit

@njit
def trueAnomaly(eccn, Eanom):
    """
    Calculates the true anomaly
    """

    ratio = (1 + eccn) / (1 - eccn)
    return 2 * np.arctan(np.sqrt(ratio) * np.tan(Eanom/2))

@njit
def distance(a, eccn, Tanom):
    """
    Calculates the distance between the star and the planet
    """

    return a * (1 - eccn*eccn) / (1 + eccn * np.cos(Tanom))

@njit
def solve_kepler_eq(eccn, Manom, Eanom, thres=1e-6, itmax=100):
    """
    Solves the Kepler equation using the Newton-Raphson method
    """

    diff = 1
    i = 0

    while (diff >= thres and i < itmax):
        diff = (Eanom - eccn*np.sin(Eanom) - Manom) / (1 - eccn*np.cos(Eanom))
        Eanom -= diff
        
        diff = abs(diff)
        i += 1

    return Eanom

def mark_intransit_data(phot, sol, tdurcut = 2.0):
    '''
    Usage: mark_intransit_data(phot, sol, tdurcut = 2)
      - phot : phot class with photometry products
      - sol  : transit model solution 
      - tdurcut : 
    '''

    itime_med = np.median(phot.itime)
    tdurcut_apply = tdurcut + itime_med #Pad tdurcut with integration time to account for smearing of transit by observation cadence 
    
    phot.tflag = np.zeros((phot.time.shape[0])) # pre-populate array to mark transit data (=1 when in transit)

    for i in range(sol.npl): #loop over all planets

        #Get period and T0 for current planet
        per = sol.per[i]
        epo = sol.t0[i]

        #Calculate Phase
        phase=(phot.time-epo)/per-np.floor((phot.time-epo)/per)
        #Keep phase in bounds of -0.5 to 0.5
        phase[phase<-0.5]+=1.0
        phase[phase>0.5]-=1.0
    
        #cut out-of-transit data to run faster.
        tdur = transitDuration(sol, i_planet = i)/per

        if tdurcut>0:
            phot.tflag[(phase>-tdurcut_apply*tdur)&(phase<tdurcut_apply*tdur)] = 1  #Mark in transit data


def transitDuration(sol, i_planet=0):
    """
    Calculates the transit duration in the same unit as the period
    """
    G = 6.674e-11

    density = sol.rho
    P = sol.per[i_planet]
    b = sol.bb[i_planet]
    Rp_Rs = sol.rdr[i_planet]

    a_Rs = 10 * np.cbrt(density * G * (P*86400)**2 / (3*np.pi))

    temp1 = (1 + Rp_Rs)**2 - b*b
    temp2 = 1 - (b/a_Rs)**2

    return P/np.pi * np.arcsin(min(1/a_Rs * np.sqrt(temp1/temp2), 1))

def rhostar(P, tdur):
    """
    Approximates the star density using the period and transit duration.
    Uses a simplified formula to relate tdur to a/Rs.
    P: period in days
    tdur: transit duration in days
    """
    G = 6.674e-11

    # Change to secs
    P = P*86400
    tdur = tdur*86400

    rho = 3*P/(np.pi**2 * tdur**3 * G)

    # Change to g/cm^3
    return rho / 1000
