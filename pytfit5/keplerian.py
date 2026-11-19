import numpy as np
from numba import njit

@njit(cache=True)
def trueAnomaly(eccn, Eanom):
    """
    Calculates the true anomaly
    """

    ratio = (1 + eccn) / (1 - eccn)
    return 2 * np.arctan(np.sqrt(ratio) * np.tan(Eanom/2))

@njit(cache=True)
def distance(a, eccn, Tanom):
    """
    Calculates the distance between the star and the planet
    """

    return a * (1 - eccn*eccn) / (1 + eccn * np.cos(Tanom))

@njit(cache=True)
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
    Total transit duration T14, in the same unit as the period, for an eccentric orbit
    (Kipping 2010, eq. 15; as transitdur.f of TransitFit5). For e = 0 this is the usual
    P/pi * arcsin(sqrt((1 + k)^2 - b^2) / (a/R* sin i)).
    """
    G = 6.674e-11

    density = sol.rho
    P = sol.per[i_planet]
    b = abs(sol.bb[i_planet])
    Rp_Rs = abs(sol.rdr[i_planet])
    ecw = getattr(sol, "ecw", [0.0] * (i_planet + 1))[i_planet]
    esw = getattr(sol, "esw", [0.0] * (i_planet + 1))[i_planet]

    a_Rs = 10 * np.cbrt(density * G * (P*86400)**2 / (3*np.pi))

    e = min(ecw*ecw + esw*esw, 0.99)
    rho_c = (1 - e*e) / (1 + np.sqrt(e)*esw)       # r/a at mid-transit; sqrt(e)*esw = e sin(omega)
    cosi = b / (a_Rs * rho_c)
    if cosi >= 1:
        return 0.0
    sini = np.sqrt(1 - cosi*cosi)

    temp1 = (1 + Rp_Rs)**2 - b*b
    if temp1 <= 0:
        return 0.0

    return P/np.pi * rho_c*rho_c/np.sqrt(1 - e*e) * np.arcsin(min(np.sqrt(temp1) / (a_Rs*rho_c*sini), 1))


def ecc_omega(ecw, esw):
    """
    e and omega (radians, in [0, 2pi)) from the fitted parameters ecw = sqrt(e) cos(omega),
    esw = sqrt(e) sin(omega). omega is the argument of periastron in the convention of the
    radial-velocity literature (Winn 2010): the transit is at true anomaly f = pi/2 - omega
    and the star's radial velocity is v = K [cos(f + omega) + e cos(omega)].
    """
    e = ecw*ecw + esw*esw
    omega = np.mod(np.arctan2(esw, ecw), 2*np.pi) if e > 0 else 0.0
    return e, omega


def ecw_esw(e, omega):
    """Fitted parameters (sqrt(e) cos(omega), sqrt(e) sin(omega)) from e and omega (radians)."""
    return np.sqrt(e)*np.cos(omega), np.sqrt(e)*np.sin(omega)


def from_legacy_transitfit5(ecw_old, esw_old):
    """
    Convert sqrt(e)cos(w), sqrt(e)sin(w) in which w was the true anomaly at mid-transit
    (w = pi/2 - omega) to the current parameters. This applies to pytfit5 before November 2026 and to
    TransitFit5 between 14 August and November 2026 (TransitFit5 fits before August 2026 already
    follow the current convention).
    The conversion is a swap: sqrt(e) cos(omega) = sqrt(e) sin(w), sqrt(e) sin(omega) = sqrt(e) cos(w).
    """
    return esw_old, ecw_old


def radial_velocity(time, t0, per, K, ecw, esw, gamma=0.0):
    """
    Radial velocity of the star (positive away from the observer), in the units of K, for a
    planet with mid-transit time t0 and period per (same units as time):
    v = gamma + K [cos(f + omega) + e cos(omega)].
    """
    e, omega = ecc_omega(ecw, esw)
    e = min(e, 0.99)
    f_tr = 0.5*np.pi - omega                                   # true anomaly at mid-transit
    E_tr = 2*np.arctan(np.sqrt((1 - e)/(1 + e)) * np.tan(0.5*f_tr))
    M_tr = E_tr - e*np.sin(E_tr)
    time = np.atleast_1d(np.asarray(time, dtype=float))
    v = np.empty(time.size)
    for i, t in enumerate(time):
        M = np.mod(M_tr + 2*np.pi*(t - t0)/per, 2*np.pi)
        E = solve_kepler_eq(e, M, M, thres=1e-12)
        f = trueAnomaly(e, E)
        v[i] = gamma + K*(np.cos(f + omega) + e*np.cos(omega))
    return v


def ecc_params_from_state(r, v, mu, observer=(0.0, 0.0, 1.0)):
    """
    pytfit5 eccentricity parameters from a position and velocity, e.g. from an N-body code.

    r, v     : position and velocity of the planet relative to the star (astrocentric) or to the
               barycentre of the interior bodies (Jacobi), in any consistent units
    mu       : G (M_star + M_planet) in the same units
    observer : vector pointing from the system towards the observer

    With e_vec the eccentricity vector, h the orbital angular momentum and o the direction to the
    observer projected onto the orbit plane (unit vectors h_hat, o_hat):
        e sin(omega) = e_vec . o_hat,    e cos(omega) = e_vec . (o_hat x h_hat).
    This needs no convention for the reference plane or the line of nodes.

    Returns ecw, esw, e, omega (radians).
    """
    r = np.asarray(r, dtype=float); v = np.asarray(v, dtype=float)
    o = np.asarray(observer, dtype=float)
    h = np.cross(r, v)
    h_hat = h / np.linalg.norm(h)
    e_vec = np.cross(v, h)/mu - r/np.linalg.norm(r)
    o_p = o - np.dot(o, h_hat)*h_hat                            # observer direction in the orbit plane
    o_hat = o_p / np.linalg.norm(o_p)
    ecosw = np.dot(e_vec, np.cross(o_hat, h_hat))
    esinw = np.dot(e_vec, o_hat)
    e = np.hypot(ecosw, esinw)
    if e == 0:
        return 0.0, 0.0, 0.0, 0.0
    omega = np.mod(np.arctan2(esinw, ecosw), 2*np.pi)
    return ecosw/np.sqrt(e), esinw/np.sqrt(e), e, omega

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
