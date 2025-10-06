import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from utils_python.transitmodel import transitModel
from utils_python.keplerian import transitDuration
from utils_python.effects import ttv_lininterp

def plotTransit(phot, sol, pl_to_plot=1, nintg=41, ntt=-1, tobs=-1, omc=-1):
    """
    Plots a transit model. Assuming time is in days. Set flux=0 for no scatterplot.

    phot: Phot object from reading data file
    sol: Transit model object with parameters
    nintg: Number of points inside the integration time
    pl_to_plot: Index of planet to plot. 1 being the first planet
    """

    # Read phot class
    time = phot.time[(phot.icut == 0)]
    if np.isclose(np.median(phot.flux_f), 0, atol=0.2):
        flux = phot.flux_f[(phot.icut == 0)] + 1
    else:
        flux = phot.flux_f[(phot.icut == 0)]
    itime = phot.itime[(phot.icut == 0)]

    pl_to_plot -= 1 # Shift indexing to start at 0

    t0 = sol.t0[pl_to_plot]
    per = sol.per[pl_to_plot]
    zpt = sol.zpt

    # Copy the original Rp/R* before modifying it
    rdr = sol.rdr.copy()

    # Remove the other planets from the model
    for i in range(sol.npl):
        if i != pl_to_plot:
            sol.rdr[i] = 0

    tmodel = transitModel(sol, time, itime, nintg, ntt, tobs, omc) - zpt
    flux = flux # - zpt # Remove the zero point to always plot around 1

    # Second model with only the other planets to substract
    sol.rdr = rdr.copy()
    sol.rdr[pl_to_plot] = 0
    tmodel2 = transitModel(sol, time, itime, nintg, ntt, tobs, omc)

    # Restore the original Rp/R*
    sol.rdr = rdr

    tdur = transitDuration(sol, pl_to_plot)*24
    if tdur < 0.01 or np.isnan(tdur):
        tdur = 2

    # Fold the time array and sort it. Handle TTVs
    ph1 = t0/per - np.floor(t0/per)
    phase = np.empty(len(time))
    for i, x in enumerate(time):
        if type(ntt) is not int and ntt[pl_to_plot] > 0:
            ttcor = ttv_lininterp(tobs, omc, ntt, x, pl_to_plot)
        else:
            ttcor = 0
        t = x - ttcor
        phase[i] = (t/per - np.floor(t/per) - ph1) * per*24
        if phase[i] >  0.5 * per*24:
            phase[i] -= per*24
        if phase[i] < -0.5 * per*24:
            phase[i] += per*24

    i_sort = np.argsort(phase)
    phase_sorted = phase[i_sort]
    model_sorted = tmodel[i_sort]

    stdev = np.std(flux - tmodel)

    # Remove the other planets
    fplot = flux - tmodel2 + 1

    # Find bounds of plot
    i1, i2 = np.searchsorted(phase_sorted, (-tdur, tdur))
    if i1 == i2:
        i1 = 0
        i2 = len(model_sorted)
    ymin = min(model_sorted[i1:i2])
    ymax = max(model_sorted[i1:i2])
    y1 = ymin - 0.1*(ymax-ymin) - 3.0*stdev
    y2 = ymax + 0.1*(ymax-ymin) + 3.0*stdev
    if np.abs(y2 - y1) < 1.0e-10:
        y1 = min(flux)
        y2 = max(flux)

    mpl.rcParams.update({'font.size': 22}) # Adjust font
    plt.figure(figsize=(12,6)) # Adjust size of figure
    plt.scatter(phase, fplot, c="blue", s=100.0, alpha=0.35, edgecolors="none") #scatter plot
    plt.plot(phase_sorted, model_sorted, c="red", lw=3.0)
    plt.xlabel('Phase (hours)') #x-label
    plt.ylabel('Relative Flux') #y-label
    plt.axis((-1.5*tdur, 1.5*tdur, y1, y2))
    plt.tick_params(direction="in")
    plt.show()

def printParams(sol):
    """
    Prints the parameters in a nice way.

    sol: Transit model object containing the parameters to print
    """

    stellarDict = {
        "ρ* (g/cm³)": "rho", "c1": "nl1", "c2": "nl2", "q1": "nl3", "q2": "nl4",
        "Dilution": "dil", "Velocity Offset": "vof", "Photometric zero point": "zpt"
    }

    planetDict = {
        "t0 (days)": "t0", "Period (days)": "per", "Impact parameter": "bb", "Rp/R*": "rdr",
        "sqrt(e)cos(w)": "ecw", "sqrt(e)sin(w)": "esw", "RV Amplitude (m/s)": "krv",
        "Thermal eclipse depth (ppm)": "ted", "Ellipsoidal variations (ppm)": "ell", "Albedo amplitude (ppm)": "alb"
    }

    # Stellar params
    for key in stellarDict:
        var_name = stellarDict[key]
        val = getattr(sol, var_name)
        try:
            err = getattr(sol, "d" + var_name)
        except:
            err = 0

        if val != 0:
            exponent = np.floor(np.log10(abs(val)))
        else:
            exponent = 1

        if abs(exponent) > 2:
            print(f"{key + ':':<30} {val:>10.3e} ± {err:.3e}")
        elif len(str(val)) > 7:
            print(f"{key + ':':<30} {val:>10.7f} ± {err:.7f}")
        else:
            print(f"{key + ':':<30} {val:>10} ± {err}")

    # Planet params
    for j in range(sol.npl):
        if sol.npl > 1:
            print(f"\nPlanet #{j + 1}:")
        for key in planetDict:
            var_name = planetDict[key]
            val = getattr(sol, var_name)
            try:
                err = getattr(sol, "d" + var_name)
            except:
                err = 0

            p_val = val[j]
            p_err = err[j]
            if p_val != 0:
                exponent = np.floor(np.log10(abs(p_val)))
            else:
                exponent = 1

            if abs(exponent) > 2:
                print(f"{key + ':':<30} {p_val:>10.3e} ± {p_err:.3e}")
            elif len(str(p_val)) > 7:
                print(f"{key + ':':<30} {p_val:>10.7f} ± {p_err:.7f}")
            else:
                print(f"{key + ':':<30} {p_val:>10} ± {p_err}")
