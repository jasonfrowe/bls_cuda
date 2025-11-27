import os

import numpy as np

import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib import ticker
import matplotlib.gridspec as gridspec

from tqdm import tqdm

from numba import jit #We will attempt to precompile some routines
from numba import njit, prange

from numba import cuda
from math import floor

from scipy.signal import medfilt

#Constants
G       = 6.674e-11           # N m^2 kg^-2  Gravitation constant
Rsun    = 696265.0e0*1000.0e0 # m  radius of Sun
Msun    = 1.9891e30           # kg  mass of Sun
pifac   = 1.083852140278e0    # (2pi)^(2/3)/pi
day2sec = 86400.0             # seconds in a day
onehour = 1.0/24.0            # one hour convected to day.


class gbls_inputs_class:
    def __init__(self):
        self.filename = "filename.txt"
        self.lcdir    = ""
        self.zerotime = 0.0
        self.freq1    = -1
        self.freq2    = -1
        self.ofac     = 8.0
        self.Mstar    = 1.0
        self.Rstar    = 1.0
        self.nper     = 5000
        self.minbin   = 5
        self.plots    = 1    # 0 = no plots, 1 = X11, 2 = PNG+X11, 3 = PNG

class gbls_ans_class:
    def __init__(self):
        self.epo      = 0.0
        self.bper     = 0.0
        self.bpower   = -1
        self.snr      = -1
        self.tdur     = 8.0
        self.depth    = 1.0

def bls(gbls_inputs):

    if gbls_inputs.lcdir == "":
        filename = gbls_inputs.filename
    else:
        filename = gbls_inputs.lcdir + "/" + gbls_inputs.filename
    Keptime  = gbls_inputs.zerotime
    freq1    = gbls_inputs.freq1
    freq2    = gbls_inputs.freq2
    ofac     = gbls_inputs.ofac
    Mstar    = gbls_inputs.Mstar
    Rstar    = gbls_inputs.Rstar
    minbin   = gbls_inputs.minbin
    nper     = gbls_inputs.nper
    
    time, flux = readfile(filename)
    #Simple transform of time and flux arrays
    mintime = np.min(time)
    time = time - mintime #time starts at zero
    flux = flux - np.median(flux)  #flux is centred around 0

    npt   = time.shape[0]                     #number of data points
    nb    = np.min((2000, time.shape[0]-1.0)) #npt - 1, sets how the data in binned in phase.
    nyq   = nyquest_cpu(time)            #nyquest [c/d] -- CPU seems to be faster x2 for 1080ti vs 10700
    mint  = np.min(time)                      #minimum time [d]
    maxt  = np.max(time)                      #max time [d]

    if freq1 <= 0:
        freq1 = 2.0/(maxt-mint)                   #lowest frequency [c/d]
    if freq2 <= 0:
        freq2 = 2.0                               #highest frequency [c/d]

    #Compute some quantities 
    steps = int(ofac * (freq2 - freq1) * npt/nyq) #naive number of frequencies to scan
    df0   = (freq2 - freq1) / steps
    
    #Calculate the number of steps needed to scan the frequency range 
    nstep = calc_nsteps_cpu(Mstar, Rstar, freq1, freq2, nyq, ofac, npt, df0, nb, minbin)
    
    #Calculate the frequencies that we will be folding at
    freqs, kmis = calc_freqs_cpu(nstep, Mstar, Rstar, freq1, freq2, nyq, ofac, npt, df0, nb, minbin)

    #Transfer time and flux arrays to GPU
    time_g  = cuda.to_device(time)
    flux_g  = cuda.to_device(flux)
    freqs_g = cuda.to_device(freqs)
    kmis_g  = cuda.to_device(kmis)
    
    #Store some constants on the GPU
    const    = np.zeros((2), dtype = np.int32)
    const[0] = nb
    const[1] = npt
    const_g  = cuda.to_device(const)
    
    #Set up threading -- this needs some real thought.  
    threads_per_block = 512 #for 1080ti the max is 1024, be a multiple of 32
    blocks_per_grid = 256
    
    # threads_per_block = 1024
    # blocks_per_grid_f = (nper + (threads_per_block - 1)) // threads_per_block
    # blocks_per_grid_t = (npt + (threads_per_block - 1)) // threads_per_block
    # blocks_per_grid_nb = int( (nb + (threads_per_block - 1)) // threads_per_block )
    # nper = 10000 #this is the number of frequencies to compute at a time on the GPU, will be GPU-Mem limited.
    
    #rough estimate of memory usage
    size_gb = (np.zeros((nper, npt)).nbytes \
               + np.zeros((nper, int(nb*2) )).nbytes \
               + np.zeros((nper, int(nb*2) ),dtype=np.int32).nbytes) \
    / 1024**3
    
    print(f"Array size: {size_gb:.6f} GB")
    
    #will stored binned phase-folded light-curves that are used by BLS
    y_g     = cuda.device_array(shape=(nper, int(nb*2)), dtype=np.float64)
    ibi_g   = cuda.device_array(shape=(nper, int(nb*2)), dtype=np.int32)
    #configure threads and blocks to zero out the arrays
    rows = nper
    cols = int(nb*2)
    threads_per_block_z = (16, 16)  # 16x16 threads per block (adjust as needed)
    blocks_per_grid_z = ((rows + 15) // 16, (cols + 15) // 16)  # Compute grid size
    
    # create arrays that will contain answers 
    zarray = np.zeros(nstep, dtype=np.float64)
    p_g = cuda.to_device(zarray)
    zarray = []
    zarray = np.zeros(nstep, dtype=np.int32)
    jn1_g = cuda.to_device(zarray)
    jn2_g = cuda.to_device(zarray)
    rn3_g = cuda.to_device(zarray)
    zarray = []
    
    #we break-up the calculations because the GPU can hold all the foldings at once.  ~5-6 GB needed for 10 000 frequencies.
    for i in tqdm(range(0, nstep, nper)):
        idx = np.arange(i, np.min((i+nper,nstep)), 1) #indicies of frequencies we will check
        # print(idx[0],idx[-1])
        idx_g = cuda.to_device(idx)                   #move the indicies to the GPU
        # %timeit 
        init_zeros[blocks_per_grid_z, threads_per_block_z](y_g)   #initialize worker arrays to zero
        init_zeros[blocks_per_grid_z, threads_per_block_z](ibi_g)
        compute_phase[blocks_per_grid, threads_per_block](freqs_g, time_g, flux_g, idx_g, const_g, y_g, ibi_g) #phase-folding
        compute_dup_yibi[blocks_per_grid, threads_per_block](y_g, ibi_g, const_g, idx_g)                       #memory copy
        compute_bls_stat[blocks_per_grid, threads_per_block](p_g, jn1_g, jn2_g, rn3_g, y_g, ibi_g, kmis_g, idx_g, const_g) #BLS
        #cuda.synchronize()
        #input()
    
    del time_g, flux_g, freqs_g, kmis_g, const_g #clean-up the mess on the GPU.
    del idx_g, y_g, ibi_g
    cuda.synchronize()
    
    p = p_g.copy_to_host() #Get the answers
    del p_g                #clean up GPU
    jn1 = jn1_g.copy_to_host()
    jn2 = jn2_g.copy_to_host()
    rn3 = rn3_g.copy_to_host()
    del jn1_g, jn2_g, rn3_g
    cuda.synchronize()     #sync up   

    periods, power, bper, epo, bpower, snr, tdur, depth = \
        calc_eph(p, jn1, jn2, npt, time, flux, freqs, ofac, nstep, nb, mintime, Keptime)
    if gbls_inputs.plots > 0:
        makeplot(periods, power, time, flux, mintime, Keptime, epo, bper, bpower, snr, tdur, depth, \
                filename, gbls_inputs.plots)

    gbls_ans = gbls_ans_class()
    gbls_ans.epo    = epo
    gbls_ans.bper   = bper
    gbls_ans.bpower = bpower
    gbls_ans.snr    = snr
    gbls_ans.tdur   = tdur
    gbls_ans.depth  = depth

    return gbls_ans


#generic routine to read in files
def readfile(filename):

    itime1 = 1765.5/86400.0 #Kepler integration time
    
    i=0
    #data=[] #Create an empty list
    time  = []
    flux  = []
    # ferr  = []
    # itime = []
    f=open(filename)
    for line in f:
        if i>0:
            line = line.strip() #removes line breaks 
            #columns = line.split() #break into columns based on a delimiter.  This example is for spaces
            columns = line.split() #if your data is seperated by commas, use this. 
            #data.append([float(r) for r in columns]) #Append 
            time.append(float(columns[0]))
            flux.append(float(columns[1]))
            # ferr.append(float(columns[2]))
            # itime.append(itime1)
        i+=1
    f.close()
    
    #data = np.array(data)
    time = np.array(time)
    flux = np.array(flux)
    # ferr = np.array(ferr)
    # itime = np.array(itime)
    
    return time, flux# , ferr, itime

@jit(nopython=True)
def nyquest_cpu(time):

    dt = np.zeros((time.shape[0]))
    
    for i in range(time.shape[0]-1):
        dt[i] = time[i+1] - time[i]

    mode = np.median(dt)
    nyq=1.0/(2.0*mode)

    return nyq

@jit(nopython=True)
def qfunc_cpu(f, Mstar, Rstar):
    '''Calculates optimal phase for circular orbit
    '''

    fsec = f/day2sec #Convert from c/d to c/sec 

    q = pifac * Rstar * Rsun / (G * Mstar * Msun)**(1.0/3.0) * fsec**(2.0/3.0)

    return q

@jit(nopython=True)
def calc_nsteps_cpu(Mstar, Rstar, freq1, freq2, nyq, ofac, npt, df0, nb, minbin):
    '''pre-calculate the number of steps to sweep the requested frequency range using optimum steping
    '''

    #Initialize counters
    f = freq1
    nstep = 0

    while(f<freq2):
        nstep+=1
        q = qfunc_cpu(f, Mstar, Rstar) #calculate shift to optimally sample based on expected transit duration
        df = q * df0 #estimate next frequency step based on optimum q
        f = f + df   #next frequency to scan

    return nstep

@jit(nopython=True)
def calc_freqs_cpu(nstep, Mstar, Rstar, freq1, freq2, nyq, ofac, npt, df0, nb, minbin):
    '''pre-calculate the number of steps to sweep the requested frequency range using optimum steping
    '''

    #Initialize arrays
    freqs = np.zeros((nstep))
    qs    = np.zeros((nstep))
    kmis  = np.zeros((3, nstep), dtype = np.int32) # kmi = 0, kma = 1, kkmi = 2
    
    #Initialize counters
    f = freq1
    i = 0
    # freqs[0] = f
    # qs[0]    = qfunc_cpu(f, Mstar, Rstar)

    while(f<freq2):
        qs[i] = qfunc_cpu(f, Mstar, Rstar) #calculate shift to optimally sample based on expected transit duration
        df = qs[i] * df0 #estimate next frequency step based on optimum q
        freqs[i] = f

        #qmi=max(onehour*f,q/2.0)
        qmi = qs[i]/2.0 
        if qmi < onehour*f:
            qmi = onehour*f

        #qma=min(0.05,q*2.0)
        qma = qs[i]*2.0
        if qma > 0.25:
            qma = 0.25

        #if(qma.lt.qmi) qma=min(qmi*2.0,0.4)
        if (qma < qmi):
            qma = qmi*2.0
            if qma > 0.4:
                qma = 0.4

        kmis[0, i] = int(qmi * nb)
        if kmis[0, i] < 0:
            kmis[0, i] = 0
        kmis[1, i] = int(qma * nb) + 1
        kmis[2, i] = int(npt * qmi)
        if kmis[2, i] < minbin:
            kmis[2, i] = minbin
            
        i += 1 #increase counter
        f = f + df   #next frequency to scan
    return freqs, kmis

@jit(nopython=True)
def stats(epo, period, tdur, npt, time, flux):
    # Import necessary libraries if not already imported (e.g., NumPy)

    f   = np.zeros((npt))
    fin = np.zeros((npt))

    tdurfac = 0.5
    
    ph1 = 0.75 - tdurfac * tdur / period
    if ph1 < 0.5:
        ph1 = 0.5
    
    ph2 = 0.75 + tdurfac * tdur / period
    if ph2 > 1.0:
        ph2 = 1.0
    
    toff = 0.75 - (epo / period - int(epo /period))
    if toff < 0.0:
        toff += 1
    
    k = 0  # number of data points out of transit
    kin = 0  # number of data points in transit

    for i in range(npt):
        phase = time[i] / period - int(time[i] / period) + toff
        
        if phase < 0.0:
            phase += 1.0
        elif phase > 1.0:
            phase -= 1.0
        
        if (phase < ph1) or (phase > ph2):
            k += 1
            # Update the out-of-transit flux array with this value
            f[k] = flux[i]
        else:
            kin += 1
            fin[kin] = flux[i]
    
    if k > 3:
        # Calculate standard deviation of out-of-transit data
        std = np.std(f[:k])
        fmean = np.mean(f[:k])
    else:
        fmean = 0.0
        std = 0.0
    
    if kin > 1:
        # Sort in-transit flux array and get median
        sorted_fin = np.sort(fin[:kin])
        depth = -sorted_fin[kin // 2] + fmean
    else:
        depth = fmean
    
    snr = 0.0

    if k > 3 and kin > 1 and std > 0.0:
        snr = depth / std * np.sqrt(kin)
    
    # Output statements can be handled as needed
    
    return fmean, std, depth, snr

def makeplot(periods, power, time, flux, mintime, Keptime, epo, bper, bpower, snr, tdur, depth, filename, plots):

    rcParams.update({'font.size': 12}) #adjust font
    rcParams['axes.linewidth'] = 2.0
    rcParams['font.family'] = 'monospace'

    # Create a figure and three subplots
    fig = plt.figure(figsize=(10,10))
    gs = gridspec.GridSpec(4, 1, height_ratios=[1, 1, 1, 1])

    # Create subplots
    ax1 = plt.subplot(gs[0])
    ax2 = plt.subplot(gs[1])
    ax3 = plt.subplot(gs[2])
    
    # Split the last row into two columns
    gs4 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[3], width_ratios=[1, 1.5])
    
    ax4a = plt.subplot(gs4[0])
    ax4b = plt.subplot(gs4[1])
    

    for i in range(6):

        if i == 0:
            # Top panel : BLS spectrum
            xlabel = "Period (days)"
            ylabel = "BLS Power"
        
            x = periods
            y = power
            
            xmin = np.min(x)
            xmax = np.max(x)
        
            ymin = 0.0 #np.min(y)
            ymax = np.max(y)
            ymax += 0.05*(ymax-ymin)
        
            xlog = 1
            ylog = 0
        
            pstyle = 1

            axn = ax1
            
        elif i == 1:
            # 2nd panel : Time series
            xlabel = "Time (days)"
            ylabel = "Relative Flux"
        
            x = time+mintime-Keptime
            y = flux + 1
            
            xmin = np.min(x)
            xmax = np.max(x)
        
            ymin = np.min(y)
            ymax = np.max(y)
            dy = (ymax-ymin)
            ymax += 0.05 * dy
            ymin -= 0.05 * dy
        
            xlog = 0
            ylog = 0
        
            pstyle = 2
            psize = 1

            axn = ax2

        elif i == 2:
            # 3rd panel : phase plot 

            xlabel = "Phase"
            ylabel = "Relative Flux"
            
            phase = (time+mintime-Keptime-epo)/bper - np.floor((time+mintime-Keptime-epo)/bper)
            phase[phase > 0.25] = phase[phase > 0.25] - 1.0
            # ph1 = 2.0 * tdur/bper

            x = phase
            y = flux + 1

            xmin = np.min(x)
            xmax = np.max(x)
        
            ymin = np.min(y)
            ymax = np.max(y)
            dy = (ymax-ymin)
            ymax += 0.05 * dy
            ymin -= 0.05 * dy

            xlog = 0
            ylog = 0 

            pstyle = 2
            psize = 1

            axn = ax3

        elif i == 3:
            # 4th panel, bottom right
            
            xlabel = "Phase (Hours)"
            ylabel = "Relative Flux"
            
            phase = (time+mintime-Keptime-epo)/bper - np.floor((time+mintime-Keptime-epo)/bper)
            phase[phase > 0.5] = phase[phase > 0.5] - 1.0
            ph1 = 2.0 * tdur/bper

            x = 24*bper*phase[(phase>-ph1)&(phase<ph1)]
            y = flux[(phase>-ph1)&(phase<ph1)]+1

            xmin = np.min(x)
            xmax = np.max(x)
        
            ymin = np.min(y)
            ymax = np.max(y)
            dy = (ymax-ymin)
            ymax += 0.05 * dy
            ymin -= 0.05 * dy

            xlog = 0
            ylog = 0 

            pstyle = 2
            psize = 1

            axn = ax4b

        elif i == 5:
            #Stats panel
            ax4a.set_xticks([])
            ax4a.set_yticks([])
            ax4a.set_frame_on(False)  # Remove the border
            ax4a.set_title("")  # Remove title

            # print("Per = ", bper, "(days)\nEPO = ", epo, "(days)\nPow = ", power, "\nSNR = ", \
            #       snr, "\nTdur = ", tdur*24, "(days)\nTdep = ", depth*1.0e6,"(ppm)")

            ln1 = f"Per  = {bper:.6f} days"
            ln2 = f"T0   = {epo:.6f} days"
            ln3 = f"Pow  = {bpower:.6f}"
            ln4 = f"SNR  = {snr:.1f}"
            ln5 = f"Tdur = {tdur*24:.1f} hours"
            ln6 = f"Tdep = {depth*1.0e6:.1f} ppm"
            lines_of_text = [ln1, ln2, ln3, ln4, ln5, ln6]
            for i, line in enumerate(lines_of_text):
                ax4a.text(0.0, 0.8 - (i/(2+len(lines_of_text))), line, ha='left')

            
    
        axn.tick_params(direction='in', which='major', bottom=True, top=True, \
                           left=True, right=True, length=10, width=2)
        axn.tick_params(direction='in', which='minor', bottom=True, top=True, \
                           left=True, right=True, length=4, width=2)
    
        
        axn.set_xlabel(xlabel)
        axn.set_ylabel(ylabel)
        
        axn.set_xlim(xmin,xmax)
        axn.set_ylim(ymin,ymax)
    
        if xlog == 1:
            axn.set_xscale('log')
        if ylog == 1:
            axn.set_yscale('log')
    
        axn.xaxis.set_major_formatter(ticker.ScalarFormatter())
        axn.yaxis.set_major_formatter(ticker.ScalarFormatter())
    
        if pstyle == 2:
            axn.scatter(x, y, s = psize, color = 'black')
        else:
            axn.plot(x, y, color = 'black')
    
    fig.tight_layout()
    if plots == 2:        
        directory = os.path.dirname(filename)
        filename_without_extension, file_extension = os.path.splitext(os.path.basename(filename))

        if is_writable(directory):
            filename_without_extension = ".".join(filename.split(".")[:-1])
            fig.savefig(filename_without_extension+".png", dpi=150)
        else:
            print("Warning: Path to datafile not writable, attempting to make plot in current directory")
            fig.savefig(filename_without_extension+".png", dpi=150)
        plt.show()

    elif plots == 3:
        directory = os.path.dirname(filename)
        filename_without_extension, file_extension = os.path.splitext(os.path.basename(filename))

        if is_writable(directory):
            filename_without_extension = ".".join(filename.split(".")[:-1])
            fig.savefig(filename_without_extension+".png", dpi=150)
        else:
            print("Warning: Path to datafile not writable, attempting to make plot in current directory")
            fig.savefig(filename_without_extension+".png", dpi=150)
        plt.close(fig)

    else:
        plt.show()
        
def is_writable(directory):
    return os.access(directory, os.W_OK)

@jit(nopython=True)
def running_std_with_filter(data, half_window):
    # Calculate the overall standard deviation
    std = np.std(data)
    
    # Initialize an array to store the running standard deviations
    running_std = np.empty(len(data))
    
    # Iterate over each element in the data
    for i in range(len(data)):
        # Define the start and end indices of the window
        start_idx = max(0, i - half_window)
        end_idx = min(len(data), i + half_window + 1)
        
        # Extract the window and filter values less than 3 * std
        window_data = data[start_idx:end_idx]
        filtered_data = window_data[window_data < 3 * std]
        
        # Calculate the standard deviation of the filtered window
        running_std[i] = np.std(filtered_data)
    
    return running_std
        
def calc_eph(p, jn1, jn2, npt, time, flux, freqs, ofac, nstep, nb, mintime, Keptime):

    periods = 1/freqs # periods (days)

    width = np.min((int(ofac*100)+1,nstep)) # clean up the 1/f ramp from BLS
    filtered = medfilt(np.sqrt(p), kernel_size=width) 

    data = np.sqrt(p) - filtered
    std = np.std(data)
    half_window = width 
    # running_std = np.array([\
    #     np.std(data[max(0, i - half_window): min(len(data), i + half_window + 1)]\
    #            [data[max(0, i - half_window): min(len(data), i + half_window + 1)] < 3 * std] 
    #           )\
    #     for i in range(len(data))\
    # ])
    running_std = running_std_with_filter(data, half_window)

    power = (np.sqrt(p) - filtered)/running_std # This is our BLS statistic array for each frequency/period

    psort = np.argsort(power) #Get sorted indicies to find best event

    bpower = power[psort[-1]]     #Power of best event
    bper   = periods[psort[-1]]   #Period of best event
    
    in1   = jn1[psort[-1]]        #start of ingress (based on BLS, so there is will be some error)
    in2   = jn2[psort[-1]]        #end of egress 

    # apply boundary fixes to in1, in2
    # pad in1,in2 by 1 and make sure we don't step outside [0,nb-1]
    if in2 > nb:
        in2 = in2 - nb

    # pad in1,in2 by 1 and make sure we don't step outside [0,nb-1]
    in1 = in1 - 1
    if in1<0:
        in1=in1+nb
    in2 = in2 + 1
    if in2 > nb - 1: 
        in2 = in2 - nb + 1
    if in1 == in2: 
        in2 = in1 + 1

    # Time of first Transit
    if in1 < in2:
        epo = mintime + bper * ( (in1 + in2)/2.0 / nb)
    else:
        epo = mintime + bper * ( (in1 + in2 - nb + 1) / 2.0 / nb )
    epo = epo - Keptime

    # Transit duration.  (units = period = days)
    if in1 <= in2:
        tdur = bper * (in2 - in1) / nb
    else:
        tdur = bper * (in2 + nb - 1 - in1) / nb

    # s = np.sqrt( p[psort[-1]] * jn2[psort[-1]] * (npt - jn2[psort[-1]]) )
    # depth = -s * npt / ( jn2[psort[-1]] * (npt - jn2[psort[-1]]) )

    #Dirty translation of Fortran from transitfind5
    fmean, std, depth, snr = stats(epo, bper, tdur, npt, time + mintime - Keptime, flux) 

    return periods, power, bper, epo, bpower, snr, tdur, depth

@cuda.jit
def init_zeros(arr):
    """ CUDA kernel to initialize a 2D array to zero """
    x, y = cuda.grid(2)  # Get thread indices for 2D grid
    if x < arr.shape[0] and y < arr.shape[1]:  # Bounds check
        arr[x, y] = np.float64(0.0)

@cuda.jit
def compute_phase(freqs_g, time_g, flux_g, idx_g, const_g, y_g, ibi_g):
    """Fast phase-folding with GPU"""

    start = cuda.grid(1)      # 1 = one dimensional thread grid, returns a single value
    stride = cuda.gridsize(1) # ditto

    # print(start, time_g.shape[0], stride)
    nb1 = const_g[0] - 1
    for k in range(start, time_g.shape[0], stride): #scan over all time-stamps
        for i in range(idx_g.shape[0]):             #scan over frequencies
            idx = idx_g[i]  # Avoid repeated global memory accesses
            phase = time_g[k]*freqs_g[idx] - floor(time_g[k]*freqs_g[idx])
            j = int(phase * nb1) #const[0] = nb
            cuda.atomic.add(ibi_g, (i, j), 1)
            cuda.atomic.add(y_g  , (i, j), flux_g[k])

@cuda.jit
def compute_dup_yibi(y_g, ibi_g, const_g, idx_g):
    """To avoid phase wrapping issues we copy the array"""

    start = cuda.grid(1)      # 1 = one dimensional thread grid, returns a single value
    stride = cuda.gridsize(1) # ditto

    # print(start, const_g[0], stride)
    for k in range(start, const_g[0], stride):
        if k < const_g[0]:
            for i in range(idx_g.shape[0]):
                ibi_g[i, k + const_g[0]] = ibi_g[i, k]
                y_g[i, k + const_g[0]]   = y_g[i, k]

@cuda.jit
def compute_bls_stat(p_g, jn1_g, jn2_g, rn3_g, y_g, ibi_g, kmis_g, idx_g, const_g):
    """Calculate BLS with GPU"""

    start = cuda.grid(1)      # 1 = one dimensional thread grid, returns a single value
    stride = cuda.gridsize(1) # ditto

    # if start == 1:
    #     print('xx', start, idx_g.shape[0], stride)
    for ii in range(start, idx_g.shape[0], stride): #scanning over frequencies
        if ii < idx_g.shape[0]:  
            idx = idx_g[ii]  # Avoid repeated global memory accesses

            # Load kmis_g[:, idx] into registers
            kmis_0 = kmis_g[0, idx] 
            kmis_1 = kmis_g[1, idx]  
            kmis_2 = kmis_g[2, idx]
            
            for i in range(const_g[0]): #scanning over binned (nb) data
                s = 0.0
                k = 0
                kk = 0
                nb2 = i + kmis_1 + 1
                
                for j in range(i, nb2):
                    k  = k + 1
                    kk = kk + ibi_g[ii, j]
                    s  = s  + y_g[ii, j]
                    
                    if k > kmis_0 and kk > kmis_2:
                        dfac = (kk * (const_g[1] - kk)) #const[1] = npt
                        if dfac > 0 :
                            pow1 = s*s / dfac
                            old_max = cuda.atomic.max(p_g, idx, pow1)

                            if pow1 > old_max:
                                cuda.atomic.exch(jn1_g, idx, i)
                                cuda.atomic.exch(jn2_g, idx, j)
                                cuda.atomic.exch(rn3_g, idx, kk)

