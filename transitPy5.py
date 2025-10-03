import sys
import os

import numpy as np
from astroquery.mast import Observations, Catalogs
from astropy.io import fits

import concurrent.futures

try: # Python 3.x
    from urllib.parse import quote as urlencode
    from urllib.request import urlretrieve
except ImportError:  # Python 2.x
    from urllib import pathname2url as urlencode
    from urllib import urlretrieve

try: # Python 3.x
    import http.client as httplib
except ImportError:  # Python 2.x
    import httplib

import json

# Data processing parameters  
class tpy5_inputs_class:
    def __init__(self):
        self.photfile  = "filename.txt"
        self.photdir   = "/path/to/photometry/"
        self.roi       = 210.01 # Roman ID
        self.boxbin    = 2.0    # Detrending window
        self.gapsize   = 0.5    # Detection of gaps in the data -- we do not detrend over gaps
        self.nfitp     = 2      # Order of polynomial for detrending.  2 = quadratic
        self.dsigclip  = 3.0    # Sigma clipping for derivative routine
        self.nsampmax  = 6      # Sample size for derivative routine
        self.detrended = 0      # Track if detrended data is used/created
        self.dataclip  = 0      # Track if clipped data is used/created 
        self.fstd_cut  = 5      # Simple Sigma-clipping
        self.rjd       = 2461345.5 # time offset for the roman JD used in the simulations
        self.mstar     = 1.0    # Mass of star   [Msun]
        self.rstar     = 1.0    # Radius of star [Rsun]
        self.teff      = 5777   # Temperature of star [K]
        self.logg      = 4.5    # Star gravity [cgs]
        self.feh       = 0.0    # star metalicticity 
        self.e_mstar   = 0.3    # Error in stellar mass [Msun]
        self.e_rstar   = 0.3    # Error in stellar radius [Rsun]
        self.e_teff    = 500    # Error in Teff [K]
        self.e_logg    = 0.2    # Error in log(g) [cgs]
        self.e_feh     = 0.3    # Error in FeH 

class exocat_class:
    def __init__(self):
        self.ticid=[]
        self.toiid=[]
        self.toiid_str=[]
        self.ra=[]
        self.dec=[]
        self.tmag=[]
        self.t0=[]
        self.t0err=[]
        self.per=[]
        self.pererr=[]
        self.tdur=[]
        self.tdurerr=[]
        self.tdep=[]
        self.tdeperr=[]

class phot_class:
    def __init__(self):
        self.time=[]  #initialize arrays
        self.flux=[]
        self.ferr=[]
        self.itime=[]

class catalogue_class:
    def __init__(self):
        #IDs
        self.tid=[]  #Catalogue ID [0]
        self.toi=[]  #KOI [1]
        self.planetid=[] #Confirmed planet name (e.g., Kepler-20b)
        #model parameters
        self.rhostarm=[] #rhostar model [31]
        self.rhostarmep=[] #+error in rhostar [32]
        self.rhostarmem=[] #-error in rhostar [33]
        self.t0=[] #model T0 [4]
        self.t0err=[] #error in T0 [5]
        self.per=[] #period [2]
        self.pererr=[] #error in period [3]
        self.b=[] #impact parameter [9]
        self.bep=[] #+error in b [10]
        self.bem=[] #-error in b [11]
        self.rdrs=[] #model r/R* [6]
        self.rdrsep=[] #+error in r/R* [7]
        self.rdrsem=[] #-error in r/R* [8]
        #stellar parameters
        self.rstar=[] #stellar radius [39]
        self.rstar_ep=[] #stellar radius +err [40]
        self.rstar_em=[] #stellar radius -err [41]
        self.teff=[] #Teff [37]
        self.teff_e=[] #Teff error [38]
        self.rhostar=[] #rhostar [34]
        self.rhostar_ep=[] #rhostar +err [35]
        self.rhostar_em=[] #rhostar -err [36]
        self.logg=[] #stellar radius [45]
        self.logg_ep=[] #stellar radius +err [46]
        self.logg_em=[] #stellar radius -err [47]
        self.feh=[] #metallicity [48]
        self.feh_e=[] #metallicity error [49]
        self.q1=[] #limb-darkening
        self.q1_e =[]
        self.q2=[] #limb-darkening
        self.q2_e =[]
        #disposition
        self.statusflag=[]

def safe_float_conversion(s, default=0.0):
    if s is None or s == '':
        #print(f"Input is None or empty. Using default value {default}.")
        return default
    try:
        return float(s)
    except ValueError as e:
        print(f"Error converting to float: {e}")
        return default

# Read 'toi_file' (from NASA EA -- new table)
def readtoicsv(toi_file):
    exocat=exocat_class() #set up class
    f=open(toi_file)

    icount=0
    for line in f:
        line = line.strip()
        row = line.split(',') #break into columns
        if row[0][0]!='#':
            #skip comments
            icount+=1
            if icount>1:
                #skip header
                exocat.ticid.append(int(float(row[2])))
                exocat.toiid.append(float(row[0]))
                exocat.toiid_str.append(row[0])
                exocat.ra.append(safe_float_conversion(row[7]))
                exocat.dec.append(safe_float_conversion(row[11]))
                exocat.tmag.append(float(row[59]))

                if row[25]=='':
                    exocat.t0.append(-1.0)
                else:
                    try:
                        exocat.t0.append(float(row[24]) - 2.457E6) # Planet Transit Midpoint Value [BJD]
                    except:
                        print(row[25])

                if row[26]=='': exocat.t0err.append(-1.0)
                else: exocat.t0err.append(float(row[25])) # Planet Transit Midpoint Upper Unc [BJD]

                if row[30]=='': exocat.per.append(-1.0)
                else: exocat.per.append(float(row[29])) # Planet Orbital Period Value [days]

                if row[31]=='': exocat.pererr.append(-1.0)
                else: exocat.pererr.append(float(row[30])) # Planet Orbital Period Upper Unc [days]

                if row[35]=='': exocat.tdur.append(-1.0)
                else: exocat.tdur.append(float(row[34])) # Planet Transit Duration Value [hours]

                if row[36]=='': exocat.tdurerr.append(-1.0)
                else: exocat.tdurerr.append(float(row[35])) # Planet Transit Duration Upper Unc [hours]

                if row[40]=='': exocat.tdep.append(-1.0)
                else: exocat.tdep.append(float(row[39])) # Planet Transit Depth Value [ppm]

                if row[41]=='': exocat.tdeperr.append(-1.0)
                else: exocat.tdeperr.append(float(row[40])) # Planet Transit Depth Upper Unc [ppm]
    f.close()

    return exocat

def get_tess_data(u_ticid,max_flag=16,out_dir='./download'):
    """Given a TIC-ID, return time,flux,ferr,itime
    u_ticid : (int) TIC ID

    returns lc_time,flux,ferr,int_time
    """
    tic_str='TIC' + str(u_ticid)
    #out_dir='/data/rowe/TESS/download/'

    # Search MAST for TIC ID
    print('Searching MAST for TIC ID (this can be slow)')
    obs_table=Observations.query_object(tic_str,radius=".002 deg")

    # Identify TESS timeseries data sets (i.e. ignore FFIs)
    print("Identifying TESS timeseries data sets")
    oti=(obs_table["obs_collection"] == "TESS") & \
            (obs_table["dataproduct_type"] == "timeseries")
    if oti.any() == True:
        data_products=Observations.get_product_list(obs_table[oti])
        dpi=[j for j, s in enumerate(data_products["productFilename"]) if "lc.fits" in s]
        manifest=Observations.download_products(data_products[dpi],download_dir=out_dir)
    else:
        manifest=[]

    lc_time=[]
    flux=[]
    ferr=[]
    int_time=[]
    for j in range(0,len(manifest)):
        fits_fname=str(manifest["Local Path"][j])
        #print(fits_fname)
        hdu=fits.open(fits_fname)
        tmp_bjd=hdu[1].data['TIME']
        tmp_flux=hdu[1].data['PDCSAP_FLUX']
        tmp_ferr=hdu[1].data['PDCSAP_FLUX_ERR']
        tmp_int_time=hdu[1].header['INT_TIME'] + np.zeros(len(tmp_bjd))
        tmp_flag=hdu[1].data['QUALITY']

        ii=(tmp_flag <= max_flag) & (~np.isnan(tmp_flux))
        tmp_bjd=tmp_bjd[ii]
        tmp_flux=tmp_flux[ii]
        tmp_ferr=tmp_ferr[ii]
        tmp_int_time=tmp_int_time[ii]
        # Shift flux measurements
        median_flux=np.median(tmp_flux)
        tmp_flux=tmp_flux / median_flux
        tmp_ferr=tmp_ferr / median_flux
        # Append to output columns
        lc_time=np.append(lc_time,tmp_bjd)
        flux=np.append(flux,tmp_flux)
        ferr=np.append(ferr,tmp_ferr)
        int_time=np.append(int_time,tmp_int_time)

        hdu.close()

    # Sort by time
    si=np.argsort(lc_time)
    lc_time=np.asarray(lc_time)[si]
    flux=np.asarray(flux)[si]
    ferr=np.asarray(ferr)[si]
    int_time=np.asarray(int_time)[si]

    phot=phot_class()
    phot.time=np.copy(lc_time)
    phot.flux=np.copy(flux)
    phot.ferr=np.copy(ferr)
    phot.itime=np.copy(int_time)    

    phot.itime=phot.itime/(60*24) #convert minutes to days

    phot.tflag = np.zeros((phot.time.shape[0])) # pre-populate array to mark transit data (=1 when in transit)
    phot.flux_f = []                            # placeholder for detrended data
    phot.icut = np.zeros((phot.time.shape[0]))  # cutting data (0==keep, 1==toss)

    return phot

def mastQuery(request):

    server='mast.stsci.edu'

    # Grab Python Version
    version = ".".join(map(str, sys.version_info[:3]))

    # Create Http Header Variables
    headers = {"Content-type": "application/x-www-form-urlencoded",
        "Accept": "text/plain",
        "User-agent":"python-requests/"+version}

    # Encoding the request as a json string
    requestString = json.dumps(request)
    requestString = urlencode(requestString)

    # opening the https connection
    conn = httplib.HTTPSConnection(server)

    # Making the query
    conn.request("POST", "/api/v0/invoke", "request="+requestString, headers)

    # Getting the response
    resp = conn.getresponse()
    head = resp.getheaders()
    content = resp.read().decode('utf-8')

    # Close the https connection
    conn.close()

    return head,content

def ticAdvancedSearch(id):
    request = {"service":"Mast.Catalogs.Filtered.Tic",
                "format":"json",
                "params":{
                "columns":"*",
                "filters":[
                    {"paramName":"id",
                        "values":[{"min":id,"max":id}]}]
                        #"values":[{261136679}]}]
                }}

    headers,outString = mastQuery(request)
    outData = json.loads(outString)

    return outData

def populate_catalogue(tic_output, exocat, toi_index):

    koicat = catalogue_class()
    
    koicat.tid.append(exocat.ticid[toi_index])
    koicat.toi.append(exocat.toiid[toi_index])
    koicat.planetid.append(" ")
    
    koicat.t0.append(exocat.t0[toi_index])
    koicat.t0err.append(exocat.t0err[toi_index])
    
    koicat.per.append(exocat.per[toi_index])
    koicat.pererr.append(exocat.pererr[toi_index])
    
    koicat.b.append(0.4)   #This is a guess because it is not populated.
    koicat.bep.append(0.1)
    koicat.bem.append(-0.1)
    
    koicat.rdrs.append(np.sqrt(exocat.tdep[toi_index]/1.0e6))
    koicat.rdrsep.append(0.001)
    koicat.rdrsem.append(-0.001)
    
    if tic_output['data'][0]['rad'] == None:
        print('Warning: No R* Available, using Sun')
        koicat.rstar.append(1)
        koicat.rstar_ep.append(0.5)
        koicat.rstar_em.append(0.5)
    else:
        koicat.rstar.append(tic_output['data'][0]['rad'])
        e_rad = tic_output['data'][0]['e_rad'] if tic_output['data'][0]['e_rad'] is not None else 0
        koicat.rstar_ep.append(e_rad)
        koicat.rstar_em.append(-e_rad)
    
    if tic_output['data'][0]['Teff'] == None:
        print('Warning: No Teff Available, using Sun')
        koicat.teff.append(5777)
        koicat.teff_e.append(500)
    else:
        koicat.teff.append(tic_output['data'][0]['Teff'])
        koicat.teff_e.append(tic_output['data'][0]['e_Teff'])
    
    if tic_output['data'][0]['rho'] == None:
        print('Warning: No rhostar Available, using 1.0')
        koicat.rhostar.append(1.0)
        koicat.rhostar_ep.append(0.5)
        koicat.rhostar_em.append(-0.5)
        koicat.rhostarm.append(1.0)
        koicat.rhostarmep.append(0.1)
        koicat.rhostarmem.append(-0.1)
    else:
        koicat.rhostar.append(tic_output['data'][0]['rho'])
        e_rho = tic_output['data'][0]['e_rho'] if tic_output['data'][0]['e_rho'] is not None else 0
        koicat.rhostar_ep.append(e_rho)
        koicat.rhostar_em.append(-e_rho)
        koicat.rhostarm.append(tic_output['data'][0]['rho'])
        koicat.rhostarmep.append(0.1)
        koicat.rhostarmem.append(-0.1)
    
    if tic_output['data'][0]['logg'] == None:
        koicat.logg.append(4.5)
        koicat.logg_ep.append(0.5)
        koicat.logg_em.append(-0.5)
        print('Warning: No log(g) Available, using 4.5')
    else:
        koicat.logg.append(tic_output['data'][0]['logg'])
        e_logg = tic_output['data'][0]['e_logg'] if tic_output['data'][0]['e_logg'] is not None else 0
        koicat.logg_ep.append(e_logg)
        koicat.logg_em.append(-e_logg)
    
    koicat.feh.append(0.0)
    koicat.feh_e.append(1.0)
    
    koicat.q1.append(0.5)
    koicat.q1_e.append(0.5)
    
    koicat.q2.append(0.5)
    koicat.q2_e.append(0.5)
    
    koicat.statusflag.append('P')

    return koicat

def get_data_and_catalogues(tpy5_inputs):

    exocat=readtoicsv(tpy5_inputs.toifile)
    toi_index = [j for j, x in enumerate(exocat.toiid) if x == tpy5_inputs.toi][0]
    print('TIC ID: ',exocat.ticid[toi_index],'| TOI ID: ',exocat.toiid[toi_index])
    
    #Get SC Lightcurve for MAST
    #Each Sector/Quarter of data is median corrected independently, then concatentated together.
    phot_SC=get_tess_data(exocat.ticid[toi_index])  #give the TIC_ID and return SC data.

    #Get Stellar parameters from MAST
    tic_output = ticAdvancedSearch(exocat.ticid[toi_index])

    toicat = populate_catalogue(tic_output, exocat, toi_index)

    return toi_index, phot_SC, toicat

def calc_meddiff(npt, x):
    """
    Used by cutoutliers to calculation distribution of derivatives
    """

    dd = np.zeros(npt - 1)

    for i in range(npt - 1):
        dd[i] = np.abs(x[i] - x[i+1])

    meddiff = np.median(dd)
    # p = np.argsort(dd)
    # meddiff =dd[p[int((npt-1)/2)]]

    return meddiff

def run_cutoutliers(phot, tpy5_inputs):

    phot.icut = cutoutliers(phot.time, phot.flux, tpy5_inputs.nsampmax, tpy5_inputs.dsigclip)
    tpy5_inputs.dataclip = 1 
    
def cutoutliers(x, y, nsampmax = 3, sigma = 3.0):
    """
    Uses derivatives to cut outliers
    """

    threshold = 0.0005 #Fixed threshold, if sigma <= 0
    # nsampmax  = 3      #number of +/- nearby samples to use for stats
    # sigma     = 3.0    #sigma cut level

    npt   = x.shape[0]
    icut  = np.zeros((npt), dtype = np.int32)
    samps = np.zeros((nsampmax * 2 + 1))

    for i in range(1,npt-1):

        i1 = np.max((0,       i - nsampmax))
        i2 = np.min((npt - 1, i + nsampmax))

        nsamp = i2 - i1 + 1
        samps[0:nsamp] = y[i1:i2+1]
        # print(i1, i2, samps)

        std = np.median(np.abs(np.diff(samps)))

        if sigma > 0:
            threshold = std * sigma

        vp = y[i] - y[i+1]
        vm = y[i] - y[i-1]

        if  (np.abs(vp) > threshold) and (np.abs(vm) > threshold) and (vp/vm > 0):
            icut[i] = 1 #cut

    return icut

def run_polyfilter(phot_SC, tpy5_inputs):

    phot_SC.flux_f = polyfilter(phot_SC.time, phot_SC.flux, phot_SC.ferr, phot_SC.tflag, \
                                tpy5_inputs.boxbin, tpy5_inputs.nfitp, tpy5_inputs.gapsize)

    tpy5_inputs.detrended = 1 # Mark that we have detrended data


def polyfilter(time, flux, ferr, tflag, boxbin, nfitp, gapsize, multipro = 1):
    """
    Computes detended data based on Kepler TRANSITFIT5 routines

    Parameters:
    time (np.array) : times series, usually in days
    flux (np.array) : relative flux
    ferr (np.array) : uncertanity on relative flux
    boxbin (float)  : full width of detrending window.  Units should match time.
    nfitp  (int)    : order of polynomial fit
    gapsize (float) : identifying boundaries to not detrend across.  Units should match time.
    multipro (int)  : if == 0, single thread operation.  if == 1, multithreading.

    Returns:
    flux_f (np.array) : detrended time-series
    """

    bbd2=boxbin/2.0  #pre-calculate the half-box width

    ts = np.argsort(time) #get indicies for data (we don't assume it's sorted) 

    ngap = 0
    gaps = [] 
    npt = time.shape[0]
    
    for i in range(npt-1):
        if time[ts[i+1]] - time[ts[i]] > gapsize:
            ngap += 1
            g1 = (time[ts[i+1]] + time[ts[i]])/2.0
            gaps.append(g1)
    gaps = np.array(gaps)

    # print("Number of gaps detected: ", ngap)

    if multipro == 0:
        # Single thread version
        offset = np.zeros((npt))
        for i in range(npt):
            offset[ts[i]] = polyrange(ts[i], time, flux, ferr, tflag, nfitp, ngap, gaps, bbd2)

    else:
    
        max_processes = os.cpu_count()
        ndiv = npt // (max_processes - 1) + 1
        offset = np.zeros((max_processes, ndiv))
    
        iarg = np.zeros((max_processes, ndiv), dtype = int)
        for i in range(0, max_processes):
            for k,j in enumerate(range(i, npt, max_processes)):
                iarg[i, k] = j
        
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_processes) as executor:
                futures = {executor.submit(compute_polyrange, iarg[i], time, flux, ferr, tflag, nfitp, \
                                           ngap, gaps, bbd2, ndiv): i for i in range(iarg.shape[0])}
                
                for future in concurrent.futures.as_completed(futures):
                    i = futures[future]
                    try:
                        result = future.result()
                        offset[i] = result
                    except Exception as exc:
                        print(f'Generated an exception: {exc}')
        
        offset = offset.T.ravel()[0:npt]
    
    flux_f = flux - offset + np.median(flux)

    return flux_f

def compute_polyrange(iarg, time, flux, ferr, tflag, nfitp, ngap, gaps, bbd2, ndiv):

    offset = np.zeros((ndiv))

    j = 0
    for i in iarg:
        offset[j] = polyrange(i, time, flux, ferr, tflag, nfitp, ngap, gaps, bbd2)
        j += 1

    return offset

def polyrange(i, time, flux, ferr, tflag, nfitp, ngap, gaps, bbd2):
    """
    Computes mask for time series
    """

    tzero = time[i]
    t1 = tzero - bbd2
    t2 = tzero + bbd2

    #Check if there is a gap in our time-span
    for j in range(ngap):
        if (gaps[j] > t1) and (gaps[j] < tzero):
            t1 = gaps[j]
        if (gaps[j] < t2) and (gaps[j] > tzero):
            t2 = gaps[j]

    #Get data inside time-span that is not a transit.
    tmask = (time > t1) & (time < t2) & (tflag == 0)

    x = time[tmask]
    y = flux[tmask]
    z = ferr[tmask]

    npt2 = x.shape[0]
    # print(t1, t2, npt2, tzero)

    if npt2 > nfitp + 1:
        offset1 = polydetrend(x, y, z, nfitp, tzero)
    elif npt2 == 0:
        offset1 = 0.0
    else:
        offset1 = np.mean(y)
    
    
    return offset1

def polydetrend(x, y, z, nfitp, x_c):

    # Make these command-line parameters
    maxiter = 10
    sigcut = 4.0

    x_centred = x - x_c
    
    ans = np.polyfit(x_centred, y, nfitp , w=1/z)
    poly_func = np.poly1d(ans)
    y_pred = poly_func(x_centred)
    chisq = np.sum(((y - y_pred) / z) ** 2)

    offset = poly_func(0.0)

    # Iteratively remove outliers and refit
    ii = 0       #count the number of iterations
    dchi = 1     #we will interate to reduce chi-sq
    ochi = chisq 
    
    while (ii < maxiter) and (dchi > 0.1):
        model = poly_func(x_centred)
        residuals = y - model
        
        std = np.std(residuals)
        mask = np.abs(residuals) < sigcut * std
        x_centred_filtered = x_centred[mask]
        y_filtered = y[mask]
        z_filtered = z[mask]
        
        coeffs = np.polyfit(x_centred_filtered, y_filtered, nfitp, w=1/z_filtered)

        # Create a polynomial function from the coefficients
        poly_func = np.poly1d(coeffs)
        
        # Calculate predicted values
        y_pred = poly_func(x_centred_filtered)

        # Calculate offset (this is removed from the data)
        offset = poly_func(0.0)
        
        # Calculate chi-squared
        chisq = np.sum(((y_filtered - y_pred) / z_filtered) ** 2)
        
        dchi = abs(chisq - ochi)
        ochi = chisq
        
        ii += 1

    return offset

#generic routine to read in photometry
def readphot(filename):

    itime1 = 1765.5/86400.0 #Kepler integration time
    
    i=0

    phot = phot_class()
    
    f=open(filename)
    for line in f:
        if i>0:
            line = line.strip() #removes line breaks 
            columns = line.split() #break into columns based on a delimiter.  This example is for spaces
            phot.time.append(float(columns[0]))
            phot.flux.append(float(columns[1]))
            if len(columns) >= 3:
                phot.ferr.append(float(columns[2]))
            if len(columns) >= 4:
                phot.itime.append(float(columns[3]))
        i+=1
    f.close()
    
    phot.time = np.array(phot.time)
    phot.flux = np.array(phot.flux)
    if len(phot.ferr) > 0:
        phot.ferr = np.array(phot.ferr)
    else:
        phot.ferr = np.ones((phot.flux.shape[0]))*np.std(phot.flux)
    if len(phot.itime) > 0:
        phot.itime = np.array(phot.itime)
    else:
        phot.itime = np.median(np.diff(phot.time))*np.ones((phot.time.shape[0]))

    phot.tflag = np.zeros((phot.time.shape[0])) # pre-populate array to mark transit data (=1 when in transit)
    phot.flux_f = []                            # placeholder for detrended data
    phot.icut = np.zeros((phot.time.shape[0]))  # cutting data (0==keep, 1==toss)
    
    return phot

#generic routine to read in files
def readbaddata(filename):
    
    i=0
    #data=[] #Create an empty list
    bad_cadence  = []
    bad_time  = []
    # ferr  = []
    # itime = []
    f=open(filename)
    for line in f:
        if i>0:
            line = line.strip() #removes line breaks 
            columns = line.split() #break into columns based on a delimiter.  This example is for spaces
            bad_cadence.append(int(columns[0]))
            bad_time.append(float(columns[1]))

        i+=1
    f.close()

    bad_cadence = np.array(bad_cadence)
    bad_time = np.array(bad_time)
    
    return bad_cadence, bad_time

# Find the closest values
def find_closest(array_list, array_values):
    # Reshape array_list to a column vector for broadcasting
    diff_matrix = np.abs(array_list[:, None] - array_values)
    
    # Get indices of minimum differences along axis 1 (columns)
    closest_indices = np.argmin(diff_matrix, axis=1)
    bad_diffs = array_list - array_values[closest_indices]

    return closest_indices, bad_diffs