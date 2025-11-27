#!/usr/bin/env python3

import sys
import bls_cuda_v2 as gbls

def print_usage():
    
    print("Usage: transitfind5_gpu <filename> [nplot] [freq1] [freq2] [Msun] [Rsun]\n\
   <filename> - photometry (time,flux)\n\
   [nplot] - 0: no plot, 1:xwindow 2:png+x 3:png\n\
   [Per1] - low Period (days) to scan, -1 for default\n\
   [Per2] - high Period (days) to scan, -1 for default\n\
   [Msun]  - mass of host star (Msun) \n\
   [Rsun]  - radius of host star (Rsun)\n\
    ")

    exit(0)

# Get the command-line arguments
args = sys.argv

nargs = len(args)
# print("Number of arguments: ", nargs)

if nargs <= 1:
    print_usage()

gbls_inputs = gbls.gbls_inputs_class()

#Get filename
gbls_inputs.filename = args[1]

if nargs > 2:
    arg = int(args[2])
    if (arg >= 0) & (arg < 4):
        gbls_inputs.plots = int(args[2])
    else:
        "[nplot] must be either 0, 1, 2 or 3."

if nargs > 3:
    arg = float(args[3])
    if arg > 0:
        gbls_inputs.freq2 = 1/arg

if nargs > 4:
    arg = float(args[4])
    if arg > 0:
        gbls_inputs.freq1 = 1/arg

if gbls_inputs.freq1 > gbls_inputs.freq2:
    print("Per2 must be greater than Per1")
    exit(-1)

if nargs > 5:
    arg = float(args[5])
    if arg > 0:
        gbls_inputs.Mstar = arg
    elif (arg != -1):
        print("Mstar must be greater than zero")

if nargs > 6:
    arg = float(args[6])
    if arg > 0:
        gbls_inputs.Rstar = arg
    else:
        print("Rstar must be greater than zero")

gbls_ans = gbls.bls(gbls_inputs)

#Dump output to stdout.
print(gbls_ans.bper, gbls_ans.epo, gbls_ans.bpower, gbls_ans.snr, gbls_ans.tdur, gbls_ans.depth)

exit(0)
