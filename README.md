# bls_cuda
An implementation of Box-Least-Squares transit search using CUDA.

How does it work:

The idea is to send a few thousand jobs at a time to the GPU.  We loop through tossing blocks of frequencies to compute BLS for, and collecting the results as we go.  I've tested with Kepler data scanning over 700~000 frequencies from 0.5 days to the length of the dataset without issues.  You can't just assign one frequency per thread or you will quickly run out of VRAM.  

## Getting Started
To get started with this project, follow these steps:
1. Clone the repository: `git clone https://github.com/jasonfrowe/bls_cuda.git`
2. Install required dependencies:  numba, numba-cuda, matplotlib (and others)
3. You can either have a look at the Jupyter Notebook (bls_cuda.ipynb) or try running transitfind5_gpu.py

## Commandline Usage
1. Make sure transitfind5_gpu.py is executable
2. `./transitfind5_gpu.py tlc29991541_5.d.dat`
3. The input file is a simple space delimited file with time and flux in two columns.
4. if everything works, then a plot should show up showing the results of your search
5. try: `./transitfind5_gpu.py` to see command usage

## CPU Version
1. You can use `transitfind5_cpu.py` if you want CPU based BLS.
2. The CPU version will distribute work equally amongst as many CPU cores that are available.  Works very well.
3. For many machines the CPU version will compete with the GPU version.

## Contributing
If you'd like to contribute to this project, go for it! There are a number of to-dos 
1. ~~Code speed can likely be made much faster.  (shared memory vs global memory)~~
2. ~~Better choices of blocks and threads-per-block needs to be explored~~
3. Making the code base into an installable package
4. ~~Make CPU threading more efficient (spread around short-period jobs that take longer)~~
5. and much more.. 

## License
This project is licensed under the GNU General Public License (GPL) version 3 or later.

## Acknowledgments
Thank you to Canada Research Chairs, NSERC Discovery, Digital Alliance Canada, Calcul Quebec, FRQNT for financial and hardware support.

This code was initially developed during the Bishop's University Winter Reading Week, making good use of profession development resources. 

This code is directly adopted from Kovacs et al. 2002 : A box-fitting algorithm in the search for periodic transits 

If you find these codes useful please reference:  
Rowe et al. 2014 ApJ, 784, 45   
Rowe et al. 2015 ApJs, 217, 16  

## Change Log
2025/11/27 : Big refresh of the code base.  First steps to pip installable package
2025/03/08 : Initial Update  
2025/03/09 : Added a 'V2'.  V2 works best with TESS CVZ lc, V1 works best with Kepler.  
2025/03/09 : Added CPU version (Numba + threading)  
2025/03/10 : V2 is now faster for CPU  
