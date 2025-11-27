# pytfit5

**An implementation of TransitFit5 and BLS in Python (with Numba).**

This package provides tools for transit search and fitting, including Box-Least-Squares (BLS) algorithms optimized for both CPU and GPU (via CUDA).

## Installation

### Prerequisites
*   Python 3.8+

### Installing from Source
It is recommended to install this package in "editable" mode inside a virtual environment. This allows you to modify the source code without needing to reinstall.

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/jasonfrowe/bls_cuda.git
    cd bls_cuda
    ```

2.  **Install via pip:**
    ```bash
    pip install -e .
    ```

## Usage

### Python Import (Recommended)
You can import the package and access the submodules (BLS, transitfit5, MCMC, etc.) directly.

```python
import pytfit5 as pytfit5

# Access submodules using the built-in aliases:
# pytfit5.gbls   -> bls_cpu
# pytfit5.tpy5   -> transitPy5
# pytfit5.kep    -> keplerian
# pytfit5.tmcmc  -> transitmcmc
```

### Jupyter Notebook Example
For a full demonstration of how to run the code, please refer to the example notebook included in this repository:

**`pytfit5_example.ipynb`**

### ⚠️ Deprecation Notice
**Command Line Usage:** Previous versions of this code allowed for execution via command line scripts. This method is **deprecated**. Please use the Python API as described above.

## Dependencies
This package requires the following libraries (installed automatically):
*   `numpy`
*   `matplotlib`
*   `tqdm`
*   `numba`
*   `scipy`
*   `pandas`
*   `astroquery`

## License
This project is licensed under the **GNU General Public License v3.0**. See the [LICENSE](LICENSE) file for details.

## Contact
Jason Rowe - jason@jasonrowe.org

## Contributing
If you'd like to contribute to this project, go for it! There are a number of to-dos 
1. ~~Code speed can likely be made much faster.  (shared memory vs global memory)~~
2. ~~Better choices of blocks and threads-per-block needs to be explored~~
3. ~~Making the code base into an installable package~~
4. ~~Make CPU threading more efficient (spread around short-period jobs that take longer)~~
5. Optimizing GPU memory transfers
6. Allow transit modelling to have different parameters for different planets
7. Better examples for TTV fitting
8. and much more.. 

## License
This project is licensed under the GNU General Public License (GPL) version 3 or later.

## Acknowledgments
Thank you to Canada Research Chairs, NSERC Discovery, Digital Alliance Canada, Calcul Quebec, FRQNT for financial and hardware support.

This code was initially developed during the Bishop's University Winter Reading Week, making good use of profession development resources. 

The implementation of [TransitFit5](https://github.com/jasonfrowe/Kepler) in Python was developed by Alexis Roy (Universite de Sherbrooke) supported by an NSERC Undergraduate Student Research Award (USRA) and iREx Trottier Fellowship.

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
