import time
import numpy as np

# Import both Float64 and Float32 versions
import cudaFiles.bls_cuda_v2 as gbls64
import cudaFiles.bls_cuda_v2_f32 as gbls32
from numba import cuda

def run_benchmark(dataset_name, lcdir, filename, zerotime):
    print(f"\n=========================================")
    print(f"Dataset: {dataset_name}")
    print(f"=========================================")
    
    # Initialize inputs
    inputs64 = gbls64.gbls_inputs_class()
    inputs64.lcdir = lcdir
    inputs64.filename = filename
    inputs64.zerotime = zerotime
    inputs64.plots = 0
    
    inputs32 = gbls32.gbls_inputs_class()
    inputs32.lcdir = lcdir
    inputs32.filename = filename
    inputs32.zerotime = zerotime
    inputs32.plots = 0
    
    # Run FP64 Baseline
    print("--- FP64 (Baseline) ---")
    cuda.synchronize()
    start_time = time.time()
    ans64 = gbls64.bls(inputs64)
    cuda.synchronize()
    fp64_time = time.time() - start_time
    
    # Run FP32 Precision limits
    print("--- FP32 (Precision Test) ---")
    cuda.synchronize()
    start_time = time.time()
    ans32 = gbls32.bls(inputs32)
    cuda.synchronize()
    fp32_time = time.time() - start_time

    # Results
    print("\n[ RESULTS ]")
    print(f"Execution Time | FP64: {fp64_time:.3f}s | FP32: {fp32_time:.3f}s | Speedup: {fp64_time/fp32_time:.2f}x")
    print(f"Peak Power     | FP64: {ans64.bpower:.3f} | FP32: {ans32.bpower:.3f}")
    print(f"Reported SNR   | FP64: {ans64.snr:.3f} | FP32: {ans32.snr:.3f}")
    print(f"Best Period    | FP64: {ans64.bper:.5f} | FP32: {ans32.bper:.5f}")

if __name__ == "__main__":
    # Test 1: TESS Lightcurve
    tic = 29991541
    lcdir = "/opt/data2/TESS/ffisearch/cvzsearch_yr1/cvz1pt003/tlc" + str(tic) + "/"
    filename = "tlc" + str(tic) + "_5.dn.dat"
    run_benchmark("TESS tlc29991541", lcdir, filename, 1325.0)
    
    # Test 2: Kepler Lightcurve
    run_benchmark("Kepler klc11446443", "/opt/data2/rowe/Kepler/Kepler_n/koi1.n/", "klc11446443.dc.dat", 0.0)
