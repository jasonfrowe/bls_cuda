import time
import cudaFiles.bls_cuda_v2 as gbls
from numba import cuda

gbls_inputs = gbls.gbls_inputs_class()
tic = 29991541
gbls_inputs.lcdir    = "/opt/data2/TESS/ffisearch/cvzsearch_yr1/cvz1pt003/tlc"+str(tic)+"/"
gbls_inputs.filename = "tlc"+str(tic)+"_5.dn.dat"
gbls_inputs.zerotime = 1325.0
gbls_inputs.plots = 0 # Disable plots so it doesn't block

cuda.synchronize()
start = time.time()
gbls_ans = gbls.bls(gbls_inputs)
cuda.synchronize()
end = time.time()

print(f"Wall time: {end - start:.2f} s")
