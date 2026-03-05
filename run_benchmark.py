import subprocess
import time

start = time.time()
subprocess.run(["jupyter", "nbconvert", "--to", "notebook", "--execute", "cudaFiles/bls_cuda_v2.ipynb", "--output", "bls_output.ipynb"], env={"PATH": "/home/rowe/Documents/python/bls_cuda/.venv/bin:/usr/bin"})
end = time.time()
print(f"Total Notebook Execution Time: {end - start:.2f} seconds")
