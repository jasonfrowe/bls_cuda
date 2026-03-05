import json

with open("cudaFiles/bls_cuda_v2.ipynb", "r") as f:
    nb = json.load(f)

for cell in nb["cells"]:
    if cell["cell_type"] == "code" and "gbls_ans = gbls.bls(gbls_inputs)" in "".join(cell["source"]):
        source = cell["source"]
        if source[0] != "%%time\n":
            source.insert(0, "%%time\n")

with open("cudaFiles/bls_cuda_v2.ipynb", "w") as f:
    json.dump(nb, f, indent=1)

print("Notebook updated!")
