import h5py

with h5py.File("dataset.h5") as f:
  min = 999999
  name = ""
  for dname, d in f["data"].items():
    if d.shape[0] < min:
      min = d.shape[0]
      name = dname

print(name, min)