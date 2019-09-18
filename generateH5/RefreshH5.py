import h5py
import datetime

# fwrite = h5py.File("I:/NYC/CitiBike/output/NYCBike_50x50_8slice.h5", "w")
# fwrite.create_dataset("data", (4367*8, 2, 50, 50), 'i')


fwrite = h5py.File("I:/NYC/CitiBike/output/NYCBike_50x50_1slice.h5", "w")
fwrite.create_dataset("data", (4368, 2, 50, 50), 'i')


fwrite.close()