import numpy as np
import h5py

import glob
import os
import os.path


def dat_to_h5(f_in, f_out):
    series = np.genfromtxt(f_in, delimiter=' ')
    dt = series[1, 0] - series[0, 0]
    values = series[:, 1]
    N = values.size

    h5 = h5py.File(f_out,libver='latest')
    h5.attrs['dt']=dt
    h5.create_dataset('values',(N,),maxshape=(None,),chunks=True)
    h5['values'][0:N] = values
    h5.close()


input_files = glob.glob('../simulations_rr1/*/output_*/*-cell*.dat')
N_f = len(input_files)
for k, f in enumerate(input_files):
    f_out = f[:-3] + 'h5' # Remove the 'dat' extension and replace it by 'h5'.

    # If the .h5 file does not exist yet, create it.
    if not (os.path.isfile(f_out)):
        print 'Processing file %d of %d; %s.' % (k, N_f, f)
        dat_to_h5(f, f_out)

    # Remove the .dat file.
    os.remove(f)

