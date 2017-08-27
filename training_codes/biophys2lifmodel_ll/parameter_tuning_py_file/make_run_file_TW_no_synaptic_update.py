# import numpy as np
from syn_weight_amp_TW import syn_weight_amp
from sys import argv
from os import makedirs
from os.path import exists
# from plot_tot_firing_rate_comparison import plot_tot_firing_rate_comparison
num_core = 30
tstop = 500
f_name = 'll2_g8_8_test%dms_no_con_lif_syn_z' % (tstop)
ref_file = 'output_ll2_g8_8_test%dms_no_con_syn_z001/tot_f_rate.dat' % (tstop)


def main(idx_syn):
    with open('run_compile.bat', 'w') as output_file:
        output_folder = 'output_' + f_name + str(idx_syn)
        if not exists(output_folder):
            makedirs(output_folder)
        syn_weight_amp(output_folder, str(idx_syn))
        sfile = f_name + str(idx_syn)
        output_file.write('mpiexec -np %d nrniv -mpi run_%s.py > output_%s/log.txt\n' % (num_core, sfile, sfile))


if __name__ == '__main__':
    main(int(argv[1]))
