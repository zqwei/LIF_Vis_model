import numpy as np
# import pandas as pd
from sys import argv

num_cell_type = [0, 3700, 7000, 8500, 9300, 10000, 39750, 45000]
cell_type = ['Scnn1a', 'Rorb', 'Nr5a1', 'PV1', 'PV2']
tstop = 500
f_name = 'lr2_g8_8_test%dms_inh_lif_syn_z' % (tstop)


def compute_estimation_error(n_file_dir, ref_file, ncol=2):
    series = np.genfromtxt(ref_file + '/tot_f_rate.dat', delimiter=' ')
    ref_firing_rate = series[0:10000, ncol]
    cell_id = series[0:10000, 0]
    series = np.genfromtxt(n_file_dir + '/tot_f_rate.dat', delimiter=' ')
    firing_rate = series[0:10000, ncol]
    diff_rate = ref_firing_rate - firing_rate
    print "ref rate, rate, difference\n"
    for cell_group in xrange(len(cell_type)):
        group = np.logical_and(cell_id >= num_cell_type[cell_group], cell_id < num_cell_type[cell_group + 1])
        print "%f, %f, %f\n" % (np.mean(ref_firing_rate[group]), np.mean(firing_rate[group]), np.mean(diff_rate[group]))


if __name__ == '__main__':
    compute_estimation_error('output_' + f_name + argv[1], 'output_ll2_g8_8_test500ms_inh_lif_syn_z104')
