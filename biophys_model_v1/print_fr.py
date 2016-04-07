from glob import glob
from os.path import exists
import numpy as np
num_cell_type = [0, 3700, 7000, 8500, 9300, 10000, 39750, 45000]
cell_type = ['Scnn1a', 'Rorb', 'Nr5a1', 'PV1', 'PV2']


def print_percent_active_cell(tot_f_rate_dir_name, ncol=2):
    file_name = tot_f_rate_dir_name+'/tot_f_rate.dat'
    stats_file_name = tot_f_rate_dir_name+'/stats_tot_f_rate.dat'
    series = np.genfromtxt(file_name, delimiter=' ')
    tot_fr = series[:, ncol]
    cell_id = series[:, 0]
    percent_active_cell = np.zeros(5)
    with open(stats_file_name, 'w') as output:
        for cell_group in xrange(len(percent_active_cell)):
            group = np.logical_and(cell_id >= num_cell_type[cell_group], cell_id < num_cell_type[cell_group+1])
            percent_active_cell[cell_group] = np.mean(tot_fr[group] > 2)
            output.write('%s: %f\n' % (cell_type[cell_group], percent_active_cell[cell_group]))


def main():
    ncol = 2
    for n_run in glob("output_*"):
        if not exists(n_run+'/stats_tot_f_rate.dat'):
            print_percent_active_cell(n_run, ncol)


if __name__ == '__main__':
    main()
