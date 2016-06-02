import numpy as np
import pandas as pd
from os.path import exists

SRN_cell_range = [0, 8500]
PV2_cell_range = [9300, 10000]
cell_range = [SRN_cell_range, PV2_cell_range]


def compute_estimation_error(n_file_dir, m_file_dir, ref_file, ncol=2):
    series = np.genfromtxt(ref_file, delimiter=' ')
    ref_firing_rate = series[0:10000, ncol]
    cell_id = series[0:10000, 0]
    series = np.genfromtxt(n_file_dir + '/tot_f_rate.dat', delimiter=' ')
    firing_rate = series[0:10000, ncol]
    diff_rate = firing_rate - ref_firing_rate
    pdUpdateOld = pd.read_csv(n_file_dir + "/cell_update_stats_old.dat")
    pdUpdateNew = pdUpdateOld
    for cell_group in xrange(len(cell_range)):
        group = np.logical_and(cell_id < cell_range[cell_group][1], cell_id >= cell_range[cell_group][0])
        # compute estimation error
        E_curr = np.mean(diff_rate[group]**2)  # mean
        E_old = pdUpdateOld['E_old'][cell_group]
        if E_curr < E_old:
            pdUpdateNew['E_old'][cell_group] = E_curr
        pdUpdateNew['E_curr'][cell_group] = E_curr
        if exists(m_file_dir + '/tot_f_rate.dat'):
            series = np.genfromtxt(m_file_dir + '/tot_f_rate.dat', delimiter=' ')
            pre_firing_rate = series[0:10000, ncol]
            f_grad = pre_firing_rate - firing_rate
            pdNthUpdate = pd.read_csv(n_file_dir + "/cell_update_stats_old.dat")
            pdMthUpdate = pd.read_csv(m_file_dir + "/cell_update_stats_old.dat")
            w_n = pdNthUpdate['w_curr'][cell_group]
            w_m = pdMthUpdate['w_curr'][cell_group]
            w_grad = w_m - w_n
            if w_grad != 0. and pdUpdateNew['dw'][cell_group] != 0.:
                f_w_grad = f_grad[group] / w_grad
                grad_curr = np.mean(diff_rate[group] * f_w_grad)  # mean
            else:
                grad_curr = 0.
            # update grad_curr, w_curr, dw
            pdUpdateNew['grad_curr'][cell_group] = grad_curr
            dw = pdUpdateNew['dw'][cell_group] * 0.5  # / 2.
            pdUpdateNew['dw'][cell_group] = dw
            w_curr = pdUpdateNew['w_curr'][cell_group]
            w_curr = w_curr - grad_curr * dw
            pdUpdateNew['w_curr'][cell_group] = w_curr
    pdUpdateNew.to_csv(n_file_dir + "/cell_update_stats_new.dat", mode='w', index=False)
