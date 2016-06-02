import numpy as np
import pandas as pd
# from scipy.stats import ttest_1samp

num_cell_type = [0, 3700, 7000, 8500, 9300, 10000, 39750, 45000]
cell_type = ['Scnn1a', 'Rorb', 'Nr5a1', 'PV1', 'PV2']


def compute_estimation_error(n_file_dir, m_file_dir, ref_file, ncol=2):
    series = np.genfromtxt(ref_file, delimiter=' ')
    ref_firing_rate = series[0:10000, ncol]
    cell_id = series[0:10000, 0]
    series = np.genfromtxt(n_file_dir + '/tot_f_rate.dat', delimiter=' ')
    firing_rate = series[0:10000, ncol]
    series = np.genfromtxt(m_file_dir + '/tot_f_rate.dat', delimiter=' ')
    pre_firing_rate = series[0:10000, ncol]
    f_grad = pre_firing_rate - firing_rate
    pdNthUpdate = pd.read_csv(n_file_dir + "/cell_update_stats_old.dat")
    pdMthUpdate = pd.read_csv(m_file_dir + "/cell_update_stats_old.dat")
    diff_rate = firing_rate - ref_firing_rate
    pdUpdateOld = pd.read_csv(n_file_dir + "/cell_update_stats_old.dat")
    pdUpdateNew = pdUpdateOld
    w_n = pdNthUpdate['w_curr']
    w_m = pdMthUpdate['w_curr']
    w_grad = w_m - w_n
    for cell_group in xrange(len(cell_type)):
        group = np.logical_and(cell_id >= num_cell_type[cell_group], cell_id < num_cell_type[cell_group + 1])
        # compute estimation error
        E_curr = np.mean(diff_rate[group]**2)  # mean
        # print(w_grad[cell_group] != 0. and pdUpdateNew['dw']!=0.)
        if w_grad[cell_group] != 0. and pdUpdateNew['dw'][cell_group] != 0.:
            f_w_grad = f_grad[group] / w_grad[cell_group]
            grad_curr = np.mean(diff_rate[group] * f_w_grad)  # mean
        else:
            grad_curr = 0.
        E_old = pdUpdateOld['E_old'][cell_group]
        #
        # decide if E_curr < E_old
        # update E_old, grad_old, w_old
        if E_curr < E_old:
            pdUpdateNew['E_old'][cell_group] = E_curr
            pdUpdateNew['grad_old'][cell_group] = grad_curr
            w_old = pdUpdateNew['w_curr'][cell_group]
            pdUpdateNew['w_old'][cell_group] = w_old
        # update Ecurr, grad_curr, w_curr, dw
        pdUpdateNew['E_curr'][cell_group] = E_curr
        pdUpdateNew['grad_curr'][cell_group] = grad_curr
        dw = pdUpdateNew['dw'][cell_group] * 0.5  # / 2.
        pdUpdateNew['dw'][cell_group] = dw
        w_old = pdUpdateNew['w_old'][cell_group]
        # grad_old = pdUpdateNew['grad_old'][cell_group]
        # w_curr = w_old - grad_old * dw
        print(dw)
        w_curr = pdUpdateNew['w_curr'][cell_group]
        w_curr = w_curr - grad_curr * dw
        pdUpdateNew['w_curr'][cell_group] = w_curr
    pdUpdateNew.to_csv(n_file_dir + "/cell_update_stats_new.dat", mode='w', index=False)
