import numpy as np
import matplotlib.pyplot as plt

import matplotlib
matplotlib.rcParams.update({'font.size': 15})

import pandas as pd

from scipy.stats.stats import pearsonr


gids = [6000]

# Decide which systems we are doing analysis for.
sys_dict = {}
#sys_dict['ll1'] = { 'cells_file': '../build/ll1.csv', 'f_1': '../simulations_ll1/output_ll1_', 'f_2': '_sd278/spk.dat', 'f_3': '_sd278/tot_f_rate.dat', 'f_out': 'Ori/ll1_rates.npy', 'f_out_pref': 'Ori/ll1_pref_stat.csv', 'grating_id': range(6, 240, 30)+range(7, 240, 30)+range(8, 240, 30)+range(9, 240, 30) }
sys_dict['ll2'] = { 'cells_file': '../build/ll2.csv', 'f_1': '../output_ll2_', 'f_2': '_sd278/spk.dat', 'f_3': '_sd278/tot_f_rate.dat', 'f_out': 'Ori/ll2_rates.npy', 'f_out_pref': 'Ori/ll2_pref_stat.csv', 'grating_id': range(7, 240, 30)+range(8, 240, 30)+range(9, 240, 30) }
#sys_dict['ll3'] = { 'cells_file': '../build/ll3.csv', 'f_1': '../output_ll3_', 'f_2': '_sd278/spk.dat', 'f_3': '_sd278/tot_f_rate.dat', 'f_out': 'Ori/ll3_rates.npy', 'f_out_pref': 'Ori/ll3_pref_stat.csv', 'grating_id': range(8, 240, 30) }
#sys_dict['rr2'] = { 'cells_file': '../build/rr2.csv', 'f_1': '../output_rr2_', 'f_2': '_sd282_cn0/spk.dat', 'f_3': '_sd282_cn0/tot_f_rate.dat', 'f_out': 'Ori/rr2_rates.npy', 'f_out_std': 'Ori/rr2_rates_std.npy', 'f_out_pref': 'Ori/rr2_pref_stat.csv', 'grating_id': range(8, 240, 30)}
sys_dict['ll2_LGN_only_no_con'] = { 'cells_file': '../build/ll2.csv', 'f_1': '../output_ll2_', 'f_2': '_sd278_LGN_only_no_con/spk.dat', 'f_3': '_sd278/tot_f_rate.dat', 'f_out': 'Ori/ll2_LGN_only_no_con_rates.npy', 'f_out_std': 'Ori/ll2_LGN_only_no_con_rates_std.npy', 'f_out_pref': 'Ori/ll2_LGN_only_no_con_pref_stat.csv', 'grating_id': range(8, 240, 30) }

# Load simulation data.
sim_data = {}
av_list = []
for i_sys, sys_name in enumerate(sys_dict.keys()):
    sim_data[sys_name] = pd.read_csv(sys_dict[sys_name]['f_out_pref'], sep=' ')
    ori_list = [x for x in sim_data[sys_name].columns.values if x not in ['id', 'ori', 'SF', 'TF', 'CV_ori', 'OSI_modulation', 'DSI']]
    ori_float = np.array([float(x) for x in ori_list])
    ind_ori_sort = ori_float.argsort()
    ori_float_sorted = ori_float[ind_ori_sort]

    for k_cell, gid in enumerate(gids):
        av = []
        std = []
        for ori in ori_list:
            tmp = sim_data[sys_name][sim_data[sys_name]['id'] == gid][ori].values[0]
            tmp = tmp[1:][:-1].split(',') # This is a string with the form '(av,std)', and so we can remove the brackets and comma to get strings 'av' and 'std', where av and std are numbers.
            av.append(float(tmp[0]))
            std.append(float(tmp[1]))
        # Convert av and std to numpy array and change the sequence of elements according to the sorted ori.
        av = np.array(av)[ind_ori_sort]
        std = np.array(std)[ind_ori_sort]

        av_list.append(av)
        plt.plot(ori_float_sorted, av, '-o', label=sys_name)

print 'Pearson R: %f' % (pearsonr(av_list[0], av_list[1])[0])

plt.legend()
plt.show()


