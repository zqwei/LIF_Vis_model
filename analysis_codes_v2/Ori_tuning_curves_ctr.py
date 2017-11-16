import numpy as np
import matplotlib.pyplot as plt

import matplotlib
matplotlib.rcParams.update({'font.size': 15})

import pandas as pd


# Cells for which tuning curves should be plotted.
#gids = [50, 5000, 8200, 8900, 9600]
gids = [50] #[8900]

# Number of trials to use for calculation of spont rate.
N_trials_spont = 20

# Decide which systems we are doing analysis for.
sys_dict = {}
sys_dict['ll2_TF4Hz'] = { 'cells_file': '../build/ll2.csv', 'f_1': '../output_ll2_', 'f_2': '_sd278/spk.dat', 'f_3': '_sd278/tot_f_rate.dat', 'f_out': 'Ori/ll2_rates_4Hz.npy', 'f_out_std': 'Ori/ll2_rates_std_4Hz.npy', 'f_out_pref': 'Ori/ll2_pref_stat_4Hz.csv', 'grating_id': range(8, 240, 30), 'plt_color': 'darkorange' } #'blue' }
sys_dict['ll2_ctr30_TF4Hz'] = { 'cells_file': '../build/ll2.csv', 'f_1': '../output_ll2_', 'f_2': '_ctr30_sd278/spk.dat', 'f_3': '_ctr30_sd278/tot_f_rate.dat', 'f_out': 'Ori/ll2_ctr30_rates_4Hz.npy', 'f_out_std': 'Ori/ll2_ctr30_rates_std_4Hz.npy', 'f_out_pref': 'Ori/ll2_ctr30_pref_stat_4Hz.csv', 'grating_id': range(8, 240, 30), 'plt_color': 'orange' } #'skyblue' }
sys_dict['ll2_ctr10_TF4Hz'] = { 'cells_file': '../build/ll2.csv', 'f_1': '../output_ll2_', 'f_2': '_ctr10_sd278/spk.dat', 'f_3': '_ctr10_sd278/tot_f_rate.dat', 'f_out': 'Ori/ll2_ctr10_rates_4Hz.npy', 'f_out_std': 'Ori/ll2_ctr10_rates_std_4Hz.npy', 'f_out_pref': 'Ori/ll2_ctr10_pref_stat_4Hz.csv', 'grating_id': range(8, 240, 30), 'plt_color': 'navajowhite' } #'lightskyblue' }

# Get the average spont firing rate for cells.
spont_rates = np.array([])
for i_trial in xrange(0, N_trials_spont):
    f_name = '../simulations_ll2/spont/output_ll2_spont_%d_sd278/tot_f_rate.dat' % (i_trial) #(sys_dict[sys_name]['f_1'], i_trial, sys_dict[sys_name]['f_3'])
    print 'Processing file %s.' % (f_name)
    tmp = np.genfromtxt(f_name, delimiter=' ')[:, 1] # Assume all files have the same columns of gids; use the 2nd column for rates.
    if (spont_rates.size == 0):
        spont_rates = tmp
    else:
        spont_rates = spont_rates + tmp
spont_rates = spont_rates / (1.0 * N_trials_spont)


#fig, axes = plt.subplots(nrows=2, ncols=3)
#axes = axes.reshape(-1)

# Load simulation data.
sim_data = {}
for sys_name in sys_dict.keys():
    sim_data[sys_name] = pd.read_csv(sys_dict[sys_name]['f_out_pref'], sep=' ')

for k_cell, gid in enumerate(gids):
    for sys_name in sys_dict.keys():
        ori_list = [x for x in sim_data[sys_name].columns.values if x not in ['id', 'ori', 'SF', 'TF', 'CV_ori', 'OSI_modulation', 'DSI']]
        ori_float = np.array([float(x) for x in ori_list])
        ind_ori_sort = ori_float.argsort()
        ori_float_sorted = ori_float[ind_ori_sort]

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

        #axes[k_cell].errorbar(ori_float_sorted, av, yerr=std, marker='o', ecolor='k')
        #axes[k_cell].set_xticks(ori_float_sorted)
        #axes[k_cell].set_ylim(bottom=0.0)
        #axes[k_cell].set_xlim((0.0, 360.0))

        #plt.errorbar(ori_float_sorted, av, yerr=std, color=sys_dict[sys_name]['plt_color'], marker='o', mfc='white', mec=sys_dict[sys_name]['plt_color'], markersize=10) 
        plt.errorbar(ori_float_sorted, av/av.max(), yerr=std/av.max(), color=sys_dict[sys_name]['plt_color'], marker='o', mfc='white', mec=sys_dict[sys_name]['plt_color'], markersize=10)
        plt.xticks(ori_float_sorted)

    spont = spont_rates[gid]
    #plt.plot([0.0, 360.0], [spont, spont], c='k', ls='--')

    plt.ylim(bottom=0.0)
    plt.xlim((-10.0, 360.0))

    plt.title('Cell ID %d' % (gid))

    #plt.savefig('Ori/ll2_and_ll2_ctr10_4Hz_tuning_curves_gid_%d.eps' % (gid), format='eps')
    #plt.savefig('Ori/ll2_and_ll2_ctr10_4Hz_tuning_curves_gid_%d_norm.eps' % (gid), format='eps')

    #plt.savefig('Ori/ll2_ctr10_30_80_4Hz_tuning_curves_gid_%d.eps' % (gid), format='eps')
    plt.savefig('Ori/ll2_ctr10_30_80_4Hz_tuning_curves_gid_%d_norm.eps' % (gid), format='eps')

    plt.show()


