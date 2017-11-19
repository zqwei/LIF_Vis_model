import numpy as np
import matplotlib
matplotlib.rcParams.update({'font.size': 15})
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

result_fig = 'Ori/tuning_curves.eps'

# Cells for which tuning curves should be plotted.
#gids = [1000, 5000, 8000, 8900, 9600]
gids = [50, 5000, 8200, 8900, 9600]

# Number of trials to use for calculation of spont rate.
N_trials_spont = 20

cell_db_path = '/allen/aibs/mat/antona/network/14-simulations/9-network/build/'
# Decide which systems we are doing analysis for.
sys_dict = {}
sys_dict['ll1'] = {'cells_file': cell_db_path+'ll1.csv',
                    'f_1': '../simulation_ll1/output_ll1_',
                    'f_2': '_sdlif_z101/spk.dat',
                    'f_3': '_sdlif_z101/tot_f_rate.dat',
                    'f_out': 'Ori/ll1_rates.npy',
                    'f_out_std': 'Ori/ll1_rates_std.npy',
                    'f_out_pref': 'Ori/ll1_pref_stat.csv',
                    'grating_id': range(7, 240, 30)+range(8, 240, 30)+range(9, 240, 30)}
sys_dict['ll2'] = {'cells_file': cell_db_path+'ll2.csv',
                    'f_1': '../simulation_ll2/output_ll2_',
                    'f_2': '_sdlif_z101/spk.dat',
                    'f_3': '_sdlif_z101/tot_f_rate.dat',
                    'f_out': 'Ori/ll2_rates.npy',
                    'f_out_std': 'Ori/ll2_rates_std.npy',
                    'f_out_pref': 'Ori/ll2_pref_stat.csv',
                    'grating_id': range(7, 240, 30)+range(8, 240, 30)+range(9, 240, 30)}
sys_dict['ll3'] = {'cells_file': cell_db_path+'ll3.csv',
                    'f_1': '../simulation_ll3/output_ll3_',
                    'f_2': '_sdlif_z101/spk.dat',
                    'f_3': '_sdlif_z101/tot_f_rate.dat',
                    'f_out': 'Ori/ll3_rates.npy',
                    'f_out_std': 'Ori/ll3_rates_std.npy',
                    'f_out_pref': 'Ori/ll3_pref_stat.csv',
                    'grating_id': range(7, 240, 30)+range(8, 240, 30)+range(9, 240, 30)}


# Load simulation data.
sim_data = {}
for sys_name in sys_dict.keys():
    sim_data[sys_name] = pd.read_csv(sys_dict[sys_name]['f_out_pref'], sep=' ')
    ori_list = [x for x in sim_data[sys_name].columns.values if x not in ['id', 'ori', 'SF', 'TF', 'CV_ori', 'OSI_modulation', 'DSI']]
    ori_float = np.array([float(x) for x in ori_list])
    ind_ori_sort = ori_float.argsort()
    ori_float_sorted = ori_float[ind_ori_sort]

    fig, axes = plt.subplots(nrows=2, ncols=3)
    axes = axes.reshape(-1)

    # Get the average spont firing rate for cells.
    spont_rates = np.array([])
    for i_trial in xrange(0, N_trials_spont):
        f_name = '%sspont_%d%s' % (sys_dict[sys_name]['f_1'], i_trial, sys_dict[sys_name]['f_3'])
        # print 'Processing file %s.' % (f_name)
        tmp = np.genfromtxt(f_name, delimiter=' ')[:, 1] # Assume all files have the same columns of gids; use the 2nd column for rates.
        if (spont_rates.size == 0):
            spont_rates = tmp
        else:
            spont_rates = spont_rates + tmp
    spont_rates = spont_rates / (1.0 * N_trials_spont)

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

        axes[k_cell].errorbar(ori_float_sorted, av, yerr=std, marker='o', ecolor='k')
        axes[k_cell].set_xticks(ori_float_sorted)
        #axes[k_cell].set_title('Gid %d' % (gid))
        axes[k_cell].set_ylim(bottom=0.0)
        axes[k_cell].set_xlim((0.0, 360.0))

        spont = spont_rates[gid]
        axes[k_cell].plot([0.0, 360.0], [spont, spont], c='k', ls='--')

    plt.show()
    plt.savefig('Ori/tuning_curves_%s.eps'%sys_name)
