import numpy as np
import pandas as pd
import h5py

import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams.update({'font.size': 20})

sys_dict = {}

sys_dict['ll1'] = { 'cells_file': '../build/ll1.csv', 'f_1': '../simulations_ll1/gratings/output_ll1_g', 'f_2': '_sd278/', 'f_3': '_sd278_LGN_only_no_con', 'grating_id': range(7, 240, 30)+range(8, 240, 30) }
sys_dict['ll2'] = { 'cells_file': '../build/ll2.csv', 'f_1': '../simulations_ll2/gratings/output_ll2_g', 'f_2': '_sd278/', 'f_3': '_sd278_LGN_only_no_con', 'grating_id': range(7, 240, 30)+range(8, 240, 30) }
sys_dict['ll3'] = { 'cells_file': '../build/ll3.csv', 'f_1': '../simulations_ll3/gratings/output_ll3_g', 'f_2': '_sd278/', 'f_3': '_sd278_LGN_only_no_con', 'grating_id': range(7, 240, 30)+range(8, 240, 30) }
f_out = 'i_amplification/i_amplification_process_LGN_crt_LL_bln_from_5_cell_test.csv'

#sys_dict['ll2_ctr30'] = { 'cells_file': '../build/ll2.csv', 'f_1': '../simulations_ll2/gratings/output_ll2_g', 'f_2': '_ctr30_sd278/', 'f_3': '_ctr30_sd278_LGN_only_no_con', 'grating_id': range(8, 240, 30) }
#f_out = 'i_amplification/i_amplification_process_LGN_crt_LL_ctr30_bln_from_5_cell_test.csv'

#sys_dict['ll2_ctr10'] = { 'cells_file': '../build/ll2.csv', 'f_1': '../simulations_ll2/gratings/output_ll2_g', 'f_2': '_ctr10_sd278/', 'f_3': '_ctr10_sd278_LGN_only_no_con', 'grating_id': range(8, 240, 30) }
#f_out = 'i_amplification/i_amplification_process_LGN_crt_LL_ctr10_bln_from_5_cell_test.csv'

g_metadata = pd.read_csv('/allen/aibs/mat/antona/network/14-simulations/6-LGN_firing_rates_and_spikes/LGN_spike_trains/movies_gratings/res_192_metadata.txt', sep=' ', header=None)
g_metadata.columns = ['path', 'ori', 'SF', 'TF', 'do_not_use']
g_metadata['grating_id'] = g_metadata['path'].str.split('.').str[-2].str.split('_').str[-1].astype(int64)
print g_metadata
quit()


N_trials = 10

t_av = [500.0, 3000.0]
'''
t_bln = [100.0, 500.0]
data_range_bln = 0.05 # Range of the data to be included in the baseline calculation.
'''

gids = range(2, 10000, 200)

crt_av_list = []
LGN_av_list = []
gid_list = []
model_list = []
type_list = []
Ntrial_list = []
grating_id_list = []

bln_df = pd.read_csv('../5_cells_measure_i_baseline/i_SEClamp_baseline.csv', sep=' ')

for sys_name in sys_dict:
    cells = pd.read_csv(sys_dict[sys_name]['cells_file'], sep=' ')
    for gid in gids:
        i_combined = {}
        i_combined['f_2'] = np.array([])
        i_combined['f_3'] = np.array([])
        gid_type = cells[cells['index']==gid]['type'].values[0]

        i_bln = bln_df[bln_df['type']==gid_type]['i_bln'].values[0]

        for grating_id in sys_dict[sys_name]['grating_id']:
            for trial in xrange(0, N_trials):
                for f_label in ['f_2', 'f_3']:
                    f_name = '%s%d_%d%s/i_SEClamp-cell-%d.h5' % (sys_dict[sys_name]['f_1'], grating_id, trial, sys_dict[sys_name][f_label], gid)
                    print 'Processing file %s.' % (f_name)
                    h5 = h5py.File(f_name, 'r')
                    values = h5['values'][...]
                    tarray = np.arange(0, values.size) * 1.0 * h5.attrs['dt']
                    h5.close()
                    if (i_combined[f_label].size == 0):
                        i_combined[f_label] = values
                    else:
                        i_combined[f_label] = i_combined[f_label] + values
            i_combined['f_2'] = i_combined['f_2'] / N_trials
            i_combined['f_3'] = i_combined['f_3'] / N_trials

            # Assume that tarray is the same between 'f_2' and 'f_3' and for all trials.
            ind_av = np.intersect1d( np.where(tarray > t_av[0]), np.where(tarray < t_av[1]) )

            av = {}
            for f_label in ['f_2', 'f_3']:
                '''
                # Compute baseline. For that, use the current values within certain window.  Then, choose the bottom 5 percentile
                # of values, as those represent the values closest to true baseline, unaffected by spontaneous activity (this follows
                # Lien and Scanziani, Nat. Neurosci., 2013).  Since stronger current is more negative, we actually choose the top 5 percentile.
                ind_bln = np.intersect1d( np.where(tarray > t_bln[0]), np.where(tarray < t_bln[1]) )
                i_tmp = i_combined[f_label][ind_bln]
                i_tmp_cutoff = i_tmp.max() - (i_tmp.max() - i_tmp.min()) * data_range_bln
                ind_bln_1 = np.where(i_tmp > i_tmp_cutoff)[0]

                av[f_label] = i_combined[f_label][ind_av].mean() - i_tmp[ind_bln_1].mean()
                '''
                # Use true baseline.
                av[f_label] = i_combined[f_label][ind_av].mean() - i_bln

                #plt.plot(tarray, i_combined[f_label])
                plt.plot(tarray[ind_av], i_combined[f_label][ind_av] - i_bln)
                #plt.plot(tarray[ind_bln], i_combined[f_label][ind_bln])
                #plt.plot(tarray[ind_bln][ind_bln_1], i_tmp[ind_bln_1])
                #plt.ylim((0.0, 0.1))
                plt.title('System %s, gid %d' % (sys_name, gid))
                plt.show()

            crt_av_list.append(av['f_2'])
            LGN_av_list.append(av['f_3'])
            gid_list.append(gid)
            model_list.append(sys_name)
            type_list.append(gid_type)
            Ntrial_list.append(N_trials)
            grating_id_list.append(grating_id)

df = pd.DataFrame()
df['model'] = model_list
df['gid'] = gid_list
df['type'] = type_list
df['N_trials'] = Ntrial_list
df['grating_id'] = grating_id_list
df['LGN'] = LGN_av_list
df['crt'] = crt_av_list

df.to_csv(f_out, sep=' ', index=False)

