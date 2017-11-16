import pickle
import numpy as np
import matplotlib.pyplot as plt

import scipy.stats as scp_stats

import pandas as pd

import f_rate_t_by_type_functions as frtbt


N_trials = 15

# Decide which systems we are doing analysis for.
sys_dict = {}
sys_dict['all_mice'] = { 'cells_file': '../build/ll1.csv', 'f_1': '/data/mat/yazan/analyzeDanielSevData/SeV_NWBfiles_FOR_YAZAN/gratings/', 'f_2': '/spks.dat', 'f_3': '/tot_f_rate.dat', 'f_out': 'Rmax_exp_DanSeVData/all_mice_Rmax.csv', 'grating_ids': range(7, 240, 30)+range(8, 240, 30)+range(9, 240, 30) }


result_fname_prefix = 'Rmax_exp_DanSeVData/Rmax_by_type'
result_fname = result_fname_prefix + '.csv'
result_fig_fname = result_fname_prefix + '.eps'


g_spec = pd.read_csv('/data/mat/antona/network/14-simulations/6-LGN_firing_rates_and_spikes/LGN_spike_trains/movies_gratings/res_192_metadata.txt', sep = ' ')
g_spec.columns = ['fname', 'ori', 'SF', 'TF', 'IGNORE']
g_spec['g_id'] = g_spec['fname'].str.split('.').str[-2].str.split('_').str[-1]


# Load simulation data and obtain Rmax for each cell.
for sys_name in sys_dict.keys():
    gratings_rates = np.array([])
    print gratings_rates.shape
    for grating_id in sys_dict[sys_name]['grating_ids']:

        g_spec_tmp = g_spec[g_spec['g_id'] == str(grating_id)]
        ori = g_spec_tmp['ori'].values[0]
        SF = g_spec_tmp['SF'].values[0]
        TF = g_spec_tmp['TF'].values[0]

        rates_tmp = np.array([])
        for i_trial in xrange(0, N_trials):
            f_name = '%s/%ddegs_%dHz/trial_%d/%s' % (sys_dict[sys_name]['f_1'], int(ori), int(TF), i_trial, sys_dict[sys_name]['f_3'])
            print 'Processing file %s.' % (f_name)

            tmp = np.genfromtxt(f_name, delimiter=' ')[:, 1] # Assume all files have the same columns of gids; use the 2nd column for rates.
            if (rates_tmp.size == 0):
                rates_tmp = tmp
            else:
                rates_tmp = rates_tmp + tmp
        rates_tmp = rates_tmp / (1.0 * N_trials)
        if (gratings_rates.size == 0):
            gratings_rates = rates_tmp
        else:
            gratings_rates = np.vstack((gratings_rates, rates_tmp))
    Rmax = np.amax(gratings_rates, axis = 0)

    sys_df = pd.read_csv(sys_dict[sys_name]['cells_file'], sep=' ')
    # Write Rmax to a file, with gid and cell type information.
    f_out = open(sys_dict[sys_name]['f_out'], 'w')
    f_out.write('gid type Rmax\n')
    for gid in xrange(0, Rmax.size):
        f_out.write('%d %s %f\n' % (gid, sys_df['type'].ix[gid], Rmax[gid]))
    f_out.close()





# Compute averages and standard deviations of Rmax by type and save to file.
result_f = open(result_fname, 'w')
result_f.write('system cell_type av_Rmax std_Rmax sem_Rmax\n')
for sys_name in sys_dict.keys():
    df_tmp = pd.read_csv(sys_dict[sys_name]['f_out'], sep=' ')
    for type in list(set(list(df_tmp['type']))):
        tmp = df_tmp[df_tmp['type'] == type]['Rmax']
        result_string = '%s %s %f %f %f' % (sys_name, type, tmp.mean(), tmp.std(), scp_stats.sem(tmp))
        result_f.write(result_string + '\n')
        print result_string
result_f.close()





# Load Rmax from simulations and narrow down to only the biophysical cells.
df_rates = pd.read_csv(result_fname, sep=' ')
df_rates_tmp = df_rates[df_rates['cell_type'].isin(['Scnn1a', 'Rorb', 'Nr5a1', 'PV1', 'PV2'])]

# Plot the average Rmax (over cells).
# With std as error bars.
#ax = df_rates_tmp.pivot(index='cell_type', columns='system', values='av_Rmax').reindex(['Scnn1a', 'Rorb', 'Nr5a1', 'PV1', 'PV2']).plot(kind='bar', yerr=df_rates_tmp.pivot(index='cell_type', columns='system', values='std_Rmax').reindex(['Scnn1a', 'Rorb', 'Nr5a1', 'PV1', 'PV2']), fontsize=20)
# With sem as error bars.
ax = df_rates_tmp.pivot(index='cell_type', columns='system', values='av_Rmax').reindex(['Scnn1a', 'Rorb', 'Nr5a1', 'PV1', 'PV2']).plot(kind='bar', yerr=df_rates_tmp.pivot(index='cell_type', columns='system', values='sem_Rmax').reindex(['Scnn1a', 'Rorb', 'Nr5a1', 'PV1', 'PV2']), fontsize=20)

# Add bars for the experimental data.
x_exp = np.array([5, 5.3, 6, 6.3])
# With std as error bars.
#ax.bar(x_exp, exp_data_mean, yerr=exp_data_std, width=0.2, color='gray', error_kw=dict(ecolor='k'))
# With sem as error bars.
ax.bar(x_exp, exp_data_mean, yerr=exp_data_sem, width=0.2, color='gray', error_kw=dict(ecolor='k'))
labels = [item.get_text() for item in ax.get_xticklabels()] # Make sure this is done before xticks are extended; otherwise, the labels list will contain more empty entries.
labels = labels + exp_labels
ax.set_xticks(list(ax.get_xticks()) + list(x_exp))
ax.set_xticklabels(labels)

ax.set_ylabel('Rmax (Hz)', fontsize=20)
ax.set_xlim([-0.5, 7.0])
ax.set_ylim(bottom=0.0)
plt.gcf().subplots_adjust(bottom=0.3)

ax.annotate('Niell and Stryker, (2008).\n L4 Exc.: ~6 Hz\n Inh.: ~13 Hz', xy=(0.5, 0.7), xycoords='axes fraction', fontsize=16,
                xytext=(-5, 5), textcoords='offset points',
                ha='right', va='bottom')

plt.savefig(result_fig_fname, format='eps')

plt.show()


