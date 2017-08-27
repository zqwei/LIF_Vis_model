import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import pandas as pd

import f_rate_t_by_type_functions as frtbt


N_trials = 10

result_fname_prefix = 'Rmax/Rmax_by_type'
result_fname = result_fname_prefix + '.csv'
result_fig_fname = result_fname_prefix + '.eps'

cell_db_path = '/data/mat/antona/network/14-simulations/9-network/build/'
# Decide which systems we are doing analysis for.
sys_dict = {}
sys_dict['ll1'] = {'cells_file': cell_db_path+'ll1.csv',
                    'f_1': '../simulation_ll1/output_ll1_',
                    'f_2': '_sdlif_z101/spk.dat',
                    'f_3': '_sdlif_z101/tot_f_rate.dat',
                    'f_out': 'Rmax/ll1_Rmax.csv',
                    'grating_ids': range(7, 240, 30)+range(8, 240, 30)+range(9, 240, 30)}
sys_dict['ll2'] = {'cells_file': cell_db_path+'ll2.csv',
                    'f_1': '../simulation_ll2/output_ll2_',
                    'f_2': '_sdlif_z101/spk.dat',
                    'f_3': '_sdlif_z101/tot_f_rate.dat',
                    'f_out': 'Rmax/ll2_Rmax.csv',
                    'grating_ids': range(7, 240, 30)+range(8, 240, 30)+range(9, 240, 30)}
sys_dict['ll3'] = {'cells_file': cell_db_path+'ll3.csv',
                    'f_1': '../simulation_ll3/output_ll3_',
                    'f_2': '_sdlif_z101/spk.dat',
                    'f_3': '_sdlif_z101/tot_f_rate.dat',
                    'f_out': 'Rmax/ll3_Rmax.csv',
                    'grating_ids': range(7, 240, 30)+range(8, 240, 30)+range(9, 240, 30)}


# Load simulation data and obtain Rmax for each cell.
for sys_name in sys_dict.keys():
    gratings_rates = np.array([])
    # print gratings_rates.shape
    for grating_id in sys_dict[sys_name]['grating_ids']:
        rates_tmp = np.array([])
        for i_trial in xrange(0, N_trials):
            f_name = '%sg%d_%d%s' % (sys_dict[sys_name]['f_1'], grating_id, i_trial, sys_dict[sys_name]['f_3'])
            # print 'Processing file %s.' % (f_name)
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
result_f.write('system cell_type av_Rmax std_Rmax\n')
for sys_name in sys_dict.keys():
    df_tmp = pd.read_csv(sys_dict[sys_name]['f_out'], sep=' ')
    for type in list(set(list(df_tmp['type']))):
        tmp = df_tmp[df_tmp['type'] == type]['Rmax']
        result_string = '%s %s %f %f' % (sys_name, type, tmp.mean(), tmp.std())
        result_f.write(result_string + '\n')
        # print result_string
result_f.close()


# Load the firing rates from the experiment.
exp_f_list = ['ANL4Exc.csv', 'AWL4Exc.csv', 'ANInh.csv', 'AWInh.csv'] #'ANL4Inh.csv', 'AWL4Inh.csv']
exp_labels = ['An. L4 Exc.', 'Aw. L4 Exc.', 'An. Inh.', 'Aw. Inh.'] #'An. L4 Inh.', 'Aw. L4 Inh.']

exp_rates_Exc = np.array([])
exp_rates_Inh = np.array([])

exp_data_mean = []
exp_data_std = []
for i_exp, exp_f in enumerate(exp_f_list):
    tmp_df = pd.read_csv('/data/mat/antona/experimental_data/ephys_Sev/2016_paper_data/gratings/' + exp_f, sep=',')
    exp_data_mean.append(tmp_df['Rmax'].mean())
    exp_data_std.append(tmp_df['Rmax'].std())

    if ('Exc' in exp_labels[i_exp]):
        exp_rates_Exc = np.append(exp_rates_Exc, tmp_df['Rmax'].values)
    else:
        exp_rates_Inh = np.append(exp_rates_Inh, tmp_df['Rmax'].values)


# Plot Rmax distributions.
hist_bins = np.linspace(0.0, 100.0, 101)


# Load simulation data and create histograms.
sim_hist_Exc = {}
sim_hist_Inh = {}
for sys_name in sys_dict.keys():
    df_Rmax = pd.read_csv(sys_dict[sys_name]['f_out'], sep=' ')
    tmp_Exc = df_Rmax[df_Rmax['type'].isin(['Scnn1a', 'Rorb', 'Nr5a1'])]['Rmax'].values
    tmp_Inh = df_Rmax[df_Rmax['type'].isin(['PV1', 'PV2'])]['Rmax'].values
    sim_hist_Exc[sys_name] = np.histogram(tmp_Exc, bins=hist_bins)[0] / (1.0 * tmp_Exc.size)
    sim_hist_Inh[sys_name] = np.histogram(tmp_Inh, bins=hist_bins)[0] / (1.0 * tmp_Inh.size)

#Create histograms from experimental data.
exp_hist_Exc = np.histogram(exp_rates_Exc, bins=hist_bins)[0] / (1.0 * exp_rates_Exc.size)
exp_hist_Inh = np.histogram(exp_rates_Inh, bins=hist_bins)[0] / (1.0 * exp_rates_Inh.size)

# Plot the histograms.
fig, axes = plt.subplots(1, 2)
for sys_name in sim_hist_Exc.keys():
    axes[0].plot(hist_bins[:-1], sim_hist_Exc[sys_name], label=sys_name)
axes[0].plot(hist_bins[:-1], exp_hist_Exc, label='Exp.')
for sys_name in sim_hist_Exc.keys():
    axes[1].plot(hist_bins[:-1], sim_hist_Inh[sys_name], label=sys_name)
axes[1].plot(hist_bins[:-1], exp_hist_Inh, label='Exp.')
for i in [0, 1]:
    axes[i].set_xlim((-0.1, 50.0))
    axes[i].set_ylim((-0.001, 0.35))
    axes[i].set_xlabel('Rmax (Hz)')
    axes[i].legend()
axes[0].set_ylabel('Fraction of cells')
axes[0].set_title('Exc. neurons.')
axes[1].set_title('Inh. neurons.')
plt.show()


# Load Rmax from simulations and narrow down to only the biophysical cells.
df_rates = pd.read_csv(result_fname, sep=' ')
df_rates_tmp = df_rates[df_rates['cell_type'].isin(['Scnn1a', 'Rorb', 'Nr5a1', 'PV1', 'PV2'])]

# Plot the average Rmax (over cells).
ax = df_rates_tmp.pivot(index='cell_type', columns='system', values='av_Rmax').reindex(['Scnn1a', 'Rorb', 'Nr5a1', 'PV1', 'PV2']).plot(kind='bar', yerr=df_rates_tmp.pivot(index='cell_type', columns='system', values='std_Rmax').reindex(['Scnn1a', 'Rorb', 'Nr5a1', 'PV1', 'PV2']), fontsize=20)

# Add bars for the experimental data.
x_exp = np.array([5, 5.3, 6, 6.3])
ax.bar(x_exp, exp_data_mean, yerr=exp_data_std, width=0.2, color='gray', error_kw=dict(ecolor='k'))
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
