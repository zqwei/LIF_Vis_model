import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import pandas as pd

import f_rate_t_by_type_functions as frtbt


bin_start = 0.0
bin_stop = 1500.0
bin_size = 20.0

N_trials = 20

list_of_types = []

result_fname_prefix = 'spont_activity/av_spont_rates_by_type'
result_fname = result_fname_prefix + '.csv'
result_fig_fname = result_fname_prefix + '.eps'
t_av_start = 500.0
t_av_stop = 1000.0

cell_db_path = '/data/mat/antona/network/14-simulations/9-network/build/'
# Decide which systems we are doing analysis for.
sys_dict = {}
sys_dict['ll1'] = { 'cells_file': cell_db_path+'ll1.csv',
										'f_1': '../simulation_ll1/output_ll1_spont_',
										'f_2': '_sdlif_z101/spk.dat',
										'f_3': '_sdlif_z101/tot_f_rate.dat',
										'f_out': 'spont_activity/ll1_spont.pkl',
										'types': [] }
sys_dict['ll2'] = { 'cells_file': cell_db_path+'ll2.csv',
										'f_1': '../simulation_ll2/output_ll2_spont_',
										'f_2': '_sdlif_z101/spk.dat',
										'f_3': '_sdlif_z101/tot_f_rate.dat',
										'f_out': 'spont_activity/ll2_spont.pkl',
										'types': [] }
sys_dict['ll3'] = { 'cells_file': cell_db_path+'ll3.csv',
										'f_1': '../simulation_ll3/output_ll3_spont_',
										'f_2': '_sdlif_z101/spk.dat',
										'f_3': '_sdlif_z101/tot_f_rate.dat',
										'f_out': 'spont_activity/ll3_spont.pkl',
										'types': [] }
#sys_dict['rr2'] = { 'cells_file': '../build/rr2.csv', 'f_1': '../output_rr2_spont_', 'f_2': '_sd282_cn0/spk.dat', 'f_3': '_sd282_cn0/tot_f_rate.dat', 'f_out': 'spont_activity/rr2_spont.pkl', 'types': [] }


for sys_name in sys_dict.keys():
    # Obtain information about cell types.
    gids_by_type = frtbt.construct_gids_by_type_dict(sys_dict[sys_name]['cells_file'])
    sys_dict[sys_name]['types'] = gids_by_type.keys()

    # Process the spike files and save computed firing rates in files.
    f_list = []
    for i_trial in xrange(0, N_trials):
        f_list.append('%s%d%s' % (sys_dict[sys_name]['f_1'], i_trial, sys_dict[sys_name]['f_2']))
    frtbt.f_rate_t_by_type(gids_by_type, bin_start, bin_stop, bin_size, f_list, sys_dict[sys_name]['f_out'])


# Compute averages and standard deviations by type and save to file.
result_f = open(result_fname, 'w')
result_f.write('system cell_type av_rate std\n')

for sys_name in sys_dict.keys():
    f = open(sys_dict[sys_name]['f_out'], 'r')
    rates_data = pickle.load(f)
    f.close()

    ind = np.intersect1d( np.where( rates_data['t_f_rate'] > t_av_start ), np.where( rates_data['t_f_rate'] < t_av_stop ) )

    for type in sys_dict[sys_name]['types']:
        result_string = '%s %s %f %f' % (sys_name, type, rates_data['mean'][type][ind].mean(), rates_data['mean'][type][ind].std())
        result_f.write(result_string + '\n')
        # print result_string

result_f.close()


# Plot time series of average (over cells) firing rates by type.
rates_dict = {}
for sys_name in sys_dict.keys():
    f = open(sys_dict[sys_name]['f_out'], 'r')
    rates_dict[sys_name] = pickle.load(f)
    f.close()


for type in sys_dict[sys_dict.keys()[0]]['types']: # Use types from the first system; assume that all systems have those types.
    for sys_name in sys_dict.keys():
        plt.plot(rates_dict[sys_name]['t_f_rate'], rates_dict[sys_name]['mean'][type], label=sys_name)
    plt.xlabel('t (ms)')
    plt.ylabel('Mean firing rate (Hz)')
    plt.legend()
    plt.title('Type %s' % (type))
    plt.show()


# Load the firing rates from the experiment.
exp_f_list = ['ANL4Exc.csv', 'AWL4Exc.csv', 'ANInh.csv', 'AWInh.csv'] #'ANL4Inh.csv', 'AWL4Inh.csv']
exp_labels = ['An. L4 Exc.', 'Aw. L4 Exc.', 'An. Inh.', 'Aw. Inh.'] #'An. L4 Inh.', 'Aw. L4 Inh.']

exp_rates_Exc = np.array([])
exp_rates_Inh = np.array([])

exp_data_mean = []
exp_data_std = []
for i_exp, exp_f in enumerate(exp_f_list):
    tmp_df = pd.read_csv('/data/mat/antona/experimental_data/ephys_Sev/2016_paper_data/gratings/' + exp_f, sep=',')
    exp_data_mean.append(tmp_df['Spont'].mean())
    exp_data_std.append(tmp_df['Spont'].std())

    if ('Exc' in exp_labels[i_exp]):
        exp_rates_Exc = np.append(exp_rates_Exc, tmp_df['Spont'].values)
    else:
        exp_rates_Inh = np.append(exp_rates_Inh, tmp_df['Spont'].values)

#ax = plt.subplot(111)
#ax.bar(range(len(exp_data_mean)), exp_data_mean, yerr=exp_data_std)
#ax.set_xticks(np.arange(len(exp_data_mean)) + 0.5)
#ax.set_xticklabels(exp_labels)
#plt.show()



# Plot distributions of spontaneous firing rates.
hist_bins = np.linspace(0.0, 100.0, 101)

# Load simulation data and create histograms.
sim_hist_Exc = {}
sim_hist_Inh = {}
gids_Exc = np.concatenate( (gids_by_type['Scnn1a'], gids_by_type['Rorb'], gids_by_type['Nr5a1']) )
gids_Inh = np.concatenate( (gids_by_type['PV1'], gids_by_type['PV2']) )

for sys_name in sys_dict.keys():
    rates_tmp = np.array([])
    for i_trial in xrange(0, N_trials):
        f_name = '%s%d%s' % (sys_dict[sys_name]['f_1'], i_trial, sys_dict[sys_name]['f_3'])
        # print 'Processing file %s.' % (f_name)
        tmp = np.genfromtxt(f_name, delimiter=' ')[:, 1] # Assume all files have the same columns of gids; use the 2nd column for rates.
        if (rates_tmp.size == 0):
            rates_tmp = tmp
        else:
            rates_tmp = rates_tmp + tmp
    rates_tmp = rates_tmp / (1.0 * N_trials)
    sim_hist_Exc[sys_name] = np.histogram(rates_tmp[gids_Exc], bins=hist_bins)[0] / (1.0 * gids_Exc.size)
    sim_hist_Inh[sys_name] = np.histogram(rates_tmp[gids_Inh], bins=hist_bins)[0] / (1.0 * gids_Inh.size)

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
    axes[i].set_xlim((-0.1, 15.0))
    axes[i].set_ylim((-0.001, 0.15))
    axes[i].set_xlabel('Spont. rate (Hz)')
    axes[i].legend()
axes[0].set_ylabel('Fraction of cells')
axes[0].set_title('Exc. neurons.')
axes[1].set_title('Inh. neurons.')
plt.show()


# Load the average spontaneous rates from simulations and narrow down to only the biophysical cells.
df_rates = pd.read_csv(result_fname, sep=' ')
df_rates_tmp = df_rates[df_rates['cell_type'].isin(['Scnn1a', 'Rorb', 'Nr5a1', 'PV1', 'PV2'])]

# Plot the average firing rates (over cells and time).
ax = df_rates_tmp.pivot(index='cell_type', columns='system', values='av_rate').reindex(['Scnn1a', 'Rorb', 'Nr5a1', 'PV1', 'PV2']).plot(kind='bar', yerr=df_rates_tmp.pivot(index='cell_type', columns='system', values='std').reindex(['Scnn1a', 'Rorb', 'Nr5a1', 'PV1', 'PV2']), fontsize=20)

# Add bars for the experimental data.
x_exp = np.array([5, 5.3, 6, 6.3])
ax.bar(x_exp, exp_data_mean, yerr=exp_data_std, width=0.2, color='gray', error_kw=dict(ecolor='k'))
labels = [item.get_text() for item in ax.get_xticklabels()] # Make sure this is done before xticks are extended; otherwise, the labels list will contain more empty entries.
labels = labels + exp_labels
ax.set_xticks(list(ax.get_xticks()) + list(x_exp))
ax.set_xticklabels(labels)

ax.set_ylabel('Spontaneous rate (Hz)', fontsize=20)
ax.set_xlim([-0.5, 7.0])
ax.set_ylim(bottom=0.0)
plt.gcf().subplots_adjust(bottom=0.3)

ax.annotate('Niell and Stryker, (2008).\n L4 Exc.: ~0.25 Hz\n Inh.: ~1.25 Hz', xy=(0.68, 0.7), xycoords='axes fraction', fontsize=16,
                xytext=(-5, 5), textcoords='offset points',
                ha='right', va='bottom')

plt.savefig(result_fig_fname, format='eps')

plt.show()
