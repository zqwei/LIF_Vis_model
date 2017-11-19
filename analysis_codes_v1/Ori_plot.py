import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd


result_fig_prefix = 'Ori/Ori_'
result_fname = 'Ori/av_tuning.csv'
result_fig_CV = 'Ori/CV_ori_av.eps'
result_fig_DSI = 'Ori/DSI_av.eps'

cell_db_path = '/allen/aibs/mat/antona/network/14-simulations/9-network/build/'
# Decide which systems we are doing analysis for.
sys_dict = {}
#sys_dict['ll1'] = { 'cells_file': '../build/ll1.csv', 'f_1': '../simulations_ll1/output_ll1_', 'f_2': '_sd278/spk.dat', 'f_3': '_sd278/tot_f_rate.dat', 'f_out': 'Ori/ll1_rates.npy', 'f_out_pref': 'Ori/ll1_pref_stat.csv', 'grating_id': range(6, 240, 30)+range(7, 240, 30)+range(8, 240, 30)+range(9, 240, 30) }
#sys_dict['ll2'] = { 'cells_file': '../build/ll2.csv', 'f_1': '../output_ll2_', 'f_2': '_sd278/spk.dat', 'f_3': '_sd278/tot_f_rate.dat', 'f_out': 'Ori/ll2_rates.npy', 'f_out_pref': 'Ori/ll2_pref_stat.csv', 'grating_id': range(7, 240, 30)+range(8, 240, 30)+range(9, 240, 30) }
#sys_dict['ll3'] = { 'cells_file': '../build/ll3.csv', 'f_1': '../output_ll3_', 'f_2': '_sd278/spk.dat', 'f_3': '_sd278/tot_f_rate.dat', 'f_out': 'Ori/ll3_rates.npy', 'f_out_pref': 'Ori/ll3_pref_stat.csv', 'grating_id': range(8, 240, 30) }
#sys_dict['rl2'] = { 'cells_file': '../build/rl2.csv', 'f_out': 'Ori/rl2_rates.npy', 'f_out_pref': 'Ori/rl2_pref_stat.csv' }
#sys_dict['lr2'] = { 'cells_file': '../build/lr2.csv', 'f_out': 'Ori/lr2_rates.npy', 'f_out_pref': 'Ori/lr2_pref_stat.csv' }
#sys_dict['rr2'] = { 'cells_file': '../build/rr2.csv', 'f_out': 'Ori/rr2_rates.npy', 'f_out_pref': 'Ori/rr2_pref_stat.csv' }
#sys_dict['ll2_TF4Hz'] = { 'cells_file': '../build/ll2.csv', 'f_out': 'Ori/ll2_rates_4Hz.npy', 'f_out_pref': 'Ori/ll2_pref_stat_4Hz.csv' }
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
sys_dict['ll2_TF4Hz'] = { 'cells_file': cell_db_path+'ll2.csv',
                    'f_1': '../simulation_ll2/output_ll2_',
                    'f_2': '_sdlif_z101/spk.dat',
                    'f_3': '_sdlif_z101/tot_f_rate.dat',
                    'f_out': 'Ori/ll2_rates_4Hz.npy',
                    'f_out_std': 'Ori/ll2_rates_std_4Hz.npy',
                    'f_out_pref': 'Ori/ll2_pref_stat_4Hz.csv',
                    'grating_id': range(8, 240, 30)}

# Load simulation data.
sim_data = {}
for sys_name in sys_dict.keys():
    sim_data[sys_name] = pd.read_csv(sys_dict[sys_name]['f_out_pref'], sep=' ')

# Load data from the experiment.
exp_f_list = ['ANL4Exc.csv', 'AWL4Exc.csv', 'ANInh.csv', 'AWInh.csv'] #'ANL4Inh.csv', 'AWL4Inh.csv']
exp_labels = ['An. L4 Exc.', 'Aw. L4 Exc.', 'An. Inh.', 'Aw. Inh.'] #'An. L4 Inh.', 'Aw. L4 Inh.']
exp_data = {}
for i_exp, exp_f in enumerate(exp_f_list):
    exp_data[exp_labels[i_exp]] = pd.read_csv('/allen/aibs/mat/antona/experimental_data/ephys_Sev/2016_paper_data/gratings/' + exp_f, sep=',')


# Plot distributions.
hist_col_name = 'CV_ori'
hist_bins = np.linspace(0.0, 1.0, 18)
#hist_col_name = 'TF'
#hist_bins = np.linspace(0.0, 20.0, 11)
#hist_col_name = 'SF'
#hist_bins = np.linspace(0.0, 0.5, 31)

# Create histograms for simulation data.
sim_hist_Exc = {}
sim_hist_Inh = {}
for sys_name in sys_dict.keys():
    sys_data = pd.read_csv(sys_dict[sys_name]['cells_file'], sep=' ')
    Exc_ids = sys_data[sys_data['type'].isin(['Scnn1a', 'Rorb', 'Nr5a1'])]['index'].values
    Inh_ids = sys_data[sys_data['type'].isin(['PV1', 'PV2'])]['index'].values
    tmp_Exc = sim_data[sys_name][sim_data[sys_name]['id'].isin(Exc_ids)][hist_col_name].values
    tmp_Inh = sim_data[sys_name][sim_data[sys_name]['id'].isin(Inh_ids)][hist_col_name].values
    sim_hist_Exc[sys_name] = np.histogram(tmp_Exc, bins=hist_bins)[0] / (1.0 * tmp_Exc.size)
    sim_hist_Inh[sys_name] = np.histogram(tmp_Inh, bins=hist_bins)[0] / (1.0 * tmp_Inh.size)

#Create histograms from experimental data.
exp_hist = {}
for label in exp_labels:
    if (hist_col_name in ['SF', 'TF']):
        exp_col_name = 'Pref' + hist_col_name
    else:
        exp_col_name = hist_col_name
    tmp_exp = exp_data[label][exp_col_name].values
    exp_hist[label] = np.histogram(tmp_exp, bins=hist_bins)[0] / (1.0 * tmp_exp.size)

# Plot the histograms.
fig, axes = plt.subplots(1, 2)
for sys_name in sim_hist_Exc.keys():
    axes[0].plot(hist_bins[:-1], sim_hist_Exc[sys_name], label=sys_name)
    axes[1].plot(hist_bins[:-1], sim_hist_Inh[sys_name], label=sys_name)

for label in exp_labels:
    if ('Exc' in label):
        axes_id = 0
    else:
        axes_id = 1
    axes[axes_id].plot(hist_bins[:-1], exp_hist[label], label=label)
for i in [0, 1]:
    axes[i].set_xlim((-0.003, 1.0))
    axes[i].set_xlabel('CV_ori', fontsize=17)
    axes[i].legend()
    [x.set_fontsize(17) for x in (axes[i].get_xticklabels() + axes[i].get_yticklabels())]
axes[0].set_ylim((-0.001, 0.35))
axes[1].set_ylim((-0.001, 0.5))
axes[0].set_ylabel('Fraction of cells', fontsize=17)
axes[0].set_title('Exc. neurons.', fontsize=17)
axes[1].set_title('Inh. neurons.', fontsize=17)
plt.savefig('%s%s_hist.eps' % (result_fig_prefix, hist_col_name), format='eps')
plt.show()



# Compute averages and standard deviations of CV_ori, OSI_modulation, and DSI by type and save to file.
result_f = open(result_fname, 'w')
result_f.write('system cell_type av_CV_ori std_CV_ori av_OSI_modulation std_OSI_modulation av_DSI std_DSI\n')
for sys_name in sys_dict.keys():
    sys_data = pd.read_csv(sys_dict[sys_name]['cells_file'], sep=' ')
    for type in list(set(list(sys_data['type']))):
        type_ids = sys_data[sys_data['type'] == type]['index'].values
        tmp = sim_data[sys_name][sim_data[sys_name]['id'].isin(type_ids)]
        result_string = '%s %s %f %f %f %f %f %f' % (sys_name, type, tmp['CV_ori'].mean(), tmp['CV_ori'].std(), tmp['OSI_modulation'].mean(), tmp['OSI_modulation'].std(), tmp['DSI'].mean(), tmp['DSI'].std())
        result_f.write(result_string + '\n')
        # print result_string
result_f.close()

# Load average data from simulations and narrow down to only the biophysical cells.
df_sim = pd.read_csv(result_fname, sep=' ')
df_sim_tmp = df_sim[df_sim['cell_type'].isin(['Scnn1a', 'Rorb', 'Nr5a1', 'PV1', 'PV2'])]

# Plot the average CV_ori (over cells).
ax = df_sim_tmp.pivot(index='cell_type', columns='system', values='av_CV_ori').reindex(['Scnn1a', 'Rorb', 'Nr5a1', 'PV1', 'PV2']).plot(kind='bar', yerr=df_sim_tmp.pivot(index='cell_type', columns='system', values='std_CV_ori').reindex(['Scnn1a', 'Rorb', 'Nr5a1', 'PV1', 'PV2']), fontsize=20)

# Add bars for the experimental data.
x_exp = np.array([5, 5.3, 6, 6.3])
exp_data_mean = []
exp_data_std = []
for i_exp, label in enumerate(exp_labels):
    exp_data_mean.append(exp_data[label]['CV_ori'].mean())
    exp_data_std.append(exp_data[label]['CV_ori'].std())

ax.bar(x_exp, exp_data_mean, yerr=exp_data_std, width=0.2, color='gray', error_kw=dict(ecolor='k'))
labels = [item.get_text() for item in ax.get_xticklabels()] # Make sure this is done before xticks are extended; otherwise, the labels list will contain more empty entries.
labels = labels + exp_labels
ax.set_xticks(list(ax.get_xticks()) + list(x_exp))
ax.set_xticklabels(labels)

ax.set_ylabel('CV_ori', fontsize=20)
ax.set_xlim([-0.5, 7.0])
ax.set_ylim(bottom=0.0)
plt.gcf().subplots_adjust(bottom=0.3)

#ax.annotate('Niell and Stryker, (2008).\n L4 Exc.: ~6 Hz\n Inh.: ~13 Hz', xy=(0.5, 0.7), xycoords='axes fraction', fontsize=16,
#                xytext=(-5, 5), textcoords='offset points',
#                ha='right', va='bottom')

plt.savefig(result_fig_CV, format='eps')
plt.show()


# Plot the average DSI (over cells).
ax = df_sim_tmp.pivot(index='cell_type', columns='system', values='av_DSI').reindex(['Scnn1a', 'Rorb', 'Nr5a1', 'PV1', 'PV2']).plot(kind='bar', yerr=df_sim_tmp.pivot(index='cell_type', columns='system', values='std_DSI').reindex(['Scnn1a', 'Rorb', 'Nr5a1', 'PV1', 'PV2']), fontsize=20)

# Add bars for the experimental data.
x_exp = np.array([5, 5.3, 6, 6.3])
exp_data_mean = []
exp_data_std = []
for i_exp, label in enumerate(exp_labels):
    exp_data_mean.append(exp_data[label]['DSI'].mean())
    exp_data_std.append(exp_data[label]['DSI'].std())

ax.bar(x_exp, exp_data_mean, yerr=exp_data_std, width=0.2, color='gray', error_kw=dict(ecolor='k'))
labels = [item.get_text() for item in ax.get_xticklabels()] # Make sure this is done before xticks are extended; otherwise, the labels list will contain more empty entries.
labels = labels + exp_labels
ax.set_xticks(list(ax.get_xticks()) + list(x_exp))
ax.set_xticklabels(labels)

ax.set_ylabel('DSI', fontsize=20)
ax.set_xlim([-0.5, 7.0])
ax.set_ylim(bottom=0.0)
plt.gcf().subplots_adjust(bottom=0.3)

#ax.annotate('Niell and Stryker, (2008).\n L4 Exc.: ~6 Hz\n Inh.: ~13 Hz', xy=(0.5, 0.7), xycoords='axes fraction', fontsize=16,
#                xytext=(-5, 5), textcoords='offset points',
#                ha='right', va='bottom')

plt.savefig(result_fig_DSI, format='eps')
plt.show()
