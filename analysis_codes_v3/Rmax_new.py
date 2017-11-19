import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scipy.stats as scp_stats
import pandas as pd

N_trials = 10

cell_db_path = '/allen/aibs/mat/antona/network/14-simulations/9-network/analysis/'

# Decide which systems we are doing analysis for.
sys_dict = {}
# sys_dict['ll1'] = { 'f_out': cell_db_path + 'Rmax/ll1_Rmax.csv', 'grating_ids': range(6, 240, 30)+range(7, 240, 30)+range(8, 240, 30)+range(9, 240, 30) }
# sys_dict['ll2'] = { 'f_out': cell_db_path + 'Rmax/ll2_Rmax.csv', 'grating_ids': range(7, 240, 30)+range(8, 240, 30)+range(9, 240, 30) }
# sys_dict['ll3'] = { 'f_out': cell_db_path + 'Rmax/ll3_Rmax.csv', 'grating_ids': range(8, 240, 30) }

# sys_dict['ll1_LIF'] = { 'f_out': '../analysis_intFire1/analysis_ll/Rmax/ll1_Rmax.csv', 'grating_ids': range(7, 240, 30)+range(8, 240, 30)+range(9, 240, 30) }
# sys_dict['ll2_LIF'] = { 'f_out': '../analysis_intFire1/analysis_ll/Rmax/ll2_Rmax.csv', 'grating_ids': range(7, 240, 30)+range(8, 240, 30)+range(9, 240, 30) }
# sys_dict['ll3_LIF'] = { 'f_out': '../analysis_intFire1/analysis_ll/Rmax/ll3_Rmax.csv', 'grating_ids': range(8, 240, 30) }

sys_dict['ll1_LIF'] = { 'f_out': '../analysis_intFire4/analysis_ll/Rmax/ll1_Rmax.csv', 'grating_ids': range(7, 240, 30)+range(8, 240, 30)+range(9, 240, 30) }
sys_dict['ll2_LIF'] = { 'f_out': '../analysis_intFire4/analysis_ll/Rmax/ll2_Rmax.csv', 'grating_ids': range(7, 240, 30)+range(8, 240, 30)+range(9, 240, 30) }
sys_dict['ll3_LIF'] = { 'f_out': '../analysis_intFire4/analysis_ll/Rmax/ll3_Rmax.csv', 'grating_ids': range(8, 240, 30) }

# result_fname_prefix = 'Rmax/Rmax_bio_ll'
# result_fname_prefix = 'Rmax/Rmax_lif1_ll'
result_fname_prefix = 'Rmax/Rmax_lif4_ll'
result_fig_fname = result_fname_prefix + '.eps'

type_color = {'Scnn1a': 'darkorange', 'Rorb': 'red', 'Nr5a1': 'magenta', 'PV1': 'blue', 'PV2': 'cyan', 'AnL4E': 'gray', 'AwL4E': 'gray', 'AnI': 'gray', 'AwI': 'gray'}
type_order = ['Scnn1a', 'Rorb', 'Nr5a1', 'AnL4E', 'AwL4E', 'PV1', 'PV2', 'AnI', 'AwI']

# Read files with Rmax from simulations.
sim_df = pd.DataFrame()
for sys_name in sys_dict.keys():
    tmp_df = pd.read_csv(sys_dict[sys_name]['f_out'], sep=' ')
    # Combine Rmax from all systems into one file.
    sim_df = pd.concat([sim_df, tmp_df], axis=0)


# Read files with Rmax from experiments.
exp_f = { 'AnL4E': '/allen/aibs/mat/antona/experimental_data/ephys_Sev/2016_paper_data/gratings/ANL4Exc.csv',
          'AwL4E': '/allen/aibs/mat/antona/experimental_data/ephys_Sev/2016_paper_data/gratings/AWL4Exc.csv',
          'AnI': '/allen/aibs/mat/antona/experimental_data/ephys_Sev/2016_paper_data/gratings/ANInh.csv',
          'AwI': '/allen/aibs/mat/antona/experimental_data/ephys_Sev/2016_paper_data/gratings/AWInh.csv' }
exp_df = pd.DataFrame()
for exp_key in exp_f:
    tmp_df = pd.read_csv(exp_f[exp_key], sep=',')
    tmp_df['type'] = exp_key
    tmp_df['gid'] = -1
    exp_df = pd.concat([exp_df, tmp_df], axis=0)

exp_df_1 = pd.DataFrame()
exp_df_1['gid'] = exp_df['gid'].values
exp_df_1['type'] = exp_df['type'].values
exp_df_1['Rmax'] = exp_df['Rmax'].values

tot_df = pd.concat([sim_df, exp_df_1], axis=0)


# Arrange data into a list of numpy arrays.
type_data = []
for type_key in type_order:
    type_data.append(tot_df[tot_df['type']==type_key]['Rmax'].values)

# Plot the results.
y_lim_top = 27.0

fig, ax = plt.subplots(figsize = (7, 5))

box = ax.boxplot(type_data, patch_artist=True, sym='c.') # notch=True
for patch, color in zip(box['boxes'], [type_color[type_key] for type_key in type_order]):
    patch.set_facecolor(color)

for i, type_key in enumerate(type_order):
    ax.errorbar([i+1], [type_data[i].mean()], yerr=[type_data[i].std() / np.sqrt(1.0 * type_data[i].size)], marker='o', ms=8, color='k', linewidth=2, capsize=5, markeredgewidth=2, ecolor='k', elinewidth=2)
    ind = np.where(type_data[i] > y_lim_top)[0]
    ax.annotate(u'$\u2191$'+'\n%d/%d' % (ind.size, type_data[i].size), xy=(i+1.2, 1.0*y_lim_top), fontsize=12)

ax.set_ylim((0.0, y_lim_top))
ax.set_xticks(range(1, len(type_order)+1))
ax.set_xticklabels(type_order)

ax.set_ylabel('Rmax (Hz)')

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
ax.tick_params(size=10)

plt.savefig(result_fig_fname, format='eps')

plt.show()
