import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams.update({'font.size': 12})

'''
f_dict = {}
f_dict['ll2'] = '/data/mat/antona/network/14-simulations/9-network/simulations_ll2/gratings/output_ll2_g8_8_sd278/tot_f_rate.dat'
f_dict['LIF_ll2'] = '/data/mat/ZiqiangW/simulation_ll_final_syn_data_lif_z102/simulation_ll2/output_ll2_g8_8_sdlif_z101/tot_f_rate.dat'
f_dict['LIF_unused_ll2'] = '/data/mat/ZiqiangW/unused_simulation/simulation_ll_syn_data_lif_z101/simulation_ll2/output_ll2_g8_8_sdlif_z101/tot_f_rate.dat'
for sys in f_dict:
    f_name = f_dict[sys]
    df = pd.read_csv(f_name, sep=' ', header=None)
    df.columns = ['gid', 'frate_w', 'frate_raw']
    plt.plot(df['gid'], pd.rolling_mean(df['frate_w'], window=100, min_periods=1), label=sys)
plt.legend()
plt.xlim([0, 10000])
plt.show()
quit()
'''

plot_types_color = {'Scnn1a': 'darkorange', 'Rorb': 'red', 'Nr5a1': 'magenta', 'PV1': 'blue', 'PV2': 'cyan'}

cells_file='../build/ll2.csv'
cells_df = pd.read_csv(cells_file, sep=' ')
gids = cells_df['index'].values

pop_gids = {}
for type in plot_types_color:
    pop_gids[type] = cells_df[cells_df['type'] == type]['index'].values

#t0 = 500.0
#t1 = 1000.0
#gid_av = 100

#spk_fname = '/data/mat/ZiqiangW/unused_simulation/simulation_ll_syn_data_lif_z101/simulation_ll2/output_ll2_g8_8_sdlif_z101/spk.dat'
#out_fig_name = 'plt_f_rate_from_spk_select_t/LIF_unused_simulation_ll2_g8_8_t500_to_1000_run_av_100.eps'

#spk_fname = '/data/mat/ZiqiangW/simulation_ll_final_syn_data_lif_z102/simulation_ll2/output_ll2_g8_8_sdlif_z101/spk.dat'
#out_fig_name = 'plt_f_rate_from_spk_select_t/LIF_ll2_g8_8_t500_to_1000_run_av_100.eps'

#spk_fname = '/data/mat/antona/network/14-simulations/9-network/simulations_ll2/gratings/output_ll2_g8_8_sd278/spk.dat'
#out_fig_name = 'plt_f_rate_from_spk_select_t/ll2_g8_8_t500_to_1000_run_av_100.eps'

t0 = 2000.0
t1 = 2500.0
gid_av = 100

#spk_fname = '/data/mat/ZiqiangW/unused_simulation/simulation_ll_syn_data_lif_z101/simulation_ll2/output_ll2_g8_8_sdlif_z101/spk.dat'
#out_fig_name = 'plt_f_rate_from_spk_select_t/LIF_unused_simulation_ll2_g8_8_t2000_to_2500_run_av_100.eps'

#spk_fname = '/data/mat/ZiqiangW/simulation_ll_final_syn_data_lif_z102/simulation_ll2/output_ll2_g8_8_sdlif_z101/spk.dat'
#out_fig_name = 'plt_f_rate_from_spk_select_t/LIF_ll2_g8_8_t2000_to_2500_run_av_100.eps'

spk_fname = '/data/mat/antona/network/14-simulations/9-network/simulations_ll2/gratings/output_ll2_g8_8_sd278/spk.dat'
out_fig_name = 'plt_f_rate_from_spk_select_t/ll2_g8_8_t2000_to_2500_run_av_100.eps'





df = pd.read_csv(spk_fname, sep=' ', header=None)
df.columns = ['t', 'gid']

df = df[df['t'] >= t0]
df = df[df['t'] <= t1]

rates = df.groupby('gid').count() * 1000.0 / (t1 - t0) # Time is in ms and rate is in Hz.
rates.columns = ['rates']

# The 'gid' label is now used as index (after the groupby operation).
# Convert it to a column; then change the index name to none, as in default.
rates['gid'] = rates.index
rates.index.names = ['']

# Find cell IDs from the cell file that do not have counterparts in the spk file
# (for example, because those cells did not fire).
# Add these cell IDs to the dataframe; fill rates with zeros.
gids_not_in_spk = gids[~np.in1d(gids, rates['gid'].values)]
rates = rates.append(pd.DataFrame(np.array([gids_not_in_spk, np.zeros(gids_not_in_spk.size)]).T, columns=['gid', 'rates']))

# Sort the rows according to the cell IDs.
rates = rates.sort('gid', ascending=True)

rates_run_av = rates
tmp = pd.rolling_mean(rates['rates'], window=gid_av, min_periods=1) # This returns a series object, since the rolling mean is performed on just one column.
rates_run_av['rates'] = tmp.values # The series object does not have columns (it's just one column with indices).


fig, ax = plt.subplots()

for type in pop_gids:
    gids_current = pop_gids[type]
    frate = rates_run_av[rates_run_av['gid'].isin(gids_current)]['rates'].values
    ax.plot(gids_current, frate, c=plot_types_color[type])

ax.set_ylim([0.0, 25.0])

plt.savefig(out_fig_name, format='eps')
plt.show()


