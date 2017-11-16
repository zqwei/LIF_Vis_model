import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib
matplotlib.rcParams.update({'font.size': 20})
from matplotlib import gridspec

'''
# Process firing rates.
N_trials = 10
for sys in ['ll1', 'll2', 'll3']:
    for grating_id in [8]:
        f_rate = np.array([])
        for trial_id in xrange(N_trials):
            df = pd.read_csv('../simulations_%s/gratings/output_%s_g%d_%d_sd278/tot_f_rate.dat' % (sys, sys, grating_id, trial_id), sep=' ')
            df.columns = ['gid', 'frate', 'frate_include_equilibration']
            del df['frate_include_equilibration']
            if (f_rate.size == 0):
                f_rate = df['frate'].values
            else:
                f_rate = f_rate + df['frate'].values
        df['frate'] = f_rate / (1.0 * N_trials)
        df.to_csv('log_normal_rates/f_rate_mean_%s_g%d.csv' % (sys, grating_id), sep=' ', index=False)
'''

# Load the firing rates from simulations.
sys_dict = {}
sys_dict['ll1'] = {'g8': 'log_normal_rates/f_rate_mean_ll1_g8.csv', 'spont': 'spont_activity/ll1_spont.csv'}
sys_dict['ll2'] = {'g8': 'log_normal_rates/f_rate_mean_ll2_g8.csv', 'spont': 'spont_activity/ll2_spont.csv'}
sys_dict['ll3'] = {'g8': 'log_normal_rates/f_rate_mean_ll3_g8.csv', 'spont': 'spont_activity/ll3_spont.csv'}
sim_df = pd.DataFrame()
for sys in sys_dict:
    df1 = pd.read_csv(sys_dict[sys]['g8'], sep=' ')
    df1 = df1.rename(columns={'frate': 'g8'})
    df2 = pd.read_csv(sys_dict[sys]['spont'], sep=' ')
    df2 = df2.rename(columns={'%s_frate' % (sys): 'Spont'})
    df = pd.merge(df1, df2, on='gid', how='inner')
    df['model'] = sys
    sim_df = pd.concat([sim_df, df], axis=0)


tmp_sim_df = sim_df[sim_df['type'].isin(['Scnn1a', 'Rorb', 'Nr5a1', 'PV1', 'PV2'])]
sim_df_pos = tmp_sim_df[tmp_sim_df['Spont']!=0.0]

bins = np.arange(-0.001, 100.0, 0.1)
av_bin_rate = np.zeros(bins.size - 1)
for i, bin in enumerate(bins[:-1]):
    bin1 = bins[i+1]
    df1 = tmp_sim_df[tmp_sim_df['Spont']>=bin]
    df1 = df1[df1['Spont']<bin1]
    if (df1.shape[0] > 0):
        av_bin_rate[i] = df1['g8'].mean()

# Extract only the non-zero entries from the bin-averaged firing rate.
ind = np.where(av_bin_rate > 0.0)
av_bin_rate1 = av_bin_rate[ind]
bins1 = bins[ind]

fig = plt.figure(figsize=(8, 12)) 
gs = gridspec.GridSpec(1, 2, width_ratios=[1, 5]) 
ax = [plt.subplot(gs[0]), plt.subplot(gs[1])]

ax[1].scatter(sim_df_pos['Spont'], sim_df_pos['g8'], c='firebrick', s=2, lw=0)
ax[1].scatter(bins1, av_bin_rate1, c='red', s=50, lw=0)
ax[1].set_xlim((0.08, 60.0))
ax[1].set_yscale('log')
ax[1].set_xscale('log')

sim_df_0 = tmp_sim_df[tmp_sim_df['Spont']==0.0]
ax[0].scatter(sim_df_0['Spont'], sim_df_0['g8'], c='firebrick', s=3, lw=0)
ax[0].scatter([0.0], [av_bin_rate[0]], c='red', s=50, lw=0)
ax[0].set_yscale('log')

ax[0].set_xticks([0.0])
ax[1].set_yticklabels([])

ax[1].set_xlabel('Spont. rate (Hz)')
ax[0].set_ylabel('Drifting grating response (Hz)')

ax[0].set_ylim((0.03, 60.0))
ax[1].set_ylim((0.03, 60.0))

plt.tight_layout()
plt.savefig('log_normal_rates/log_normal_rates_spont_g8.eps', format='eps')
plt.show()


'''
# Load the firing rates from simulations.
sys_dict = {}
sys_dict['ll1'] = {'Rmax': 'Rmax/ll1_Rmax.csv', 'spont': 'spont_activity/ll1_spont.csv'}
sys_dict['ll2'] = {'Rmax': 'Rmax/ll2_Rmax.csv', 'spont': 'spont_activity/ll2_spont.csv'}
sys_dict['ll3'] = {'Rmax': 'Rmax/ll3_Rmax.csv', 'spont': 'spont_activity/ll3_spont.csv'}
sim_df = pd.DataFrame()
for sys in sys_dict:
    df1 = pd.read_csv(sys_dict[sys]['Rmax'], sep=' ')
    df2 = pd.read_csv(sys_dict[sys]['spont'], sep=' ')
    df2 = df2.rename(columns={'%s_frate' % (sys): 'Spont'})
    df = pd.merge(df1, df2, on=['gid', 'type'], how='inner')
    df['model'] = sys
    sim_df = pd.concat([sim_df, df], axis=0)

# Load the firing rates from the experiment.
exp_f_list = ['ANL4Exc.csv', 'AWL4Exc.csv'] #, 'ANInh.csv', 'AWInh.csv']
exp_df = pd.DataFrame()
for exp_f in exp_f_list:
    tmp_df = pd.read_csv('/allen/aibs/mat/antona/experimental_data/ephys_Sev/2016_paper_data/gratings/' + exp_f, sep=',')
    exp_df = pd.concat([exp_df, tmp_df], axis=0)

tmp_sim_df = sim_df[sim_df['Spont']!=0.0]
tmp_exp_df = exp_df[exp_df['Spont']!=0.0]

tmp_sim_df = tmp_sim_df[tmp_sim_df['type'].isin(['Scnn1a', 'Rorb', 'Nr5a1', 'PV1', 'PV2'])]


plt.figure(figsize=(9, 9))
plt.scatter(tmp_sim_df['Spont'], tmp_sim_df['Rmax'], c='red', s=5, lw=0)
#plt.scatter(tmp_exp_df['Spont'], tmp_exp_df['Rmax'], c='gray', s=100)
#plt.ylim((3.0, 38.0))
plt.yscale('log')
plt.xscale('log')
plt.xlabel('Spont. rate (Hz)')
plt.ylabel('Rmax (Hz)')
plt.savefig('log_normal_rates/log_normal_rates_spont_Rmax.eps', format='eps')
plt.show()
'''
'''
#sim_hist, sim_bins = np.histogram(tmp_sim_df['Rmax'] / tmp_sim_df['Spont'], bins = np.arange(0, 1000.0, 1.0))
#exp_hist, exp_bins = np.histogram(tmp_exp_df['Rmax'] / tmp_exp_df['Spont'], bins = np.arange(0, 1000.0, 1.0))
sim_hist, sim_bins = np.histogram(tmp_sim_df['Rmax'].values / tmp_sim_df['Spont'].values, bins = np.logspace(-8, 10, 50, base=2.0))
exp_hist, exp_bins = np.histogram(tmp_exp_df['Rmax'].values / tmp_exp_df['Spont'].values, bins = np.logspace(-8, 10, 50, base=2.0))
plt.plot(sim_bins[1:], sim_hist/(1.0*tmp_sim_df.shape[0]), c='r')
plt.plot(exp_bins[1:], exp_hist/(1.0*tmp_exp_df.shape[0]), c='gray')
plt.xscale('log')
plt.show()
'''

