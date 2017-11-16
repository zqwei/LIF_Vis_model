import numpy as np
import matplotlib.pyplot as plt

import pandas as pd

import matplotlib
matplotlib.rcParams.update({'font.size': 20})

from scipy.optimize import leastsq

def lin_mpl(x, A, B):
    return A * x + B

def lin_mpl_fit_2( params, x, y ):
    return (y - lin_mpl( x, params[0], params[1] ))



N_trials = 10

# Decide which systems we are doing analysis for.
# RESTRICT ANALYSIS TO LL2 ONLY, BECAUSE LL1 AND LL3 DO NOT HAVE CONTRASTS OTHER THAN 80%. 
sys_dict = {}
sys_dict['ll2'] = { 'cells_file': '../build/ll2.csv', 'f_1': '../simulations_ll2/gratings/output_ll2_', 'f_2': '_sd278/spk.dat', 'f_3': '_sd278/tot_f_rate.dat', 'f_out': 'lin_figs/tot_f_rate_ll2.csv', 'grating_ids': range(7, 240, 30)+range(8, 240, 30)+range(9, 240, 30), 'model': 'll2', 'contrast': 0.8 }
sys_dict['ll2_ctr30'] = { 'cells_file': '../build/ll2.csv', 'f_1': '../simulations_ll2/gratings/output_ll2_', 'f_2': '_ctr30_sd278/spk.dat', 'f_3': '_ctr30_sd278/tot_f_rate.dat', 'f_out': 'lin_figs/tot_f_rate_ll2_ctr30.csv', 'grating_ids': range(8, 240, 30), 'model': 'll2', 'contrast': 0.3 }
sys_dict['ll2_ctr10'] = { 'cells_file': '../build/ll2.csv', 'f_1': '../simulations_ll2/gratings/output_ll2_', 'f_2': '_ctr10_sd278/spk.dat', 'f_3': '_ctr10_sd278/tot_f_rate.dat', 'f_out': 'lin_figs/tot_f_rate_ll2_ctr10.csv', 'grating_ids': range(8, 240, 30), 'model': 'll2', 'contrast': 0.1  }
sys_dict['ll2_LGN_only_no_con'] = { 'cells_file': '../build/ll2.csv', 'f_1': '../simulations_ll2/gratings/output_ll2_', 'f_2': '_sd278_LGN_only_no_con/spk.dat', 'f_3': '_sd278_LGN_only_no_con/tot_f_rate.dat', 'f_out': 'lin_figs/tot_f_rate_ll2_LGN_only_no_con.csv', 'grating_ids': range(7, 240, 30)+range(8, 240, 30), 'model': 'll2', 'contrast': 0.8 }
sys_dict['ll2_ctr30_LGN_only_no_con'] = { 'cells_file': '../build/ll2.csv', 'f_1': '../simulations_ll2/gratings/output_ll2_', 'f_2': '_ctr30_sd278_LGN_only_no_con/spk.dat', 'f_3': '_ctr30_sd278_LGN_only_no_con/tot_f_rate.dat', 'f_out': 'lin_figs/tot_f_rate_ll2_ctr30_LGN_only_no_con.csv', 'grating_ids': range(8, 240, 30), 'model': 'll2', 'contrast': 0.3 }
sys_dict['ll2_ctr10_LGN_only_no_con'] = { 'cells_file': '../build/ll2.csv', 'f_1': '../simulations_ll2/gratings/output_ll2_', 'f_2': '_ctr10_sd278_LGN_only_no_con/spk.dat', 'f_3': '_ctr10_sd278_LGN_only_no_con/tot_f_rate.dat', 'f_out': 'lin_figs/tot_f_rate_ll2_ctr10_LGN_only_no_con.csv', 'grating_ids': range(8, 240, 30), 'model': 'll2', 'contrast': 0.1 }

'''
# Combine total firing rates, averaged over trials, in summary files.
for sys_name in sys_dict.keys():

    sys_df = pd.read_csv(sys_dict[sys_name]['cells_file'], sep=' ')

    df = pd.DataFrame()
    types = sys_df['type'].values
    gids = sys_df['index'].values

    for grating_id in sys_dict[sys_name]['grating_ids']:
        rates_tmp = np.array([])
        for i_trial in xrange(0, N_trials):
            f_name = '%sg%d_%d%s' % (sys_dict[sys_name]['f_1'], grating_id, i_trial, sys_dict[sys_name]['f_3'])
            print 'Processing file %s.' % (f_name)
            tmp = np.genfromtxt(f_name, delimiter=' ')[:, 1] # Assume all files have the same columns of gids; use the 2nd column for rates.
            if (rates_tmp.size == 0):
                rates_tmp = tmp
            else:
                rates_tmp = rates_tmp + tmp
        rates_tmp = rates_tmp / (1.0 * N_trials)

        tmp_df = pd.DataFrame()
        tmp_df['type'] = types
        tmp_df['gid'] = gids
        tmp_df['contrast'] = sys_dict[sys_name]['contrast'] * np.ones(gids.size)
        tmp_df['model'] = sys_dict[sys_name]['model']
        tmp_df['grating_id'] = grating_id
        cnd_ln = 'full'
        if ('LGN_only_no_con' in sys_name):
            cnd_ln = 'LGN_only'
        tmp_df['condition'] = cnd_ln
        tmp_df['f_rate'] = rates_tmp
        
        df = pd.concat([df, tmp_df], axis=0)

    df.to_csv(sys_dict[sys_name]['f_out'], sep=' ', index=False)
'''




grating_id_color = {8: 'red', 38: 'orange', 68: 'yellow', 98: 'yellowgreen', 128: 'green', 158: 'cyan', 188: 'blue', 218: 'purple', 7: 'red', 37: 'orange', 67: 'yellow', 97: 'yellowgreen', 127: 'green', 157: 'cyan', 187: 'blue', 217: 'purple'}
'''
gid_sel_dict = { 'biophys_exc': {'sel': range(0, 8500, 1), 'color': 'firebrick'}, 'biophys_inh': {'sel': range(8500, 10000, 1), 'color': 'steelblue'} }
contrasts = [0.8, 0.3, 0.1]
grating_ids = range(7, 240, 30) + range(8, 240, 30)
f_data_out = 'lin_figs/tot_f_rate_lin_fit.csv'
fig_out = 'lin_figs/tot_f_rate_lin_fit_example.eps'
fig_out_lim = [-0.001, 9.0]
fig_out_summary = 'lin_figs/tot_f_rate_lin_fit_summary.eps'
fig_out_summary_A_ylim = [-5.0, 15.0]
'''
'''
gid_sel_dict = { 'biophys_exc': {'sel': range(0, 8500, 1), 'color': 'firebrick'}, 'biophys_inh': {'sel': range(8500, 10000, 1), 'color': 'steelblue'} }
contrasts = [0.8, 0.3, 0.1]
grating_ids = [67, 68]
f_data_out = 'lin_figs/tot_f_rate_lin_fit_ori90.csv'
fig_out = 'lin_figs/tot_f_rate_lin_fit_ori90_example.eps'
fig_out_lim = [-0.001, 9.0]
fig_out_summary = 'lin_figs/tot_f_rate_lin_fit_ori90_summary.eps'
fig_out_summary_A_ylim = [-5.0, 15.0]
'''

gid_sel_dict = { 'biophys_exc': {'sel': range(0, 8500, 1), 'color': 'firebrick'}, 'biophys_inh': {'sel': range(8500, 10000, 1), 'color': 'steelblue'} }
contrasts = [0.8]
grating_ids = [7, 37, 67, 97, 127, 157, 187, 217]
f_data_out = 'lin_figs/tot_f_rate_lin_fit_TF2Hz_ctr80.csv'
fig_out = 'lin_figs/tot_f_rate_lin_fit_TF2Hz_ctr80_example.eps'
fig_out_lim = [-0.001, 9.0]
fig_out_summary = 'lin_figs/tot_f_rate_lin_fit_TF2Hz_ctr80_summary.eps'
fig_out_summary_A_ylim = [-2.0, 5.0]


gids = []
for gid_sel_key in gid_sel_dict:
    gid_sel_dict[gid_sel_key]['A_param'] = []
    gid_sel_dict[gid_sel_key]['B_param'] = []
    gid_sel_dict[gid_sel_key]['rsquared'] = []
    [gids.append(x) for x in gid_sel_dict[gid_sel_key]['sel']]
    gids = list(set(gids))
#gids = [4600]





save_data_df = pd.DataFrame(columns=['model', 'gid', 'A', 'B', 'rsquared'])
save_data_df_i_loc = 0
for model_name in ['ll2']:
    df = pd.DataFrame()
    df_LGN_only = pd.DataFrame()

    tmp_keys = [x for x in sys_dict.keys() if model_name in x]
    for sys_key in tmp_keys:
        tmp_df = pd.read_csv(sys_dict[sys_key]['f_out'], sep=' ')
        df = pd.concat([df, tmp_df], axis=0)

    for gid in gids:
        if (gid % 10 == 0):
            print model_name, gid

        tmp_df = df[df['gid'] == gid]

        # The code below is used to make sure that 'full' and 'LGN_only' data points are paired properly for exactly the same stimulus.
        df_gid = pd.DataFrame(columns=['LGN_only', 'full', 'contrast', 'grating_id'])
        i_loc = 0
        for ctr in contrasts:
            tmp_df1 = tmp_df[tmp_df['contrast'] == ctr]
            for grating_id in grating_ids:
                tmp_df2 = tmp_df1[tmp_df1['grating_id'] == grating_id]
                if (tmp_df2.size > 0):
                    x = tmp_df2[tmp_df2['condition'] == 'LGN_only']['f_rate'].values[0]
                    y = tmp_df2[tmp_df2['condition'] == 'full']['f_rate'].values[0]
                    df_gid.loc[i_loc] = [x, y, ctr, grating_id]
                    i_loc += 1

        params_start = (1.0, 0.0)
        fit_params, cov, infodict, mesg, ier = leastsq( lin_mpl_fit_2, params_start, args=(df_gid['LGN_only'], df_gid['full']), full_output=True )
        ss_err = (infodict['fvec']**2).sum()
        ss_tot = ((df_gid['full'] - df_gid['full'].mean())**2).sum()
        if (ss_tot != 0.0):
            rsquared = 1 - (ss_err/ss_tot)
        else:
            rsquared = 0.0

        save_data_df.loc[save_data_df_i_loc] = [model_name, gid, fit_params[0], fit_params[1], rsquared]
        save_data_df_i_loc += 1

        if ((gid in [4600]) and (model_name in ['ll2'])):
            for g_tmp in grating_id_color:
                tmp1 = df_gid[df_gid['grating_id'] == g_tmp]
                plt.scatter(tmp1['LGN_only'], tmp1['full'], s=100, c=grating_id_color[g_tmp], edgecolors='none')
            plt.plot(df_gid['LGN_only'], lin_mpl(df_gid['LGN_only'], fit_params[0], fit_params[1]), c='k')

            plt.title('Model %s, gid = %d; y=Ax+B, A=%f, B=%f, r^2=%f' % (model_name, gid, fit_params[0], fit_params[1], rsquared), fontsize=12)
            plt.gca().set_aspect('equal')
            #max_xy = np.max([df_gid['LGN_only'].max(), df_gid['full'].max()]) * 1.1
            #plt.xlim(left=-0.001)
            #plt.ylim(bottom=0.0)
            #plt.xlim(right=max_xy)
            #plt.ylim(top=max_xy)
            plt.xlim(fig_out_lim)
            plt.ylim(fig_out_lim)
            plt.xlabel('LGN-only firing rate (Hz)')
            plt.ylabel('Full simulation firing rate (Hz)')

            plt.savefig(fig_out, format='eps')

            plt.show()

save_data_df.to_csv(f_data_out, sep=' ', index=False)



# Plot summary figures.
df = pd.read_csv(f_data_out, sep=' ')
for gid_sel_key in gid_sel_dict:
    for gid in gid_sel_dict[gid_sel_key]['sel']:
        tmp_df = df[df['gid'] == gid]
        gid_sel_dict[gid_sel_key]['A_param'].append(1.0/tmp_df['A'].values[0])
        gid_sel_dict[gid_sel_key]['B_param'].append(tmp_df['B'].values[0])
        gid_sel_dict[gid_sel_key]['rsquared'].append(tmp_df['rsquared'].values[0])

fig, axes = plt.subplots(1, 2, figsize=(10, 5))

box0 = axes[0].boxplot([gid_sel_dict[gid_sel_key]['A_param'] for gid_sel_key in gid_sel_dict], patch_artist=True) # notch=True
box1 = axes[1].boxplot([gid_sel_dict[gid_sel_key]['B_param'] for gid_sel_key in gid_sel_dict], patch_artist=True) # notch=True

#box0 = axes[0].boxplot([gid_sel_dict['biophys_exc']['A_param'], gid_sel_dict['biophys_inh']['A_param']], patch_artist=True) # notch=True
#box1 = axes[1].boxplot([gid_sel_dict['biophys_exc']['B_param'], gid_sel_dict['biophys_inh']['B_param']], patch_artist=True) # notch=True

for patch, color in zip(box0['boxes'], [gid_sel_dict[gid_sel_key]['color'] for gid_sel_key in gid_sel_dict]):
    patch.set_facecolor(color)
for patch, color in zip(box1['boxes'], [gid_sel_dict[gid_sel_key]['color'] for gid_sel_key in gid_sel_dict]):
    patch.set_facecolor(color)

axes[0].set_ylabel('1/A')
axes[1].set_ylabel('B (Hz)')
for i, param_key in enumerate(['A_param', 'B_param']):
    tmp_str = ''
    if (param_key == 'A_param'):
        param_key_usage = '1/A_param'
    else:
        param_key_usage = param_key
    for gid_sel_key in gid_sel_dict:
        tmp_str = tmp_str + '%s: %s=%.3f+/-%.3f, n=%d\n' % (gid_sel_key, param_key_usage, np.mean(gid_sel_dict[gid_sel_key][param_key]), np.std(gid_sel_dict[gid_sel_key][param_key]), len(gid_sel_dict[gid_sel_key][param_key]))
    axes[i].annotate(tmp_str, xy=(-0.1, 0.8), xycoords='axes fraction', fontsize=12)

axes[0].set_ylim(fig_out_summary_A_ylim)

# Add the information about goodness of fit.
tmp_str = ''
for gid_sel_key in gid_sel_dict:
    tmp_str = tmp_str + '%s: r^2=%.2f+/-%.2f\n' % ( gid_sel_key, np.mean(gid_sel_dict[gid_sel_key]['rsquared']), np.std(gid_sel_dict[gid_sel_key]['rsquared']) )
axes[0].annotate(tmp_str, xy=(0.3, 0.5), xycoords='axes fraction', fontsize=12)

for ax in axes:
    ax.set_xticklabels(gid_sel_dict.keys())

plt.savefig(fig_out_summary, format='eps')
plt.show()


