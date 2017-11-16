import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams.update({'font.size': 12})

plot_types_color = {'Scnn1a': 'darkorange', 'Rorb': 'red', 'Nr5a1': 'magenta', 'PV1': 'blue', 'PV2': 'cyan'}

cells_file='../build/ll2.csv'
cells_df = pd.read_csv(cells_file, sep=' ')
pop_gids = {}
for type in plot_types_color:
    pop_gids[type] = cells_df[cells_df['type'] == type]['index'].values

fig, ax = plt.subplots(2, 3, figsize=(16, 5))

#f_rate_fname = ['/data/mat/zihao/models_roland/output_ll2_g7_0_sd278_Sup_Scnn1a_100_LIF/tot_f_rate.dat']

f_rate_fname = ['/data/mat/zihao/models_roland/output_ll2_g8_0_sd278_test5/tot_f_rate.dat', '/data/mat/zihao/models_roland/output_ll2_g8_0_sd278_test1/tot_f_rate.dat', '/data/mat/zihao/models_roland/output_ll2_g8_0_sd278_test3/tot_f_rate.dat', '/data/mat/zihao/models_roland/output_ll2_g8_0_sd278_test4/tot_f_rate.dat', '/data/mat/zihao/models_roland/output_ll2_g8_0_sd278_test2/tot_f_rate.dat', '/data/mat/zihao/models_roland/output_ll2_g8_0_sd278_test0/tot_f_rate.dat']
out_fig_name = 'optogenetics/frate_ll2_g8_0_tests_excitation.png'

#f_rate_fname = ['/data/mat/zihao/models_roland/output_ll2_g8_0_sd278_sup5/tot_f_rate.dat', '/data/mat/zihao/models_roland/output_ll2_g8_0_sd278_sup1/tot_f_rate.dat', '/data/mat/zihao/models_roland/output_ll2_g8_0_sd278_sup3/tot_f_rate.dat', '/data/mat/zihao/models_roland/output_ll2_g8_0_sd278_sup4/tot_f_rate.dat', '/data/mat/zihao/models_roland/output_ll2_g8_0_sd278_sup2/tot_f_rate.dat', '/data/mat/zihao/models_roland/output_ll2_g8_0_sd278_sup0/tot_f_rate.dat']
#out_fig_name = 'optogenetics/frate_ll2_g8_0_sup.png'

iy = 0
ix = 0
for i, f in enumerate(f_rate_fname):
    if (i == 3):
        iy = 1
        ix = 0

    df = pd.read_csv(f, sep=' ', header=None)
    df.columns = ['gid', 'frate_w', 'frate_raw']

    for type in pop_gids:
        gids_current = pop_gids[type]
        frate = df[df['gid'].isin(gids_current)]['frate_w'].values
        #print gids_current.size, frate.size
        ax[iy, ix].plot(gids_current, frate, c=plot_types_color[type])
    ax[iy, ix].set_ylim((0.0, 120.0))
    ax[iy, ix].yaxis.set_ticks(np.arange(0, 120.0, 20.0))
    #ax[iy, ix].set_ylim((0.0, 30.0))
    #ax[iy, ix].yaxis.set_ticks(np.arange(0, 30.01, 5.0)) 
    ax[iy, ix].xaxis.set_ticks(np.arange(0, 10000.01, 2500))
    if (iy == 0):
        ax[iy, ix].xaxis.set_ticklabels([])
    else:
        ax[iy, ix].set_xlabel('Neuron ID')
    if (ix != 0):
        ax[iy, ix].yaxis.set_ticklabels([])
    else:
        ax[iy, ix].set_ylabel('Firing rate (Hz)')

    ix += 1

plt.savefig(out_fig_name, format='png')
plt.show()


