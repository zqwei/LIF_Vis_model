import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams.update({'font.size': 12})


sys_dict = {}
sys_dict['ll2_flash_2'] = { 'cells_file': '../build/ll2.csv', 'f_1': '../simulations_ll2/flashes/output_ll2_flash_2', 'f_2': '_sd278/spk.dat', 'f_3': '_sd278/tot_f_rate.dat', 'f_out': 'flashes/ll2_flash_2.csv' }


t0 = 500.0
t1 = 1200.0
hist_step = 10.0
hist_bins = np.arange(t0, t1, hist_step)

Ntrial = 10

for sys in sys_dict:

    cells_df = pd.read_csv(sys_dict[sys]['cells_file'], sep=' ')
    gids = cells_df['index'].values

    df_tot = pd.DataFrame(columns=['t', 'gid'])
    for i_trial in xrange(0, Ntrial):
        f_name = '%s_%d%s' % (sys_dict[sys]['f_1'], i_trial, sys_dict[sys]['f_2'])
        df = pd.read_csv(f_name, sep=' ', header=None)
        df.columns = ['t', 'gid']
        df = df[df['t'] >= t0]
        df = df[df['t'] <= t1]
        df_tot = df_tot.append(df)

    grouped = df_tot.groupby('gid')
    for group in grouped:
        gid = int(group[0])
        t = group[1]['t'].values
        frate_hist = np.histogram(t, bins=hist_bins)[0] * 1000.0 / (hist_step * Ntrial) # Make sure the resulting units are Hz (time is in ms).
        print 'Processing system %s, gid %d' % (sys, gid)
        plt.plot(hist_bins[:-1], frate_hist); plt.show()

