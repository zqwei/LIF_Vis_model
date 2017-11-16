import numpy as np
import pandas as pd


def compute_sprns_array(cells_file, spk_f_names, f_out_r, t_start, t_stop, bin_size):
    cells_db = pd.read_csv(cells_file, sep=' ')

    t_bins = np.arange(t_start, t_stop, bin_size)
    r_data = np.zeros( (len(cells_db.index), t_bins[:-1].size) )

    t = np.array([])
    gids = np.array([])
    for f_name in spk_f_names:
        print 'Processing file %s.' % (f_name)
        data = np.genfromtxt(f_name, delimiter=' ')
        if (data.size == 0):
            t_tmp = np.array([])
            gids_tmp = np.array([])
        elif (data.size == 2):
            t_tmp = np.array([data[0]])
            gids_tmp = np.array([data[1]])
        else:
            t_tmp = data[:, 0]
            gids_tmp = data[:, 1]

        t = np.concatenate( (t, t_tmp) )
        gids = np.concatenate( (gids, gids_tmp) )

    for k_t, t_bin in enumerate(t_bins[:-1]):
        print 'Computing rates in bins; working on bin %d of %d.' % (k_t, t_bins[:-1].size)
        ind = np.intersect1d( np.where(t >= t_bin), np.where(t < (t_bin + bin_size)) )
        t_tmp = t[ind]
        gids_tmp = gids[ind]
        df = pd.DataFrame( {'gid': gids_tmp, 't': t_tmp} )
        df_tmp = df.groupby('gid').count() * 1000.0 / bin_size # Time is in ms and rate is in Hz.
        df_tmp.columns = ['rates']
        for gid in df_tmp.index:
            r_data[gid, k_t] = df_tmp['rates'].loc[gid] 

    np.save(f_out_r, r_data)

