import numpy as np
import h5py
import math
import scipy
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib import gridspec
import matplotlib

def syn_analysis_features_one_cell(spk_times, f_data):
    h5 = h5py.File(f_data, 'r')
    values = h5['values'][...]

    # Downsample the data.
    downsample_bin = 5 # Number of consecutive data points to be lumped together.
    pad_size = math.ceil(float(values.size)/downsample_bin)*downsample_bin - values.size
    values = np.append(values, np.zeros(pad_size)*np.NaN)
    values = scipy.nanmean(values.reshape(-1,downsample_bin), axis=1)
    tarray = np.arange(0, values.size) * 1.0 * h5.attrs['dt'] * downsample_bin
    #plt.plot(tarray, values)
    #plt.show()

    peak_list = []
    t_to_peak_list = []
    t_rise_list = []
    t_decay_list = []
    t_width_list = []

    # Process individual events for each spike.
    spk_t_window = 200.0
    for i_spk, spk_t in enumerate(spk_times):
        ind = np.intersect1d( np.where(tarray >= spk_t), np.where(tarray <= spk_t + spk_t_window) )
        r_tmp = values[ind]
        r_tmp = r_tmp - r_tmp[0]
        t_tmp = tarray[ind] - spk_t
        #plt.plot(t_tmp, r_tmp)
        #plt.show()

        # Convert r_tmp to a mostly positive array, as much as possible.  This should make it easier to extract features,
        # because one doesn't have to worry about choosing max() vs. min(), and so on.
        #r_tmp_sign = np.sign( np.mean(r_tmp - (r_tmp[0] + (r_tmp[-1] - r_tmp[0]) * (t_tmp - t_tmp[0]) / (t_tmp[-1] - t_tmp[0]))) )
        r_tmp_sub_lin_base = r_tmp - (r_tmp[0] + (r_tmp[-1] - r_tmp[0]) * (t_tmp - t_tmp[0]) / (t_tmp[-1] - t_tmp[0]))
        #if (i_spk == 83):
        #    plt.plot(t_tmp, r_tmp_sub_lin_base)
        #    plt.show()
        if (np.abs(r_tmp_sub_lin_base.max()) >= np.abs(r_tmp_sub_lin_base.min())):
            r_tmp_sign = 1.0
        else:
            r_tmp_sign = -1.0
        r_tmp = r_tmp_sign * r_tmp
     
        peak = r_tmp.max()
        ind_peak = r_tmp.argmax()
        t_to_peak = t_tmp[ind_peak]

        #print i_spk
        #if (i_spk == 93):
        #   plt.plot(t_tmp, r_tmp)
        #   print peak, ind_peak, t_to_peak, r_tmp.shape, r_tmp[:ind_peak]
        #   plt.show()

        ind_rise_20pct = np.where(r_tmp[:ind_peak+1] >= 0.2 * peak)[0][0] # Include the data point for the peak, in case the rise is too fast.
        ind_rise_80pct = np.where(r_tmp[:ind_peak+1] >= 0.8 * peak)[0][0] # Include the data point for the peak, in case the rise is too fast.
        t_rise = t_tmp[ind_rise_80pct] - t_tmp[ind_rise_20pct]

        r_tmp_decay = r_tmp[ind_peak:]
        t_tmp_decay = t_tmp[ind_peak:]
        #print i_spk
        #if (i_spk == 83):
        #   plt.plot(t_tmp, r_tmp)
        #   plt.plot(t_tmp_decay, r_tmp_decay)
        #   print peak, ind_peak, t_to_peak, r_tmp.shape, r_tmp[:ind_peak]
        #   plt.show()
        ind_decay_80pct = np.where(r_tmp_decay <= 0.8 * peak)[0][0]
        ind_decay_20pct = np.where(r_tmp_decay <= 0.2 * peak)[0][0]
        r_tmp_decay = r_tmp_decay[ind_decay_80pct:ind_decay_20pct]
        t_tmp_decay = t_tmp_decay[ind_decay_80pct:ind_decay_20pct]

        par = np.polyfit(t_tmp_decay-t_tmp_decay[0], np.log(r_tmp_decay), 1)
        t_decay = np.abs(1.0/par[0]) # Define t_decay using the exponential fit.
    
        t_50pct = t_tmp[r_tmp >= 0.5*peak]
        t_width = t_50pct[-1] - t_50pct[0]
    
        #print i_spk, peak, t_to_peak, t_rise, t_decay, t_width
    
        #plt.plot(t_tmp, r_tmp)
        #plt.scatter([t_to_peak], [peak], s=20)
        #plt.plot([t_tmp[ind_rise_20pct], t_tmp[ind_rise_20pct]+t_rise], [r_tmp[ind_rise_20pct], r_tmp[ind_rise_80pct]], '-o')
        #plt.plot(t_tmp_decay, r_tmp_decay[0]*np.exp(-(t_tmp_decay - t_tmp_decay[0])/t_decay))
        #plt.plot([t_50pct[0], t_50pct[0]+t_width], np.array([1, 1])*0.5*peak, '-o')
        #plt.show()

        peak_list.append(peak)
        t_to_peak_list.append(t_to_peak)
        t_rise_list.append(t_rise)
        t_decay_list.append(t_decay)
        t_width_list.append(t_width)

    features_df = pd.DataFrame()
    features_df['peak'] = np.array(peak_list)
    features_df['t_to_peak'] = np.array(t_to_peak_list)
    features_df['t_rise'] = np.array(t_rise_list)
    features_df['t_decay'] = np.array(t_decay_list)
    features_df['t_width'] = np.array(t_width_list)

    return features_df



types_dict = {}
for gid in xrange(0, 4):
    types_dict[gid] = 'Scnn1a'
for gid in xrange(4, 8):
    types_dict[gid] = 'Rorb'
for gid in xrange(8, 12):
    types_dict[gid] = 'Nr5a1'
for gid in xrange(12, 16):
    types_dict[gid] = 'PV1'
for gid in xrange(16, 20):
    types_dict[gid] = 'PV2'

E_I_type = {}
E_I_type['Scnn1a'] = 'Exc'
E_I_type['Rorb'] = 'Exc'
E_I_type['Nr5a1'] = 'Exc'
E_I_type['PV1'] = 'Inh'
E_I_type['PV2'] = 'Inh'

PS_rec = {}
for gid in xrange(0, 20):
    if (gid % 4 == 0):
        PS_rec[gid] = 'EPSP'
    elif (gid % 4 == 1):
        PS_rec[gid] = 'EPSC'
    elif (gid % 4 == 2):
        PS_rec[gid] = 'IPSP'
    elif (gid % 4 == 3):
        PS_rec[gid] = 'IPSC'

sys_list = ['ll1', 'll2', 'll3', 'lr1', 'lr2', 'lr3', 'rl1', 'rl2', 'rl3','rr1', 'rr2', 'rr3']
sys_list_labels = ['', 'LL', '', '', 'LR', '', '', 'RL', '', '', 'RR', '']

sys_joint_order = ['LL', 'LR', 'RL', 'RR']
sys_joint_dict = {'LL': ['ll1', 'll2', 'll3'], 'LR': ['lr1', 'lr2', 'lr3'], 'RL': ['rl1', 'rl2', 'rl3'], 'RR': ['rr1', 'rr2', 'rr3']}
sys_joint_color = {'LL': 'orangered', 'LR': 'tan', 'RL': 'orange', 'RR': 'darkslategray'}

'''
# Process the individual PSPs and PSCs and build a set of dataframes containing their features.
spk_times = np.loadtxt('syn_analysis/build/syn_spk.dat')

for sys in sys_list:
    dir = 'syn_analysis/output_syn_%s/' % (sys)
    out_base = '%s/features' % (dir)

    data_dict = {}
    data_dict['Exc_EPSP'] = pd.DataFrame()
    data_dict['Exc_EPSC'] = pd.DataFrame()
    data_dict['Exc_IPSP'] = pd.DataFrame()
    data_dict['Exc_IPSC'] = pd.DataFrame()
    data_dict['Inh_EPSP'] = pd.DataFrame()
    data_dict['Inh_EPSC'] = pd.DataFrame()
    data_dict['Inh_IPSP'] = pd.DataFrame()
    data_dict['Inh_IPSC'] = pd.DataFrame()

    for gid in xrange(0, 20):
        if (gid % 2 == 0):
            f_data = '%s/v_out-cell-%d.h5' % (dir, gid)
        elif (gid % 2 == 1):
            f_data = '%s/i_SEClamp-cell-%d.h5' % (dir, gid)

        print 'Processing file %s' % (f_data)
        df = syn_analysis_features_one_cell(spk_times, f_data)

        type_rec_label = '%s_%s' % (E_I_type[types_dict[gid]], PS_rec[gid])
        if (len(data_dict[type_rec_label].index) == 0):
            data_dict[type_rec_label] = df
        else:
            data_dict[type_rec_label] = data_dict[type_rec_label].append(df, ignore_index=True)

    for type_rec_label in data_dict:
        data_dict[type_rec_label].to_csv('%s_%s.csv' % (out_base, type_rec_label), sep=' ', index=False)
'''

# Read the dataframes with features from files.
# Plot statistics of the features.
matplotlib.rcParams.update({'font.size': 12})
for P_C in ['P', 'C']:

    fig, ax = plt.subplots(4, 5, figsize = (16, 8))
    #mng = plt.get_current_fig_manager()
    #mng.window.showMaximized()

    '''
    for i_sys, sys in enumerate(sys_list):
        sys_bar_pos = i_sys + i_sys / 3 # Use integer division here to introduce gaps between ll, lr, rl, and rr systems for plotting.
        dir = 'syn_analysis/output_syn_%s/' % (sys)
        out_base = '%s/features' % (dir)
        print 'Processing data from the directory %s.' % (dir)
        for i_rec, rec in enumerate(['Exc_EPS', 'Exc_IPS', 'Inh_EPS', 'Inh_IPS']):
            type_rec_label = '%s%s' % (rec, P_C)
            if (i_sys == 0):
                ax[i_rec, 0].annotate(type_rec_label, xy=(-0.5, 0.5), xycoords='axes fraction')
            data_dict[type_rec_label] = pd.read_csv('%s_%s.csv' % (out_base, type_rec_label), sep=' ')
            for i_col, col in enumerate(data_dict[type_rec_label].columns):
                #if (col == 'peak'):
                #    tmp = data_dict[type_rec_label][col].values
                #    hist, bins = np.histogram(tmp, bins=20)
                #    ax[i_rec, i_col].plot(bins[:-1], hist/(1.0*tmp.size))
                #else:
                ax[i_rec, i_col].bar([sys_bar_pos], [data_dict[type_rec_label][col].mean()], yerr=[data_dict[type_rec_label][col].values.std()], color='lightblue', ecolor='k', align='center')
                #plt.setp( ax[i_rec, i_col].get_xticklabels(), visible=False)
                ax[i_rec, i_col].set_xticklabels([])
                if ((i_sys == 0) and (i_rec == 0)):
                    if (col == 'peak'):
                        if (P_C == 'P'):
                            plt_units = 'mV'
                        else:
                            plt_units = 'nA'
                    else:
                        plt_units = 'ms'
                    ax[i_rec, i_col].set_title('%s (%s)' % (col, plt_units))

    #for i in xrange(4):
    for j in xrange(5):
        ax[-1, j].set_xticks([(i_sys + i_sys / 3) for i_sys in range(len(sys_list))])
        #ax[i, j].set_xticklabels(sys_list, rotation=50)
        ax[-1, j].set_xticklabels(sys_list_labels)

    plt.savefig('syn_analysis/summary_features_%s.eps' % (P_C), format='eps')
    plt.show()
    '''

    for i_rec, rec in enumerate(['Exc_EPS', 'Exc_IPS', 'Inh_EPS', 'Inh_IPS']):
        type_rec_label = '%s%s' % (rec, P_C)

        sys_joint_data = {}
        for sys_set in sys_joint_order:
            sys_joint_data[sys_set] = pd.DataFrame()
            for sys in sys_joint_dict[sys_set]:
                dir = 'syn_analysis/output_syn_%s/' % (sys)
                out_base = '%s/features' % (dir)
                print 'Processing data from the directory %s.' % (dir)
                tmp_df = pd.read_csv('%s_%s.csv' % (out_base, type_rec_label), sep=' ')
                sys_joint_data[sys_set] = pd.concat([sys_joint_data[sys_set], tmp_df], axis=0)

        for i_col, col in enumerate(tmp_df.columns):
            box_list = []
            for sys_set in sys_joint_order:
                box_list.append(sys_joint_data[sys_set][col].values)
            box = ax[i_rec, i_col].boxplot(box_list, patch_artist=True, sym='c.') # notch=True
            for patch, color in zip(box['boxes'], [sys_joint_color[sys_set] for sys_set in sys_joint_order]):
                patch.set_facecolor(color)

            for i, sys_set in enumerate(sys_joint_order):
                ax[i_rec, i_col].errorbar([i+1], [box_list[i].mean()], yerr=[box_list[i].std() / np.sqrt(1.0 * box_list[i].size)], marker='o', ms=8, color='k', linewidth=2, capsize=5, markeredgewidth=2, ecolor='k', elinewidth=2)

            ax[i_rec, i_col].set_ylim(bottom=0.0)
            ax[i_rec, i_col].set_xticklabels([])
            if (i_rec == 0):
                if (col == 'peak'):
                    if (P_C == 'P'):
                        plt_units = 'mV'
                    else:
                        plt_units = 'nA'
                else:
                    plt_units = 'ms'
                ax[i_rec, i_col].set_title('%s (%s)' % (col, plt_units))

        tmp_str = type_rec_label
        for sys_set in sys_joint_order:
            tmp_str = tmp_str + '\n%s, n=%d' % (sys_set, sys_joint_data[sys_set].shape[0])
        ax[i_rec, 0].annotate(tmp_str, xy=(-0.8, 0.5), xycoords='axes fraction')

    for j in xrange(5):
        ax[-1, j].set_xticks([(i_sys + 1) for i_sys in range(len(sys_joint_order))])
        ax[-1, j].set_xticklabels(sys_joint_order)

    plt.savefig('syn_analysis/summary_features_%s.eps' % (P_C), format='eps')
    plt.show()

# Plot an example distribution of the PSC peaks on a log scale.
matplotlib.rcParams.update({'font.size': 15})
#fig, ax = plt.subplots(4)
fig, ax = plt.subplots(figsize = (16, 7))
type_rec_label = 'Exc_EPSC'
col = 'peak'
plt_units = 'nA' #'pA'

line_color_dict = { 'll': 'red', 'rl': 'darkorange', 'lr': 'olive', 'rr': 'black' }

ax.set_xscale('log', basex=10)

data_dict = {}
for i_sys, sys in enumerate(sys_list): #enumerate(['ll2', 'lr2', 'rl2', 'rr2']):
    dir = 'syn_analysis/output_syn_%s/' % (sys)
    out_base = '%s/features' % (dir)
    data_dict[type_rec_label] = pd.read_csv('%s_%s.csv' % (out_base, type_rec_label), sep=' ')
    tmp = data_dict[type_rec_label]['peak'].values #/ 0.001 # Convert PSC peaks from nA to pA.
    weights = np.ones_like(tmp)/float(tmp.size)
    hist, bins = np.histogram(tmp, weights=weights, bins=np.logspace(-4, -1, 30, base=10))
    #ax[i_sys].set_xscale('log', basex=10)
    #ax[i_sys].plot(bins[:-1], hist, '-o')
    #ax[i_sys].set_ylim(bottom=0.0)
    #if (i_sys != 3):
    #    plt.setp( ax[i_sys].get_xticklabels(), visible=False)
    if (i_sys % 3 == 0):
        label_text = sys[:2].upper()
        ax.plot(bins[:-1], hist, label=label_text, c=line_color_dict[sys[:2]])
    else:
        ax.plot(bins[:-1], hist, c=line_color_dict[sys[:2]])

#ax[-1].set_xlabel('%s %s (%s)' % (type_rec_label, col, plt_units))
#ax[2].set_ylabel('Fraction of synapses')
ax.set_ylim(bottom=0.0)
ax.set_xlabel('%s %s (%s)' % (type_rec_label, col, plt_units))
ax.set_ylabel('Fraction of synapses')
ax.legend(loc='upper left')

plt.savefig('syn_analysis/summary_peaks_comparison.eps', format='eps')
plt.show()


