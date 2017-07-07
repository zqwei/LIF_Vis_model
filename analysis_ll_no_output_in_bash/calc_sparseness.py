import numpy as np
import pandas as pd
import cPickle as pickle
import os
import scipy.io as sio
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams.update({'font.size': 20})

def calc_sprns_by_range_and_plot(r_data,inds_range,sp_flag):
    rshape = np.shape(r_data)
    print np.shape(r_data)
    if sp_flag=='pop':
        sp_ind = 0
    elif sp_flag=='lt':
        sp_ind = 1
    else:
        print 'Error: unknown sparseness flag'


    n_frames = rshape[sp_ind]
    rates_array = r_data[inds_range,:]

    r_data_sq = rates_array**2
    nr = (np.sum(rates_array,sp_ind)/n_frames)**2
    dr = (np.sum(r_data_sq,sp_ind)/n_frames)

    S = (1 - nr/dr)/(1-(1/n_frames))

    S=S[~np.isnan(S)]

    if sp_ind == 1:
        plt.figure()
        plt.hist(S)
        plt.show()
    else:
        plt.figure()
        plt.plot(S)
        plt.show()

    return S[~np.isnan(S)]


def evaluate_and_plot_sparseness_by_cell_type(sim_data,r_data,sp_flag):

    ctype_list = ['Scnn1a','Rorb','Nr5a1','PV1','PV2','LIF_exc','LIF_inh','all_bp_exc','all_bp_inh']
    ctr = 0
    fig,ax_list = plt.subplots(3,3)

    for ii in range(3):
        for jj in range(3):
            ax = ax_list[ii,jj]
            if ctr<=len(ctype_list):
                ctype_str = ctype_list[ctr]
                #print sim_data['cells_file']
                S = calc_sprns_by_cell_type(sim_data['cells_file'],r_data,ctype_str,sp_flag)

                # these are matplotlib.patch.Patch properties
                props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

                if sp_flag=='lt':
                    mu=np.mean(S)
                    median = np.median(S)
                    sigma=np.std(S)
                    textstr = '$\mu=%.2f$\n$\mathrm{median}=%.2f$\n$\sigma=%.2f$'%(mu, median, sigma)
                    #ax.hist(S)
                    spr_hist, bins = np.histogram(S, bins=np.linspace(0, 1.0, 10))
                    ax.plot(bins[:-1], spr_hist)
                    ax.set_ylim((0, 8000.0))
                    # place a text box in upper left in axes coords
                    ax.text(0.25, 0.95, textstr, transform=ax.transAxes, fontsize=14,
                            verticalalignment='top', bbox=props)
                    ax.set_title(ctype_str)
                    ctr = ctr+1
                elif sp_flag=='pop':
                    mu=np.mean(S)
                    median = np.median(S)
                    sigma=np.std(S)
                    textstr = '$\mu=%.5f$\n$\mathrm{median}=%.5f$\n$\sigma=%.5f$'%(mu, median, sigma)
                    ax.plot(S)
                    ax.set_ylim([0.7,1])
                    # place a text box in upper left in axes coords
                    ax.text(0.25, 0.95, textstr, transform=ax.transAxes, fontsize=14,
                            verticalalignment='top', bbox=props)
                    ax.set_title(ctype_str)
                    ctr = ctr+1
                else:
                    print 'Error: unknown sparseness flag'

    mng = plt.get_current_fig_manager()
    mng.window.showMaximized()
    plt.savefig(sim_data['f_out_spr_hist_eps'], format='eps')
    plt.show()


def calc_sprns_by_cell_type(cells_file,r_data,ctype_str,sp_flag):
    cells_db = pd.read_csv(cells_file, sep=' ')
    rshape = np.shape(r_data)

    if sp_flag=='pop':
        sp_ind = 0
    elif sp_flag=='lt':
        sp_ind = 1
    else:
        print 'Error: unknown sparseness flag'

    if ctype_str=='all_bp_exc':
        #ct_inds = []
        ct_inds_1 = np.array(np.where(cells_db['type']=='Scnn1a'))
        ct_inds_2 = np.array(np.where(cells_db['type']=='Rorb'))
        ct_inds_3 = np.array(np.where(cells_db['type']=='Nr5a1'))
        ct_inds = np.concatenate((ct_inds_1[0],ct_inds_2[0],ct_inds_3[0]))
    elif ctype_str =='all_bp_inh':
        ct_inds_1 = np.array(np.where(cells_db['type']=='PV1'))
        ct_inds_2 = np.array(np.where(cells_db['type']=='PV2'))
        ct_inds = np.concatenate((ct_inds_1[0],ct_inds_2[0]))
    else:
        ct_inds = np.array(np.where(cells_db['type']==ctype_str))
        ct_inds = ct_inds[0]
        #print ct_inds, ctype_str

    rates_array = r_data[ct_inds]
    n_frames = rshape[sp_ind]

    r_data_sq = rates_array**2
    nr = (np.sum(rates_array,sp_ind)/n_frames)**2
    dr = (np.sum(r_data_sq,sp_ind)/n_frames)

    S = (1 - nr/dr)/(1-(1/n_frames))

    return S[~np.isnan(S)]

def compute_fr_array_mov(cells_file, spk_f_names, f_out_r, t_start, t_stop, bin_size,ntr):
    cells_db = pd.read_csv(cells_file, sep=' ')

    t_bins = np.arange(t_start, t_stop, bin_size)
    r_data = np.zeros( (len(cells_db.index), t_bins[:-1].size) )

    t = np.array([])
    gids = np.array([])
    for f_name in spk_f_names:
        #f_name = spk_f_names
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
        df_tmp = df.groupby('gid').count() * 1000.0 / bin_size/ntr # Time is in ms and rate is in Hz.
        df_tmp.columns = ['rates']
        for gid in df_tmp.index:
            r_data[gid, k_t] = df_tmp['rates'].loc[gid]

    np.save(f_out_r, r_data)

def compute_fr_array_gratings(cells_file, spk_f_names, f_out_r, t_start, t_stop, bin_size,ntr):
    cells_db = pd.read_csv(cells_file, sep=' ')

    t_bins = np.arange(t_start, t_stop, bin_size)
    r_data = np.zeros( (len(cells_db.index), t_bins[:-1].size) )

    t = np.array([])
    gids = np.array([])
    for f_name in spk_f_names:
        #f_name = spk_f_names
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
        df_tmp = df.groupby('gid').count() * 1000.0 / bin_size/ntr # Time is in ms and rate is in Hz.
        df_tmp.columns = ['rates']
        for gid in df_tmp.index:
            r_data[gid, k_t] = df_tmp['rates'].loc[gid]

    np.save(f_out_r, r_data)

def create_nat_movie_sim_dict(base_dir,sys_name):
    st_frame_list = ['1530','3600','5550']
    end_frame_list = ['1680','3750','5700']

    sim_dict_list = {}

    for kk in range(len(st_frame_list)):
        st_frame = st_frame_list[kk]
        end_frame = end_frame_list[kk]
        f1_str = st_frame+'_to_'+end_frame+'_'
        expt_str = sys_name+'_toe'+st_frame

        # Decide which simulations we are doing analysis for.
        sim_dict = {}
        if sys_name=='ll2':
            #f2 = '_sd278/spk.dat'
            f2 = '_sdlif_z101/spk.dat'
        elif sys_name=='rr2':
            f2 = '_sd282_cn0/spk.dat'

        sim_dict[expt_str] = {'cells_file': '/data/mat/antona/network/14-simulations/9-network/build/'+sys_name+'.csv',
                            't_start': 500.0,
                            't_stop': 5000.0,
                            'bin_size':33.3,
                            'N_trials':10,
                            #'f_1': base_dir+'simulations_'+sys_name+'/natural_movies/output_'+sys_name+'_TouchOfEvil_frames_'+f1_str,
                            'f_1': base_dir+'simulation_'+sys_name+'/output_'+sys_name+'_TouchOfEvil_frames_'+f1_str,
                            'f_2': f2,
                            'f_out_r': 'LIF' + expt_str+'_r.npy',
                            'f_out_spr_hist_eps': 'LIF' + expt_str + 'spr_hist.eps'}

        sim_dict_list[kk] = sim_dict

    return sim_dict_list

def create_grating_sim_dict(base_dir,sys_name):
    gc_list = ['8','38','68']

    sim_dict_list = {}

    for kk in range(len(gc_list)):
        f1_str = gc_list[kk]
        expt_str = sys_name+'grating_g'+f1_str

        # Decide which simulations we are doing analysis for.
        sim_dict = {}
        if sys_name=='ll2':
            #f2 = '_sd278/spk.dat'
            f2 = '_sdlif_z101/spk.dat'
        elif sys_name=='rr2':
            f2 = '_sd282_cn0/spk.dat'

        sim_dict[expt_str] = {'cells_file': '/data/mat/antona/network/14-simulations/9-network/build/'+sys_name+'.csv',
                            't_start': 500.0,
                            't_stop': 3000.0,
                            'bin_size':33.3,
                            'N_trials':10,
#                            'f_1': base_dir+'simulations_'+sys_name+'/gratings/output_'+sys_name+'_g'+f1_str+'_',
                            'f_1': base_dir+'simulation_'+sys_name+'/output_'+sys_name+'_g'+f1_str+'_',
                            'f_2': f2,
                            'f_out_r': 'LIF' + expt_str + '_r_v2.npy',
                            'f_out_spr_hist_eps': 'LIF' + expt_str + 'spr_hist.eps'}

        sim_dict_list[kk] = sim_dict

    return sim_dict_list

def sparseness_main(input_dict,sprns_type, plot_only_flag):

    for kk in range(len(input_dict)):
        sim_dict = input_dict[kk]
        for sim_key in sim_dict.keys():
            sim_data = sim_dict[sim_key]

            if plot_only_flag!=1:
                spk_f_names = []
                for i in xrange(sim_data['N_trials']):
                    spk_f_names.append('%s%d%s' % (sim_data['f_1'], i, sim_data['f_2']))

                compute_fr_array_mov(sim_data['cells_file'], spk_f_names, sim_data['f_out_r'], sim_data['t_start'], sim_data['t_stop'], sim_data['bin_size'],sim_data['N_trials'])
                #compute_fr_array_imgs(sim_data['cells_file'], spk_f_names, sim_data['f_out_r'], sim_data['t_start'], sim_data['t_stop'], sim_data['bin_size'],sim_data['N_trials'])
                #compute_fr_array_gratings(sim_data['cells_file'], spk_f_names, sim_data['f_out_r'], sim_data['t_start'], sim_data['t_stop'], sim_data['bin_size'],sim_data['N_trials'])

            print sim_data['f_out_r']
            r_data = np.load(sim_data['f_out_r'])
            evaluate_and_plot_sparseness_by_cell_type(sim_data,r_data,sprns_type)
            #calc_sprns_by_range_and_plot(r_data,np.arange(0,8500,1),'lt')



if __name__ == '__main__':

    #base_dir = '/data/mat/antona/network/14-simulations/9-network/'
    #sys_list = ['ll2','rr2']
    base_dir = '/data/mat/ZiqiangW/simulation_ll_final_syn_data_lif_z102/'
    sys_list = ['ll2']
    plot_only_flag = 1 #0

    for ss in range(len(sys_list)):
        sys_name=sys_list[ss]

        nat_sim_dict = create_nat_movie_sim_dict(base_dir,sys_name)
        sparseness_main(nat_sim_dict,'lt',plot_only_flag)

        #grating_sim_dict = create_grating_sim_dict(base_dir,sys_name)
        #sparseness_main(grating_sim_dict,'lt',plot_only_flag)
