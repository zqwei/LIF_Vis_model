import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import linecache
import math

def w_L_function(tar_angle, src_angle):
    if ((tar_angle != 'None') and (src_angle != 'None')):
        delta_tuning = abs(abs(abs(180.0 - abs(float(tar_angle) - float(src_angle)) % 360.0) - 90.0) - 90.0)
        return math.exp( -(delta_tuning / 50.0)**2 )
    else:
        return 1.0


def compute_i_amplification(sim_config, syn_data_path, tar_gids, N_tar_gid_per_con_file, output_spk_file, type_syn_matrix, syn_mode, t_start, t_stop, dt):

    f_config = open(sim_config, 'r')
    config = json.load(f_config)
    f_config.close()

    cells_db = pd.read_csv([x['spec'] for x in config['manifest'] if x['key']=='CELL_DB'][0], sep=' ')

    out_spk = pd.read_csv(output_spk_file, sep=' ', header=None)
    out_spk.columns = ['t', 'gid']

    f_syn_data = open(syn_data_path, 'r')
    syn_data = json.load(f_syn_data)
    f_syn_data.close()

    con_path = config['connections']

    ext_inputs = {}
    for x in config['ext_inputs']:
        ext_inputs[x] = {'map': pd.read_csv(config['ext_inputs'][x]['map'], sep=' '), 'trial': config['ext_inputs'][x]['trial'], 't_shift': config['ext_inputs'][x]['t_shift'], 'trials_in_file': config['ext_inputs'][x]['trials_in_file']}

    # Process the selected target cells and construct sequences of times of received spikes together with the associated synaptic weights.
    w_df = pd.DataFrame() # Initialize the dataframe to hold the computed information.
    for tar_gid in tar_gids:
        tar_cell_df = cells_db[cells_db['index']==tar_gid]
        tar_type = tar_cell_df['type'].values[0]
        tar_angle = tar_cell_df['tuning'].values[0]
        tar_syn_data = syn_data[tar_type]

        spk_and_w = {}

        # Process the external inputs to this cell.
        for ext_inp_path in ext_inputs:
            N_trials_in_file = ext_inputs[ext_inp_path]['trials_in_file']
            k_trial = ext_inputs[ext_inp_path]['trial']
            tmp_df = ext_inputs[ext_inp_path]['map'][ext_inputs[ext_inp_path]['map']['index']==tar_gid]
            src_gids = tmp_df['src_gid'].values
            presyn_type = tmp_df['presyn_type'].values
            N_syn = tmp_df['N_syn'].values

            spk_times = np.array([])
            w_array = np.array([])
            for i_src, src_type in enumerate(presyn_type):
                w = tar_syn_data[src_type]['w'] * N_syn[i_src]
                tmp_spk = [float(y) for y in linecache.getline(ext_inp_path, src_gids[i_src] * N_trials_in_file + k_trial + 1).split()] # Line numbers in file start from 1.
                spk_times = np.concatenate((spk_times, np.array(tmp_spk)))
                w_array = np.concatenate((w_array, np.ones(len(tmp_spk))*w))

            spk_and_w[ext_inp_path] = pd.DataFrame({'t': spk_times, 'w': w_array})

        # Process the contributions from the recurrent connections.
        f_key_for_tar_gid = N_tar_gid_per_con_file * (tar_gid / N_tar_gid_per_con_file) # Note that integer division is used here.
        f_con_name = '%s/target_%d_%d.dat' % (con_path, f_key_for_tar_gid, f_key_for_tar_gid + N_tar_gid_per_con_file)
        con_df = pd.read_csv(f_con_name, sep=' ', header=None)
        con_df.columns = ['tar', 'src', 'N_syn']
        tmp_df = con_df[con_df['tar']==tar_gid]
        src_gids = tmp_df['src'].values
        N_syn = tmp_df['N_syn'].values
        tmp_out_spk = out_spk[out_spk['gid'].isin(src_gids)]

        spk_times = np.array([])
        w_array = np.array([])
        for i_src, src_gid in enumerate(src_gids):
            src_tmp_df = cells_db[cells_db['index']==src_gid]
            src_type = src_tmp_df['type'].values[0]
            presyn_type = type_syn_matrix[src_type]
            src_angle = src_tmp_df['tuning'].values[0]

            if (presyn_type != 'inh'): # Process contributions from excitatory connections only.
                if (syn_mode != 'L'):
                    w_multiplier = 1.0
                else:
                    w_multiplier = w_L_function(tar_angle, src_angle)
                w = tar_syn_data[presyn_type]['w'] * N_syn[i_src] * w_multiplier
                tmp_spk = tmp_out_spk[tmp_out_spk['gid']==src_gid]['t'].values
                spk_times = np.concatenate((spk_times, tmp_spk))
                w_array = np.concatenate((w_array, np.ones(len(tmp_spk))*w))

        spk_and_w['recurrent'] = pd.DataFrame({'t': spk_times, 'w': w_array})

        # Convert time and weight arrays to histograms.
        #spk_hist = {}
        #for x in spk_and_w:
        #    t = spk_and_w[x]['t'].values
        #    ind = np.intersect1d( np.where(t >= t_start), np.where(t < t_stop) )
        #    spk_hist[x], bins = np.histogram(t[ind], weights=spk_and_w[x]['w'].values[ind], bins=np.arange(0.0, config['run']['tstop'], dt))
        #    print tar_gid, x[-15:], spk_hist[x].mean()
        #    plt.plot(bins[:-1], spk_hist[x], label=x)
        #plt.legend()
        #plt.show()
        #print ''

        # Compute average input currents for the cell (in units of weight per time).
        t_tot = t_stop - t_start
        w_t_mean = {}
        #tmp_tot = 0.0
        for x in spk_and_w:
            # Convert labels to be more descriptive.
            label = x
            if ('tw' in x[-20:]):
                label = 'tw'
            elif ('LGN' in x[-20:]):
                label = 'LGN'
            t = spk_and_w[x]['t'].values
            ind = np.intersect1d( np.where(t >= t_start), np.where(t < t_stop) )
            w_t_mean[label] = spk_and_w[x]['w'].values[ind].sum() / t_tot
            #tmp_tot += w_t_mean[label]
        #w_t_mean['total'] = tmp_tot
        #if (tmp_tot != 0.0):
        #    w_t_mean['LGN_frac'] = w_t_mean['LGN'] / tmp_tot
        #else:
        #    w_t_mean['LGN_frac'] = 0.0
        #print w_t_mean

        # Combine results from individual cells into a dataframe.
        if w_df.empty:
            w_df = pd.DataFrame([w_t_mean])
        else:
            w_df = w_df.append(pd.DataFrame([w_t_mean]))
    w_df['gid'] = tar_gids
    w_df = w_df.reset_index(drop=True)

    return w_df


#sim_config = '/data/mat/ZiqiangW/simulation_ll_final_syn_data_lif_z102/simulation_ll2/config_ll2_g8_0_sdlif_z101.json'
#syn_data_path = '/data/mat/ZiqiangW/simulation_ll_final_syn_data_lif_z102/syn_data_lif_z101.json'
#output_spk_file = '/data/mat/ZiqiangW/simulation_ll_final_syn_data_lif_z102/simulation_ll2/output_ll2_g8_0_sdlif_z101/spk.dat'
#tar_gids = [2, 1002, 2002, 3002, 4002, 5002, 6002, 7002, 8002, 9002]
#N_tar_gid_per_con_file = 100
#type_syn_matrix = { 'Scnn1a': 'exc', 'Rorb': 'exc', 'Nr5a1': 'exc', 'LIF_exc': 'exc', 'PV1': 'inh', 'PV2': 'inh', 'LIF_inh': 'inh'}
#syn_mode = 'L'
#t_start = 500.0
#t_stop = 3000.0
#dt = 10.0 # In ms.
#compute_i_amplification(sim_config, syn_data_path, tar_gids, N_tar_gid_per_con_file, output_spk_file, type_syn_matrix, syn_mode, t_start, t_stop, dt)


def run_i_amp_LIF_grating_ori(gids, files_dict, sim_process_param):
    # Process gratings metadata and find gratings with the specified TF and SF.
    g_metadata = pd.read_csv(files_dict['gratings_metadata'], sep=' ', header=None)
    g_metadata.columns = ['grating_id', 'ori', 'SF', 'TF', 'contrast']
    g_metadata['grating_id'] = g_metadata['grating_id'].str.split('.').str[-2].str.split('_').str[-1]

    g_metadata = g_metadata[g_metadata['TF']==files_dict['TF']]
    g_metadata = g_metadata[g_metadata['SF']==files_dict['SF']]

    # Create a dataframe where each cell is mapped to a specific grating, based on the preferred direction
    # that was computed before (even though the set of gratings that we are using here may not contain the one
    # that evokes the strongest response, given that the SF and TF may not be the most preferred ones for every cell).
    g_pref_data = pd.read_csv(files_dict['f_gratings_pref'], sep=' ')
    g_pref_data = g_pref_data[g_pref_data['id'].isin(gids)]
    #print g_pref_data[['id', 'ori']]
    pref_g_list = []
    for gid in gids:
        pref_ori = g_pref_data[g_pref_data['id']==gid]['ori'].values[0]
        pref_g_list.append(g_metadata[g_metadata['ori']==pref_ori]['grating_id'].values[0])

    gid_pref_g = pd.DataFrame( {'gid': gids, 'pref_grating_id': pref_g_list} )
    #plt.scatter(g_pref_data['ori'], gid_pref_g['pref_grating_id'])
    #plt.show()

    # Compute currents and amplification for groups of cells that have the same preferred grating.
    pref_g_unique = gid_pref_g['pref_grating_id'].unique()
    w_combined_mean = pd.DataFrame()
    for trial_id in xrange(files_dict['N_trial']):
        print 'System %s, processing trial %d of %d.' % (files_dict['sys_name'], trial_id, files_dict['N_trial'])

        w_combined = pd.DataFrame()

        for grating_id in pref_g_unique:
            tar_gids = gid_pref_g[gid_pref_g['pref_grating_id']==grating_id]['gid'].values
            sim_config = files_dict['dir'] + '/config_%s_g%s_%d_%s.json' % (files_dict['sys_name'], grating_id, trial_id, files_dict['f_label'])
            output_spk_file = files_dict['dir'] + '/output_%s_g%s_%d_%s/spk.dat' % (files_dict['sys_name'], grating_id, trial_id, files_dict['f_label'])
            w_tmp = compute_i_amplification(sim_config, files_dict['syn_data'], tar_gids, sim_process_param['N_tar_gid_per_con_file'], output_spk_file, sim_process_param['type_syn_matrix'], files_dict['syn_mode'], sim_process_param['t_start'], sim_process_param['t_stop'], sim_process_param['dt'])
            w_combined = w_combined.append(w_tmp)

        w_combined.sort('gid', inplace=True)
        #print w_combined

        if w_combined_mean.empty:
            w_combined_mean = w_combined
        else:
            w_combined_mean = w_combined_mean.add(w_combined, fill_value=0)

    w_combined_mean = w_combined_mean / files_dict['N_trial']
    w_combined_mean['total'] = w_combined_mean['LGN'] + w_combined_mean['tw'] + w_combined_mean['recurrent']
    w_combined_mean['LGN_frac'] = w_combined_mean['LGN'] / w_combined_mean['total']
    w_combined_mean = w_combined_mean.reset_index(drop=True)
    #print w_combined_mean

    w_combined_mean.to_csv(files_dict['f_out'], sep=' ', index=False)





gids = range(2, 10000, 200)
sim_process_param = {   'N_tar_gid_per_con_file': 100,
                        'type_syn_matrix': { 'Scnn1a': 'exc', 'Rorb': 'exc', 'Nr5a1': 'exc', 'LIF_exc': 'exc', 'PV1': 'inh', 'PV2': 'inh', 'LIF_inh': 'inh'},
                        't_start': 500.0,
                        't_stop': 3000.0,
                        'dt': 10.0
                    }
sys_dict = {}
sys_dict['ll1_2Hz'] = {    'sys_name': 'll1',
                  'dir': '/data/mat/ZiqiangW/simulation_ll_final_syn_data_lif_z102/simulation_ll1/',
                  'f_label': 'sdlif_z101',
                  'SF': 0.05,
                  'TF': 2.0,
                  'gratings_metadata': '/data/mat/antona/network/14-simulations/6-LGN_firing_rates_and_spikes/LGN_spike_trains/movies_gratings/res_192_metadata.txt',
                  'N_trial': 10,
                  'syn_data': '/data/mat/ZiqiangW/simulation_ll_final_syn_data_lif_z102/syn_data_lif_z101.json',
                  'f_gratings_pref': 'Ori_LIF/ll1_pref_stat.csv',
                  'syn_mode': 'L',
                  'f_out': 'i_amplification_LIF/i_amplification_ll1_2Hz.csv'
             }
sys_dict['ll2_2Hz'] = {    'sys_name': 'll2',
                  'dir': '/data/mat/ZiqiangW/simulation_ll_final_syn_data_lif_z102/simulation_ll2/',
                  'f_label': 'sdlif_z101',
                  'SF': 0.05,
                  'TF': 2.0,
                  'gratings_metadata': '/data/mat/antona/network/14-simulations/6-LGN_firing_rates_and_spikes/LGN_spike_trains/movies_gratings/res_192_metadata.txt',
                  'N_trial': 10,
                  'syn_data': '/data/mat/ZiqiangW/simulation_ll_final_syn_data_lif_z102/syn_data_lif_z101.json',
                  'f_gratings_pref': 'Ori_LIF/ll2_pref_stat.csv',
                  'syn_mode': 'L',
                  'f_out': 'i_amplification_LIF/i_amplification_ll2_2Hz.csv'
             }
sys_dict['ll3_2Hz'] = {    'sys_name': 'll3',
                  'dir': '/data/mat/ZiqiangW/simulation_ll_final_syn_data_lif_z102/simulation_ll3/',
                  'f_label': 'sdlif_z101',
                  'SF': 0.05,
                  'TF': 2.0,
                  'gratings_metadata': '/data/mat/antona/network/14-simulations/6-LGN_firing_rates_and_spikes/LGN_spike_trains/movies_gratings/res_192_metadata.txt',
                  'N_trial': 10,
                  'syn_data': '/data/mat/ZiqiangW/simulation_ll_final_syn_data_lif_z102/syn_data_lif_z101.json',
                  'f_gratings_pref': 'Ori_LIF/ll3_pref_stat.csv',
                  'syn_mode': 'L',
                  'f_out': 'i_amplification_LIF/i_amplification_ll3_2Hz.csv'
             }
sys_dict['ll1_4Hz'] = {    'sys_name': 'll1',
                  'dir': '/data/mat/ZiqiangW/simulation_ll_final_syn_data_lif_z102/simulation_ll1/',
                  'f_label': 'sdlif_z101',
                  'SF': 0.05,
                  'TF': 4.0,
                  'gratings_metadata': '/data/mat/antona/network/14-simulations/6-LGN_firing_rates_and_spikes/LGN_spike_trains/movies_gratings/res_192_metadata.txt',
                  'N_trial': 10,
                  'syn_data': '/data/mat/ZiqiangW/simulation_ll_final_syn_data_lif_z102/syn_data_lif_z101.json',
                  'f_gratings_pref': 'Ori_LIF/ll1_pref_stat.csv',
                  'syn_mode': 'L',
                  'f_out': 'i_amplification_LIF/i_amplification_ll1_4Hz.csv'
             }
sys_dict['ll2_4Hz'] = {    'sys_name': 'll2',
                  'dir': '/data/mat/ZiqiangW/simulation_ll_final_syn_data_lif_z102/simulation_ll2/',
                  'f_label': 'sdlif_z101',
                  'SF': 0.05,
                  'TF': 4.0,
                  'gratings_metadata': '/data/mat/antona/network/14-simulations/6-LGN_firing_rates_and_spikes/LGN_spike_trains/movies_gratings/res_192_metadata.txt',
                  'N_trial': 10,
                  'syn_data': '/data/mat/ZiqiangW/simulation_ll_final_syn_data_lif_z102/syn_data_lif_z101.json',
                  'f_gratings_pref': 'Ori_LIF/ll2_pref_stat.csv',
                  'syn_mode': 'L',
                  'f_out': 'i_amplification_LIF/i_amplification_ll2_4Hz.csv'
             }
sys_dict['ll3_4Hz'] = {    'sys_name': 'll3',
                  'dir': '/data/mat/ZiqiangW/simulation_ll_final_syn_data_lif_z102/simulation_ll3/',
                  'f_label': 'sdlif_z101',
                  'SF': 0.05,
                  'TF': 4.0,
                  'gratings_metadata': '/data/mat/antona/network/14-simulations/6-LGN_firing_rates_and_spikes/LGN_spike_trains/movies_gratings/res_192_metadata.txt',
                  'N_trial': 10,
                  'syn_data': '/data/mat/ZiqiangW/simulation_ll_final_syn_data_lif_z102/syn_data_lif_z101.json',
                  'f_gratings_pref': 'Ori_LIF/ll3_pref_stat.csv',
                  'syn_mode': 'L',
                  'f_out': 'i_amplification_LIF/i_amplification_ll3_4Hz.csv'
             }

#for sys in sys_dict:
#    run_i_amp_LIF_grating_ori(gids, sys_dict[sys], sim_process_param)





fig, axes = plt.subplots(2, 1, figsize = (5, 10))
for k_TF, TF_label in enumerate(['2Hz', '4Hz']):
    combined_df = pd.DataFrame()
    for sys_name in ['ll1', 'll2', 'll3']:
        sys = '%s_%s' % (sys_name, TF_label)
        tmp_df = pd.read_csv(sys_dict[sys]['f_out'], sep=' ')
        combined_df = combined_df.append(tmp_df)

    for type in ['exc', 'inh']:
        if (type == 'exc'):
            sel_gid1 = 0
            sel_gid2 = 8500
            ax = axes[0]
        elif (type == 'inh'):
            sel_gid1 = 8500
            sel_gid2 = 10000
            ax = axes[1]
        tmp_df = combined_df[combined_df['gid'].isin(range(sel_gid1, sel_gid2))]

        hist, bins = np.histogram(tmp_df['LGN_frac'], bins=np.arange(0, 1.0, 0.025))
        hist = hist / (1.0 * tmp_df['LGN_frac'].size)

        LGN_frac_mean = tmp_df['LGN_frac'].mean()
        LGN_frac_std = tmp_df['LGN_frac'].std()

        ax.step(bins[:-1], hist, label='%s Hz' % (TF_label[:-2]))
        ax.set_title('%s.' % (type))
        ax.annotate('%s Hz: %f +/- %f' % (TF_label[:-2], LGN_frac_mean, LGN_frac_std), xy=(0.05, 0.9 - 0.1 * k_TF), xycoords='axes fraction', horizontalalignment='left', verticalalignment='top')

for ax in axes:
    ax.legend()
plt.savefig('i_amplification_LIF/i_amplification_LIF.eps', format='eps')
plt.show()


