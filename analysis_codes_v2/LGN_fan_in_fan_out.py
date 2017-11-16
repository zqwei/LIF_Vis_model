import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams.update({'font.size': 20})


sys_dict = {}
sys_dict['ll1'] = { 'cells_file': '../build/ll1.csv', 'src_file': '/data/mat/antona/network/14-simulations/6-LGN_firing_rates_and_spikes/LGN_spike_trains/LGN_visual_space_positions_and_cell_types.dat', 'con_file': '../build/ll1_inputs_from_LGN.csv', 'f_N_src': 'LGN_fan_in_fan_out/ll1_fan_in_from_LGN.csv', 'f_N_tar': 'LGN_fan_in_fan_out/ll1_fan_out_from_LGN.csv' }
sys_dict['ll2'] = { 'cells_file': '../build/ll2.csv', 'src_file': '/data/mat/antona/network/14-simulations/6-LGN_firing_rates_and_spikes/LGN_spike_trains/LGN2_visual_space_positions_and_cell_types.dat', 'con_file': '../build/ll2_inputs_from_LGN.csv', 'f_N_src': 'LGN_fan_in_fan_out/ll2_fan_in_from_LGN.csv', 'f_N_tar': 'LGN_fan_in_fan_out/ll2_fan_out_from_LGN.csv' }
sys_dict['ll3'] = { 'cells_file': '../build/ll3.csv', 'src_file': '/data/mat/antona/network/14-simulations/6-LGN_firing_rates_and_spikes/LGN_spike_trains/LGN3_visual_space_positions_and_cell_types.dat', 'con_file': '../build/ll3_inputs_from_LGN.csv', 'f_N_src': 'LGN_fan_in_fan_out/ll3_fan_in_from_LGN.csv', 'f_N_tar': 'LGN_fan_in_fan_out/ll3_fan_out_from_LGN.csv' }

for sys in sys_dict:

    cells_df = pd.read_csv(sys_dict[sys]['cells_file'], sep=' ')
    gids = cells_df['index'].values

    src_df = pd.read_csv(sys_dict[sys]['src_file'], sep=' ')
    src_df.columns = ['LGN_type', 'x', 'y', 'x_offset', 'y_offset', 'sigma_c', 'sigma_s', 'r0', 'scaling_factor', 'k_alpha']
    src_df['LGN_gid'] = src_df.index
    src_gids = src_df['LGN_gid']

    con_df = pd.read_csv(sys_dict[sys]['con_file'], sep=' ')

    # Characterize the fan-in from the input sources.
    tmp_df = con_df[['index', 'src_gid']].groupby('index').count()
    tmp_df.columns = ['N_src']
    tmp_df['tar_gid'] = tmp_df.index
    tmp_df = tmp_df.reset_index(drop=True)
    # Find cell IDs from the cell file that are not listed in the inputs file (in case there were no sources to those cells).
    # Add these cell IDs to the dataframe; fill number of sources with zeros.
    tmp_df = tmp_df.append(pd.DataFrame(np.array([gids[~np.in1d(gids, tmp_df['tar_gid'].values)], np.zeros(gids.size - tmp_df['tar_gid'].size)]).T, columns=['tar_gid', 'N_src']))
    # Sort the rows according to the cell IDs.
    tmp_df = tmp_df.sort('tar_gid', ascending=True)
    #plt.plot(tmp_df['tar_gid'], tmp_df['N_src'])
    #plt.show()
    tmp_df.to_csv(sys_dict[sys]['f_N_src'], sep=' ', index=False)

    sel_gid = {'exc': np.array(range(8500)), 'inh': np.array(range(8500, 10000))}
    for key in sel_gid:
        df1 = tmp_df[tmp_df['tar_gid'].isin(sel_gid[key])]
        print 'System %s; fan-in from LGN to %s: %f +/- %f' % (sys, key, df1['N_src'].mean(), df1['N_src'].std())

    # Characterize the fan-out from the input sources.
    tmp_df = con_df[['index', 'src_gid']].groupby('src_gid').count()
    #print tmp_df
    tmp_df.columns = ['N_tar']
    tmp_df['src_gid'] = tmp_df.index
    #tmp_df = tmp_df.reset_index(drop=True)
    # Find source IDs that are not listed in the inputs file (in case there were no targets for those sources).
    # Add these source IDs to the dataframe; fill number of targets with zeros.
    tmp_df = tmp_df.append(pd.DataFrame(np.array([src_gids[~np.in1d(src_gids, tmp_df['src_gid'].values)], np.zeros(src_gids.size - tmp_df['src_gid'].size)]).T, columns=['src_gid', 'N_tar']))
    # Sort the rows according to the source IDs.
    tmp_df = tmp_df.sort('src_gid', ascending=True)
    tmp_df = tmp_df.reset_index(drop=True)
    #print tmp_df
    #plt.plot(tmp_df['src_gid'], tmp_df['N_tar'])
    #plt.show()

    #df1 = tmp_df[tmp_df['N_tar'] <= 0]
    #src_df1 = src_df[src_df['LGN_gid'].isin(df1['src_gid'])]
    #plt.scatter(src_df1['x'], src_df1['y'], c='w')
    #df1 = tmp_df[tmp_df['N_tar'] > 0]
    #src_df1 = src_df[src_df['LGN_gid'].isin(df1['src_gid'])]
    #plt.scatter(src_df1['x'], src_df1['y'], c='r')
    #plt.show()

    tmp_df.to_csv(sys_dict[sys]['f_N_tar'], sep=' ', index=False)
    print 'System %s; fan-out from LGN: %f +/- %f.' % (sys, tmp_df['N_tar'].mean(), tmp_df['N_tar'].std())


