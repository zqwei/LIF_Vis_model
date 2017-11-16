import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams.update({'font.size': 20})

# NOTE THAT THE FAN-IN INFORMATION IS AVAILABLE DIRECTLY FROM FILES SUCH AS
#../build/ll2_connections_statistics.txt


sys_dict = {}

sys_dict['ll1'] = { 'cells_file': '../build/ll1.csv', 'con_dir': '../build/ll1_connections/', 'f_out': 'recurrent_fan_in_fan_out/ll1.csv' }
sys_dict['ll2'] = { 'cells_file': '../build/ll2.csv', 'con_dir': '../build/ll2_connections/', 'f_out': 'recurrent_fan_in_fan_out/ll2.csv' }
sys_dict['ll3'] = { 'cells_file': '../build/ll3.csv', 'con_dir': '../build/ll3_connections/', 'f_out': 'recurrent_fan_in_fan_out/ll3.csv' }
sys_dict['rl1'] = { 'cells_file': '../build/rl1.csv', 'con_dir': '../build/rl1_connections/', 'f_out': 'recurrent_fan_in_fan_out/rl1.csv' }
sys_dict['rl2'] = { 'cells_file': '../build/rl2.csv', 'con_dir': '../build/rl2_connections/', 'f_out': 'recurrent_fan_in_fan_out/rl2.csv' }
sys_dict['rl3'] = { 'cells_file': '../build/rl3.csv', 'con_dir': '../build/rl3_connections/', 'f_out': 'recurrent_fan_in_fan_out/rl3.csv' }
sys_dict['lr1'] = { 'cells_file': '../build/lr1.csv', 'con_dir': '../build/lr1_connections/', 'f_out': 'recurrent_fan_in_fan_out/lr1.csv' }
sys_dict['lr2'] = { 'cells_file': '../build/lr2.csv', 'con_dir': '../build/lr2_connections/', 'f_out': 'recurrent_fan_in_fan_out/lr2.csv' }
sys_dict['lr3'] = { 'cells_file': '../build/lr3.csv', 'con_dir': '../build/lr3_connections/', 'f_out': 'recurrent_fan_in_fan_out/lr3.csv' }
sys_dict['rr1'] = { 'cells_file': '../build/rr1.csv', 'con_dir': '../build/rr1_connections/', 'f_out': 'recurrent_fan_in_fan_out/rr1.csv' }
sys_dict['rr2'] = { 'cells_file': '../build/rr2.csv', 'con_dir': '../build/rr2_connections/', 'f_out': 'recurrent_fan_in_fan_out/rr2.csv' }
sys_dict['rr3'] = { 'cells_file': '../build/rr3.csv', 'con_dir': '../build/rr3_connections/', 'f_out': 'recurrent_fan_in_fan_out/rr3.csv' }

N_tar_gid_per_file = 100

for sys in sys_dict:

    cells_df = pd.read_csv(sys_dict[sys]['cells_file'], sep=' ')
    gids = cells_df['index'].values

    con_df = '' # Need to create this here so that later we can check whether this is a data frame or not yet.

    for k, gid in enumerate(range(0, gids.size, N_tar_gid_per_file)):
        print 'System %s; processing connections file %d out of %d.' % (sys, k, gids.size/N_tar_gid_per_file)
        tmp_df = pd.read_csv('%s/target_%d_%d.dat' % (sys_dict[sys]['con_dir'], gid, gid + N_tar_gid_per_file), sep=' ', header=None)
        tmp_df.columns = ['tar', 'src', 'N_syn']
        if isinstance(con_df, pd.DataFrame):
            con_df = con_df.append( tmp_df )
        else:
            con_df = tmp_df

    # Characterize the fan-in for the recurrent connections.
    tmp_df = con_df[['tar', 'src']].groupby('tar').count()
    tmp_df.columns = ['N_src']
    tmp_df['gid'] = tmp_df.index
    # Find cell IDs that are not listed in connections files (in case there were no sources to those cells).
    # Add these cell IDs to the dataframe; fill number of sources with zeros.
    tmp_df = tmp_df.append(pd.DataFrame(np.array([gids[~np.in1d(gids, tmp_df['gid'].values)], np.zeros(gids.size - tmp_df['gid'].size)]).T, columns=['gid', 'N_src']))
    # Sort the rows according to the cell IDs.
    tmp_df = tmp_df.sort('gid', ascending=True)
    tmp_df = tmp_df.reset_index(drop=True)

    # Characterize the fan-out for the recurrent connections.
    tmp_df1 = con_df[['tar', 'src']].groupby('src').count()
    tmp_df1.columns = ['N_tar']
    tmp_df1['gid'] = tmp_df1.index
    # Find source IDs that are not listed in the connection files (in case there were no targets for those cells).
    # Add these source IDs to the dataframe; fill number of targets with zeros.
    tmp_df1 = tmp_df1.append(pd.DataFrame(np.array([gids[~np.in1d(gids, tmp_df1['gid'].values)], np.zeros(gids.size - tmp_df1['gid'].size)]).T, columns=['gid', 'N_tar']))
    # Sort the rows according to the source IDs.
    tmp_df1 = tmp_df1.sort('gid', ascending=True)
    tmp_df1 = tmp_df1.reset_index(drop=True)

    combined_df = pd.merge(tmp_df, tmp_df1, on='gid', how='inner')
    combined_df.to_csv(sys_dict[sys]['f_out'], sep=' ', index=False)


for sys in sys_dict:

    combined_df = pd.read_csv(sys_dict[sys]['f_out'], sep=' ')

    sel_gid = {'exc': np.array(range(8500)), 'inh': np.array(range(8500, 10000))}
    for key in sel_gid:
        df1 = combined_df[combined_df['gid'].isin(sel_gid[key])]
        print 'System %s, %s neurons: fan-in is %f +/- %f; fan-out is %f +/- %f.' % (sys, key, df1['N_src'].mean(), df1['N_src'].std(), df1['N_tar'].mean(), df1['N_tar'].std())
 

    #plt.plot(combined_df['gid'], combined_df['N_src'], c='b')
    #plt.plot(combined_df['gid'], combined_df['N_tar'], c='r')
    #plt.show()

