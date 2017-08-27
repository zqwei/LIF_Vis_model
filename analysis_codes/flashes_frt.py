import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import pandas as pd

import f_rate_t_by_type_functions as frtbt

import matplotlib
matplotlib.rcParams.update({'font.size': 20})

bin_start = 0.0
bin_stop = 1500.0
bin_size = 20.0

N_trials = 10

list_of_types = []

result_fname_prefix = 'flashes/ll2_flash_2'
result_fname = result_fname_prefix + '.csv'
result_fig_fname = result_fname_prefix
t_av_start = 500.0
t_av_stop = 1000.0


# Decide which systems we are doing analysis for.
sys_dict = {}
cell_db_path = '/data/mat/antona/network/14-simulations/9-network/build/'
sys_dict['ll2'] = {'cells_file': cell_db_path+'ll2.csv',
                    'f_1': '../simulation_ll2/output_ll2_flash_2_',
                    'f_2': '_sdlif_z101/spk.dat',
                    'f_3': '_sdlif_z101/tot_f_rate.dat',
                    'f_out': 'flashes/ll2_flash_2.pkl',
                    'types': []}


for sys_name in sys_dict.keys():
    # Obtain information about cell types.
    gids_by_type = frtbt.construct_gids_by_type_dict(sys_dict[sys_name]['cells_file'])
    sys_dict[sys_name]['types'] = gids_by_type.keys()
    # Process the spike files and save computed firing rates in files.
    f_list = []
    for i_trial in xrange(0, N_trials):
        f_list.append('%s%d%s' % (sys_dict[sys_name]['f_1'], i_trial, sys_dict[sys_name]['f_2']))
    frtbt.f_rate_t_by_type(gids_by_type, bin_start, bin_stop, bin_size, f_list, sys_dict[sys_name]['f_out'])

# Compute averages and standard deviations by type and save to file.
result_f = open(result_fname, 'w')
result_f.write('system cell_type av_rate std\n')

for sys_name in sys_dict.keys():
    f = open(sys_dict[sys_name]['f_out'], 'r')
    rates_data = pickle.load(f)
    f.close()

    ind = np.intersect1d( np.where( rates_data['t_f_rate'] > t_av_start ), np.where( rates_data['t_f_rate'] < t_av_stop ) )

    for type in sys_dict[sys_name]['types']:
        result_string = '%s %s %f %f' % (sys_name, type, rates_data['mean'][type][ind].mean(), rates_data['mean'][type][ind].std())
        result_f.write(result_string + '\n')
        # print result_string

result_f.close()

# Plot time series of average (over cells) firing rates by type.
rates_dict = {}
for sys_name in sys_dict.keys():
    f = open(sys_dict[sys_name]['f_out'], 'r')
    rates_dict[sys_name] = pickle.load(f)
    f.close()


for type in sys_dict[sys_dict.keys()[0]]['types']: # Use types from the first system; assume that all systems have those types.
    for sys_name in sys_dict.keys():
        plt.plot(rates_dict[sys_name]['t_f_rate'], rates_dict[sys_name]['mean'][type], c='darkorange', label=sys_name)
    plt.xlabel('Time (ms)')
    plt.ylabel('Mean firing rate (Hz)')
    plt.legend()
    plt.title('Type %s' % (type))
    plt.xlim((600.0, 1050.0))
    plt.show()
    plt.savefig(result_fig_fname+'_av_'+type+'.png', format='png')
    plt.close()
