import sparseness_functions as sprns

import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams.update({'font.size': 20})


bin_size = 33.3 # Time window, in ms, for binning the spikes.
N_trials = 10

# Decide which simulations we are doing analysis for.
sim_dict = {}
sim_dict['ll2_toe3600'] = { 'cells_file': '../build/ll2.csv', 't_start': 500.0, 't_stop': 5000.0, 'f_1': '../simulations_ll2/natural_movies/output_ll2_TouchOfEvil_frames_3600_to_3750_', 'f_2': '_sd278/spk.dat', 'f_out_r': 'sparseness/ll2_toe3600_r.npy', 'f_out_av': 'sparseness/ll2_toe3600_av.csv' }
sim_dict['ll2_g8'] = { 'cells_file': '../build/ll2.csv', 't_start': 500.0, 't_stop': 3000.0, 'f_1': '../simulations_ll2/gratings/output_ll2_g8_', 'f_2': '_sd278/spk.dat', 'f_out_r': 'sparseness/ll2_g8_r.npy', 'f_out_av': 'sparseness/ll2_g8_av.csv' }

# Process the data and obtain arrays of responses for each neuron within each time bin, averaged over trials.
for sim_key in sim_dict.keys():
    sim_data = sim_dict[sim_key]
    spk_f_names = []
    for i in xrange(N_trials):
        spk_f_names.append('%s%d%s' % (sim_data['f_1'], i, sim_data['f_2']))
    sprns.compute_sprns_array(sim_data['cells_file'], spk_f_names, sim_data['f_out_r'], sim_data['t_start'], sim_data['t_stop'], bin_size)


