import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
# import matplotlib
# matplotlib.rcParams.update({'font.size': 20})

bin_start = 500.0
bin_stop = 3000.0
bin_size = 1.0

electrode_pos = [0.0, 0.0, 0.0]
r_cutoff = 10.0 # Distance, in um, below which the weights for 1/r contributions are set to 0.

N_trials = 10



cell_db_path = '/allen/aibs/mat/antona/network/14-simulations/9-network/build/'
file_old_path = '/allen/aibs/mat/antona/network/14-simulations/9-network/'
file_db_path = '/allen/aibs/mat/ZiqiangW/'

# Decide which systems we are doing analysis for.
sys_dict = {}

sys_dict['ll1'] = { 'f_out_prefix': '/allen/aibs/mat/antona/network/14-simulations/9-network/analysis/oscillations/ll1_spectrum', 'grating_ids': [range(6, 240, 30), range(7, 240, 30), range(8, 240, 30), range(9, 240, 30)], 'marker': '--' }
sys_dict['ll2'] = { 'f_out_prefix': '/allen/aibs/mat/antona/network/14-simulations/9-network/analysis/oscillations/ll2_spectrum', 'grating_ids': [range(7, 240, 30), range(8, 240, 30), range(9, 240, 30)], 'marker': '-' }
sys_dict['ll3'] = { 'f_out_prefix': '/allen/aibs/mat/antona/network/14-simulations/9-network/analysis/oscillations/ll3_spectrum', 'grating_ids': [range(8, 240, 30)], 'marker': ':' }

# sys_dict['LIF_ll1'] = { 'f_out_prefix': file_db_path + 'analysis_intFire1/analysis_ll/oscillations/ll1_spectrum', 'grating_ids': [range(7, 240, 30), range(8, 240, 30), range(9, 240, 30)], 'marker': '--' }
# sys_dict['LIF_ll2'] = { 'f_out_prefix': file_db_path + 'analysis_intFire1/analysis_ll/oscillations/ll2_spectrum', 'grating_ids': [range(7, 240, 30), range(8, 240, 30), range(9, 240, 30)], 'marker': '-' }
# sys_dict['LIF_ll3'] = { 'f_out_prefix': file_db_path + 'analysis_intFire1/analysis_ll/oscillations/ll3_spectrum', 'grating_ids': [range(7, 240, 30), range(8, 240, 30), range(9, 240, 30)], 'marker': ':' }

# sys_dict['LIF_IntFire4_ll1'] = { 'f_out_prefix': file_db_path + 'analysis_intFire4/analysis_ll/oscillations/ll1_spectrum', 'grating_ids': [range(7, 240, 30), range(8, 240, 30), range(9, 240, 30)], 'marker': '--' }
# sys_dict['LIF_IntFire4_ll2'] = { 'f_out_prefix': file_db_path + 'analysis_intFire4/analysis_ll/oscillations/ll2_spectrum', 'grating_ids': [range(7, 240, 30), range(8, 240, 30), range(9, 240, 30)], 'marker': '-' }
# sys_dict['LIF_IntFire4_ll3'] = { 'f_out_prefix': file_db_path + 'analysis_intFire4/analysis_ll/oscillations/ll3_spectrum', 'grating_ids': [range(7, 240, 30), range(8, 240, 30), range(9, 240, 30)], 'marker': ':' }

output_fig = 'oscillations/oscillations_Bio_ll.eps'
# output_fig = 'oscillations/oscillations_LIF_InteFire1.eps'
# output_fig = 'oscillations/oscillations_LIF_InteFire4.eps'

# Plot the results.
for sys_name in sys_dict.keys():
    grating_start = 8
    f_name = '%s_%d.pkl' % (sys_dict[sys_name]['f_out_prefix'], grating_start)
    f = open(f_name, 'r')
    freq_fft_abs, av_fft_abs, std_fft_abs = pickle.load(f)
    f.close()
    ind = np.intersect1d( np.where(freq_fft_abs > 0.0), np.where(freq_fft_abs < 100.0) )
    #plt.errorbar(freq_fft_abs[ind], av_fft_abs[ind], yerr=std_fft_abs[ind], marker=sys_dict[sys_name]['marker'], ms=10, markevery=5, color='k', linewidth=2, capsize=0, ecolor='lightgray', elinewidth=5, label=f_name)
    plt.errorbar(freq_fft_abs[ind], 1000.0*av_fft_abs[ind], yerr=1000.0*std_fft_abs[ind], ls=sys_dict[sys_name]['marker'], color='k', linewidth=2, capsize=0, ecolor='lightgray', elinewidth=5, label=f_name)

#plt.yscale('log')
# plt.legend()
plt.xlim(0, 100)
plt.ylim(0, 0.8)
plt.ylabel('Power (arb. u.)')
plt.xlabel('Frequency (Hz)')
plt.savefig(output_fig, format='eps')
plt.show()
