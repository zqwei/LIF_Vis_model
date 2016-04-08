import numpy as np
from estimation_error import compute_estimation_error
from syn_weight_amp_LGN import syn_weight_amp

num_core = 30
idx_syn = 102
idx_pre_syn = idx_syn - 1
f_name = 'll2_g8_8_test500ms_LGN_only_no_con_lif_syn_z'

# update of synaptic weight
ref_file = 'output_ll2_g8_8_test500ms_LGN_only_no_con_syn_z002/tot_f_rate.dat'
output_folder = 'output_ll2_g8_8_test500ms_LGN_only_no_con_lif_syn_z' + str(idx_pre_syn)
pre_syn_weight = np.load('syn_amp_trial_%d.npy' % (idx_pre_syn))
new_syn_weight_file = 'syn_amp_trial_%d' % (idx_syn)

with open('run_compile.bat', 'w') as output_file:
	if np.mean(pre_syn_weight[:, 1]-pre_syn_weight[:, 0] < pre_syn_weight[:, 1]*0.01) < 1:
		compute_estimation_error(output_folder, ref_file, amp_syn=pre_syn_weight, amp_syn_output=new_syn_weight_file)
		syn_weight_amp(new_syn_weight_file, str(idx_syn))
		sfile = f_name + str(idx_syn)
		output_file.write('mpiexec -np %d nrniv -mpi run_%s.py > output_%s/log.txt\n' % (num_core, sfile, sfile))
