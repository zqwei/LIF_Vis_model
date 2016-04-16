import numpy as np
from estimation_error import compute_estimation_error


idx_syn = 111
pre_syn_weight = np.load('syn_amp_trial_%d.npy' % (idx_syn))
new_syn_weight_file = 'syn_amp_trial_%d' % (idx_syn+1)
output_folder = 'output_ll2_g8_8_test500ms_LGN_only_no_con_lif_syn_z' + str(idx_syn)
ref_file = 'output_ll2_g8_8_test500ms_LGN_only_no_con_syn_z002/tot_f_rate.dat'
compute_estimation_error(output_folder, ref_file, amp_syn=pre_syn_weight, amp_syn_output=new_syn_weight_file)
