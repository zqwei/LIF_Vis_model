# import numpy as np
from estimation_error import compute_estimation_error
from syn_weight_amp_LGN import syn_weight_amp
from sys import argv
from os import makedirs
from os.path import exists
from shutil import copyfile
import pandas as pd
# from plot_tot_firing_rate_comparison import plot_tot_firing_rate_comparison
num_core = 30
tstop = 500
f_name = 'll2_g8_8_test%dms_LGN_only_no_con_lif_syn_z' % (tstop)
ref_file = 'output_ll2_g8_8_test%dms_LGN_only_no_con_syn_z002/tot_f_rate.dat' % (tstop)


def main(idx_syn):
	idx_pre_syn = idx_syn - 1
	# update of synaptic weight
	output_pre_folder = 'output_' + f_name + str(idx_pre_syn)
	output_pre_m_folder = 'output_' + f_name + str(idx_pre_syn - 1)
	# copy the old synaptic file to current folder
	if exists(output_pre_folder + '/tot_f_rate.dat'):
		if not exists(output_pre_folder + '/cell_update_stats_new.dat'):
			compute_estimation_error(output_pre_folder, output_pre_m_folder, ref_file)
			# plot_tot_firing_rate_comparison(output_pre_folder)
		pdSyn = pd.read_csv(output_pre_folder + '/cell_update_stats_new.dat')
		pdOldWeight = pdSyn["w_old"]
		pdNewWeight = pdSyn["w_curr"]
		E_old = pdSyn["E_old"]
		E_curr = pdSyn["E_curr"]
		# print((E_old < E_curr).sum())
		# print((abs(pdOldWeight-pdNewWeight) > pdOldWeight*1e-5).sum())
		with open('run_compile.bat', 'w') as output_file:
			if (E_old < E_curr).sum() > 0 or (abs(pdOldWeight - pdNewWeight) > pdOldWeight * 1e-5).sum() > 0:
				output_folder = 'output_' + f_name + str(idx_syn)
				if not exists(output_folder):
					makedirs(output_folder)
				copyfile(output_pre_folder + '/cell_update_stats_new.dat', output_folder + '/cell_update_stats_old.dat')
				syn_weight_amp(output_folder, str(idx_syn))
				sfile = f_name + str(idx_syn)
				output_file.write('mpiexec -np %d nrniv -mpi run_%s.py > output_%s/log.txt\n' % (num_core, sfile, sfile))


if __name__ == '__main__':
	main(int(argv[1]))
