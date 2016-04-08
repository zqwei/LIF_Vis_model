import numpy as np
# from scipy.stats import ttest_1samp

num_cell_type = [0, 3700, 7000, 8500, 9300, 10000, 39750, 45000]
cell_type = ['Scnn1a', 'Rorb', 'Nr5a1', 'PV1', 'PV2']
percent_diff_cell = np.zeros([5, 6])
# amp_syn = np.ones([5, 3]) # max value, # min value # update value
default_amp_syn = np.ones([5, 3])
default_amp_syn[:, 0] = 0.01


def compute_estimation_error(n_file_dir, ref_file, amp_syn=default_amp_syn, amp_syn_output='new_syn_weight', ncol=2):
	series = np.genfromtxt(ref_file, delimiter=' ')
	ref_firing_rate = series[0:10000, ncol]
	cell_id = series[0:10000, 0]
	series = np.genfromtxt(n_file_dir+'/tot_f_rate.dat', delimiter=' ')
	firing_rate = series[0:10000, ncol]
	diff_rate = firing_rate-ref_firing_rate
	with open(n_file_dir+'/cell_update_stats.dat', 'w') as output_file:
		output_file.write('cell_type, %neg(<-2), %pos(>2), %other, min_err, max_err, min_amp, max_amp, new_amp\n')
		for cell_group in xrange(len(percent_diff_cell)):
			group = np.logical_and(cell_id >= num_cell_type[cell_group], cell_id < num_cell_type[cell_group+1])
			output_file.write(cell_type[cell_group]+', ')
			per_neg = np.mean(diff_rate[group] < -2)*100.
			per_pos = np.mean(diff_rate[group] > 2)*100.
			per_other = 100. - per_pos-per_neg
			output_file.write('%.2f, %.2f, %.2f, ' % (per_neg, per_pos, per_other))
			output_file.write('%d, %d, ' % (diff_rate[group].min(), diff_rate[group].max()))
			min_amp = amp_syn[cell_group, 0]
			max_amp = amp_syn[cell_group, 1]
			new_amp = amp_syn[cell_group, 2]
			if per_other < 95 and max_amp-min_amp > max_amp*0.01:
				if per_pos == 0. or per_neg/per_pos > 3:
					min_amp = new_amp
					max_amp = max_amp * 2.
					new_amp = (min_amp+max_amp)/2.
				elif per_neg == 0. or per_pos/per_neg > 3:
					max_amp = new_amp
					min_amp = min_amp/2.
					new_amp = (min_amp+max_amp)/2.
			output_file.write('%.2f, %.2f, %.2f\n' % (min_amp, max_amp, new_amp))
			amp_syn[cell_group, 0] = min_amp
			amp_syn[cell_group, 1] = max_amp
			amp_syn[cell_group, 2] = new_amp
	np.save(amp_syn_output, amp_syn)
