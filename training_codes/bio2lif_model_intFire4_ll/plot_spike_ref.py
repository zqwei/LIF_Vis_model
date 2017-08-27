import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from glob import glob
from os.path import exists
from sys import argv


num_cell_type = [0, 3700, 7000, 8500, 9300, 10000, 39750, 45000]
color_cell_type = ['k', 'darkred', 'blue', 'green', 'm', 'aqua', 'gray']
color_cell_type = np.array(color_cell_type)

def plot_spikes(base_dir):
	filename = 'spk.dat'
	full_f_name = base_dir + '/' + filename
	print 'Processing file %s.' % (full_f_name)
	if exists(full_f_name):
		series = np.genfromtxt(full_f_name, delimiter=' ')
		group = np.zeros(len(series)).astype('int8')
		for cell_idx in xrange(len(color_cell_type)):
			group[np.logical_and(series[:,1] >= num_cell_type[cell_idx], series[:,1] < num_cell_type[cell_idx + 1])] = cell_idx
		group_color = color_cell_type[group]
		plt.scatter(series[:, 0], series[:, 1], s=1, color=group_color)
		plt.title('%s' % (base_dir))
		plt.ylim(0, 10000)
		# plt.ylim(0, series[:, 1].max())
		plt.xlim(0, series[:, 0].max())
		plt.xlabel('Time (ms)')
		plt.ylabel('Neuron index')
		plt.savefig(base_dir + '_raster.png', dpi=150)
		plt.close('all')

if __name__ == '__main__':
	plot_spikes('output_ll2_g8_8_sd278_test500ms')