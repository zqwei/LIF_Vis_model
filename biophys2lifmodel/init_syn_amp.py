import numpy as np
# from scipy.stats import ttest_1samp

num_cell_type = [0, 3700, 7000, 8500, 9300, 10000, 39750, 45000]
cell_type = ['Scnn1a', 'Rorb', 'Nr5a1', 'PV1', 'PV2']
percent_diff_cell = np.zeros([5, 6])
# amp_syn = np.ones([5, 3]) # max value, # min value # update value
default_amp_syn = np.ones([5, 3])
default_amp_syn[:, 0] = 0.0
default_amp_syn[:,1] = 10.0

np.save("syn_amp_trial_100",default_amp_syn)

amp_syn = np.load("syn_amp_trial_100.npy")

print(amp_syn)

