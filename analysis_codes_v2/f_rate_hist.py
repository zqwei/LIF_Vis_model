import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib
matplotlib.rcParams.update({'font.size': 20})


#hist_bins = np.linspace(0.0, 100.0, 101)
hist_bins = np.linspace(0.0, 100.0, 27)
gids_exc = np.array(range(0, 8500))
gids_inh = np.array(range(8500, 10000))

#for grating_id in [8]: #xrange(8, 240, 30):
#    for trial_id in [8]: #xrange(10):
for stim in ['spont']:
    for trial_id in [5]: #xrange(20):
        # Load simulation data and create histograms.
        #f_name = '../simulations_ll2/gratings/output_ll2_g%d_%d_sd278/tot_f_rate.dat' % (grating_id, trial_id)
        f_name = '../simulations_ll2/spont/output_ll2_%s_%d_sd278/tot_f_rate.dat' % (stim, trial_id)
        print 'Processing file %s.' % (f_name)
        rates = np.genfromtxt(f_name, delimiter=' ')[:, 1] # Assume all files have the same columns of gids; use the 2nd column for rates.

        hist_exc = np.histogram(rates[gids_exc], bins=hist_bins)[0] / (1.0 * gids_exc.size)
        hist_inh = np.histogram(rates[gids_inh], bins=hist_bins)[0] / (1.0 * gids_inh.size)

        fig, axes = plt.subplots(1, 2)
        axes[0].plot(hist_bins[:-1], hist_exc)
        axes[1].plot(hist_bins[:-1], hist_inh)
        axes[0].set_title('Excitatory neurons')
        axes[1].set_title('Inhibitory neurons')
        #axes[0].set_xlim((0, 20.0))
        axes[0].set_xlim((0, 50.0))
        axes[1].set_xlim((0, 50.0))
        axes[0].set_ylabel('Fraction of cells')
        axes[0].set_xlabel('Firing rate (Hz)')
        axes[1].set_xlabel('Firing rate (Hz)')
        plt.show()


