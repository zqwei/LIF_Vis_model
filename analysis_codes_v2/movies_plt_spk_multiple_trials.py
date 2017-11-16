import matplotlib.pyplot as plt
import numpy as np


cell_gids = [1000] #range(0, 10)

t_vis_stim = [500.0, 5000.0]
tw_src_id = 552
trial_list = range(0, 10)

for i, trial in enumerate(trial_list): #[8]: #[0, 3, 4, 6, 8]:
    print 'Trial %d' % (trial)
    tw_id = 510 + trial

    N_shift = i * len(cell_gids) + 2
    # Read spikes.
    series = np.genfromtxt('../simulations_ll2/natural_movies/output_ll2_TouchOfEvil_frames_3600_to_3750_%d_sd278/spk.dat' % (trial), delimiter=' ')
    tf_mask = np.in1d(series[:, 1], cell_gids)
    plt.scatter(series[tf_mask, 0], N_shift + series[tf_mask, 1], s=50, lw=0)

plt.show()


