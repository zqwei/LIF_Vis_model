import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams.update({'font.size': 20})

# Plot spikes as position of a neuron (either x, or y, or z) vs. spike time and draw a linear
# function fitting these data.
def plt_spk_pos_vs_t_linear_fit(f_name, pos_axis, out_fig_name):
    df = pd.read_csv(f_name, sep=' ', header=None)
    df.columns = ['t', 'x', 'y', 'z']

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(df['t'], df[pos_axis], s=5, lw=0, facecolor='lightgray')

    # Linear fit.
    fit = np.polyfit(df['t'], df[pos_axis], 1)
    fit_fn = np.poly1d(fit)  # Defined this way, fit_fn is a function which uses x as an argument and returns an estimate for y based on the 'fit' result.

    ax.plot(df['t'], fit_fn(df['t']), c='black')

    ax.set_xlim([500.0, 2700.0])
    ax.set_ylim([-450.0, 450.0])
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel(r'%s ($\mu$m)' % (pos_axis))
    ax.set_title(r'Linear fit coefficient: %f $\mu$m/ms' % (fit[0]))

    plt.savefig(out_fig_name, format='eps')
    plt.show()





plt_spk_pos_vs_t_linear_fit('/data/mat/antona/network/12-new-populations/2-simulations-and-connectivity-boundary-rules/10000_cells_xy_square_pbc/output_4_Wbar_w2_v13pixps_hor/spk_xyz_pyramids.dat', 'x', 'bars_pbc_illustration/10000_cells_xy_square_pbc_4_Wbar_w2_v13pixps_hor.eps')

plt_spk_pos_vs_t_linear_fit('/data/mat/antona/network/12-new-populations/2-simulations-and-connectivity-boundary-rules/10000_cells_cylinder/hybrid/output_27_Wbar_w2_v13pixps_hor/spk_xyz_pyramids.dat', 'x', 'bars_pbc_illustration/10000_cells_cylinder_hybrid_27_Wbar_w2_v13pixps_hor.eps')

