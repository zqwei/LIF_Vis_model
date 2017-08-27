import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import plot_functions


tw_map = '/allen/aibs/mat/antona/network/14-simulations/9-network/'
cell_db_path = '/allen/aibs/mat/antona/network/14-simulations/9-network/build/'

t_vis_stim = [500.0, 3000.0]
tw_src_id = 552
for trial in [1]:
    # print 'Trial %d' % (trial)
    tw_id = 180 + trial
    fig, axes = plot_functions.plot_spikes_tw('../simulation_ll2/output_ll2_g8_%d_sdlif_z101/spk.dat' % (trial),
                                              tw_map+'/tw_data/ll2_tw_build/2_tw_src/f_rates_%d.pkl' % (tw_id),
                                              tw_src_id,
                                              cells_file=cell_db_path+'ll2.csv',
                                              tstop=3000.0)
    axes[0].set_ylim((0, 10000))
    # axes[0].set_ylim((5400, 5600))
    axes[0].xaxis.set_ticklabels([])
    axes[2].set_ylabel('Bkg. (arb. u.)')
    axes[2].set_ylim((0, 14))
    axes[2].yaxis.set_ticks([0, 7, 14])
    # Use a line to indicate the onset and duration of stimulus.
    axes[1].hlines(0.0, t_vis_stim[0], t_vis_stim[1], linewidth=10, color='lightgreen')
    plt.savefig('gratings/gratings_plt_spk_ll2_g8_%d_sdlif_z101.png'%trial)
    plt.show()
