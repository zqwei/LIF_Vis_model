import matplotlib.pyplot as plt
import plot_functions


t_vis_stim = [500.0, 3000.0]
tw_src_id = 552

# for trial in [8]:
#    print 'Trial %d' % (trial)
#    tw_id = 68 #180 + trial
#    # The system file we use below is for the biophysical system, but since the LIF system has all the same cells except for the level of resoulution employed
#    # to model them, this should work fine.
#    fig, axes = plot_functions.plot_spikes_tw('../model_for_Ziqiang/simulation_results_2016_04_29/output_ll2_g8_%d_test3000ms_inh_lif_syn_z176/spk.dat' % (trial), '../tw_data/ll2_tw_build/2_tw_src/f_rates_%d.pkl' % (tw_id), tw_src_id, cells_file='../build/ll2.csv', tstop=3000.0)

for trial in [1]:
    print 'Trial %d' % (trial)
    tw_id = 180 + trial
    # The system file we use below is for the biophysical system, but since the LIF system has all the same cells except for the level of resoulution employed
    # to model them, this should work fine.
    fig, axes = plot_functions.plot_spikes_tw('/data/mat/ZiqiangW/simulation_ll_syn_data_lif_z102/simulation_ll2/output_ll2_g8_%d_sdlif_z101/spk.dat' % (trial), '../tw_data/ll2_tw_build/2_tw_src/f_rates_%d.pkl' % (tw_id), tw_src_id, cells_file='../build/ll2.csv', tstop=3000.0)
    #fig, axes = plot_functions.plot_spikes_tw('/data/mat/ZiqiangW/simulation_rr2/output_rr2_g8_%d_sdlif_z101/spk.dat' % (trial), '../tw_data/ll2_tw_build/2_tw_src/f_rates_%d.pkl' % (tw_id), tw_src_id, cells_file='../build/ll2.csv', tstop=3000.0)

    axes[0].set_ylim((0, 10000))
    #axes[0].set_ylim((5400, 5600))
    axes[0].xaxis.set_ticklabels([])
    axes[2].set_ylabel('Bkg. (arb. u.)')
    axes[2].set_ylim((0, 14))
    axes[2].yaxis.set_ticks([0, 7, 14])
    # Use a line to indicate the onset and duration of stimulus.
    axes[1].hlines(0.0, t_vis_stim[0], t_vis_stim[1], linewidth=10, color='lightgreen')
    plt.show()


