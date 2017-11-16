import matplotlib.pyplot as plt
import plot_functions

#fig, axes = plot_functions.plot_spikes('../output_ll2_spont_8_sd278/spk.dat', cells_file='../build/ll2.csv')
#axes.set_xlim((0.0, 1000.0))
#axes.set_ylim((0, 45000))
#plt.show()

tw_src_id = 552
#fig, axes = plot_functions.plot_spikes_tw('../simulations_ll2/spont/output_ll2_spont_8_sd278/spk.dat', '../tw_data/ll2_tw_build/2_tw_src/f_rates_8.pkl', tw_src_id, cells_file='../build/ll2.csv', tstop=1000.0)
fig, axes = plot_functions.plot_spikes_tw('../simulations_ll2/spont/output_ll2_spont_3000ms_8_sd278/spk.dat', '../tw_data/ll2_tw_build/2_tw_src/f_rates_8.pkl', tw_src_id, cells_file='../build/ll2.csv', tstop=3000.0)
axes[0].set_ylim((0, 45000))
axes[0].xaxis.set_ticklabels([])
axes[2].set_ylabel('Bkg. (arb. u.)')
#plt.savefig('spont_activity/spont_plt_spk_ll2_spont_8_sd278.eps', format='eps')
plt.savefig('spont_activity/spont_plt_spk_ll2_spont_3000ms_8_sd278.eps', format='eps')
plt.show()


