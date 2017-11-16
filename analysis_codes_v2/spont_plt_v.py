import matplotlib.pyplot as plt
import plot_functions

tw_src_id = 552
f_list = []
gid_list = [4700, 8150, 9300] #[1100, 4700, 8150, 8600, 9300]
for gid in gid_list:
    #f_list.append('../simulations_ll2/spont/output_ll2_spont_8_sd278/v_out-cell-%d.h5' % (gid))
    f_list.append('../simulations_ll2/spont/output_ll2_spont_3000ms_8_sd278/v_out-cell-%d.h5' % (gid))
fig, axes = plot_functions.plot_series_tw(f_list, 'values', gid_list, '../tw_data/ll2_tw_build/2_tw_src/f_rates_8.pkl', tw_src_id, cells_file='../build/ll2.csv', tstop=3000.0)
axes[0].set_ylim((-92.0, 38.0))
axes[0].set_ylabel('V (mV)')
axes[0].xaxis.set_ticklabels([])
axes[2].set_ylabel('Bkg. (arb. u.)')
#plt.savefig('spont_activity/spont_plt_v_ll2_spont_8_sd278.eps', format='eps')
plt.savefig('spont_activity/spont_plt_v_ll2_spont_3000ms_8_sd278.eps', format='eps')
plt.show()


