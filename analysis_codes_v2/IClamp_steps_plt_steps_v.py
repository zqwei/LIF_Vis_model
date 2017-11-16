import matplotlib.pyplot as plt
import plot_functions

f_list = []
gid_list = [4700, 8150, 9300] #[1100, 4700, 8150, 8600, 9300]
for gid in gid_list:
    f_list.append('../output_ll2_g8_0_sd278_IClamp_steps/v_out-cell-%d.h5' % (gid))
fig, axes = plot_functions.plot_series(f_list, 'values', gid_list, cells_file='../build/ll2.csv', tstop=3000.0)
axes.set_ylim((-92.0, 38.0))
axes.set_ylabel('V (mV)')
plt.show()


