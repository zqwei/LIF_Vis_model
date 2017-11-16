import matplotlib.pyplot as plt
import plot_functions

f_list = []
gid_list = range(0, 20, 2)
for gid in gid_list:
    f_list = ['syn_analysis/output_syn_ll2/v_out-cell-%d.h5' % (gid)]
    fig, axes = plot_functions.plot_series(f_list, 'values', [gid], cells_file='syn_analysis/build/20cells.csv', tstart=7000.0, tstop=30000.0)
    #axes.set_ylim((-83.0, -68.0))
    axes.set_ylabel('V (mV)')
    plt.show()

gid_list = range(1, 20, 2)
for gid in gid_list:
    f_list = ['syn_analysis/output_syn_ll2/i_SEClamp-cell-%d.h5' % (gid)]
    fig, axes = plot_functions.plot_series(f_list, 'values', [gid], cells_file='syn_analysis/build/20cells.csv', tstart=7000.0, tstop=30000.0)
    #axes.set_ylim((-0.05, 0.15))
    axes.set_ylabel('I (nA)')
    plt.show()

