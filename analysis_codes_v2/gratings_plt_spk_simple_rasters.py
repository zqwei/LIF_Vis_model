import matplotlib.pyplot as plt
import plot_functions

t_sim_stop = 3000.0

#fig, axes = plot_functions.plot_spikes('../simulations_ll2/gratings/output_ll2_g8_8_sd278/spk.dat', cells_file='../build/ll2.csv', tstop=t_sim_stop)
#axes.set_ylim((0, 10000))
#axes.set_xlim((0, t_sim_stop))
#plt.show()

fig, axes = plot_functions.plot_spikes('/data/mat/slg/ice/sims/layer4/ll2/output002/spikes.txt', cells_file='../build/ll2.csv', tstop=t_sim_stop)
axes.set_ylim((0, 10000))
axes.set_xlim((0, t_sim_stop))
plt.show()


