import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plot_functions
import h5py

from scipy.optimize import leastsq

def exp_function(t, t0, f0, df, params):
    (tau) = params
    return f0 + df * np.exp( (t0 - t) / tau )


def exp_function_fit( params, t, y_av, t0, f0, df ):
    return (y_av - exp_function( t, t0, f0, df, params ))


def exp_fit_within_t_bounds(t, a, t0, t_fit_start, t_fit_end, tau_start):
    ind = np.where( t >= t_fit_start )[0]
    t_tmp = t[ind]
    a_tmp = a[ind]
    ind1 = np.where( t_tmp <= t_fit_end )[0]
    t_tmp = t_tmp[ind1]
    a_tmp = a_tmp[ind1]

    params_start = [tau_start]
    a0 = a_tmp[-1]
    da = a_tmp[0] - a0

    return t_tmp, a0, da, leastsq( exp_function_fit, params_start, args=(t_tmp, a_tmp, t0, a0, da) )





v_threshold = -60.0

#gid = 50
Scnn1a_gids = range(0, 3700, 50)
Rorb_gids = range(3700, 7000, 50)
Nr5a1_gids = range(7000, 8500, 50)
PV1_gids = range(8500, 9300, 50)
PV2_gids = range(9300, 10000, 50)


t_IClamp_steps = [800.0, 1005.0, 1210.0, 1415.0, 1820.0, 2025.0, 2230.0, 2435.0, 2640.0, 2845.0]
#t_IClamp_steps = [800.0, 1000.0, 1200.0, 1400.0, 1800.0, 2000.0, 2200.0, 2400.0, 2600.0, 2800.0]
#t_IClamp_steps = [700.0, 900.0, 1100.0, 1300.0, 1500.0, 1900.0, 2100.0, 2300.0, 2500.0, 2700.0]
t_IClamp_duration = 0.5
t_window = 50.0

cells_file = '../build/ll2.csv'


fig_name = 'IClamp_steps/IClamp_steps_tau_fit_PV2.eps'
f_names = []
for gid in PV2_gids:
    #f_names = []
    #for i_trial in xrange(0, 10):
    #    f_names.append('../output_ll2_g8_%d_sd278_IClamp_steps/v_out-cell-%d.h5' % (i_trial, gid))
    for grating_id in xrange(8, 240, 30):
        f_names.append('../output_ll2_g%d_0_sd278_IClamp_steps/v_out-cell-%d.h5' % (grating_id, gid))



#cells_db = pd.read_csv(cells_file, sep=' ')
for k, f_name in enumerate(f_names):
   print 'Processing file %s.' % (f_name)
   h5 = h5py.File(f_name, 'r')
   values = h5['values'][...]
   tarray = np.arange(0, values.size) * 1.0 * h5.attrs['dt']

   for t_current_IClamp in t_IClamp_steps:
       ind = np.where((tarray >= t_current_IClamp) & (tarray < t_current_IClamp + t_window))[0]
       t_tmp = tarray[ind] - t_current_IClamp
       v_tmp = values[ind]

       # Apply a simple threshold to all voltage values.
       #ind1 = np.where( v_tmp > v_threshold )[0]
       #v_tmp[ind1] = v_threshold

       # Remove traces that have action potential (or close to that -- use a simple threshold rule).
       if ( np.where( v_tmp > v_threshold )[0].size > 0 ):
           continue

       # Shift the trace so that it is considered relative to the point of where the current injection started.
       v_tmp = v_tmp - v_tmp[0]

       if 'v_mean' in dir():
           v_mean += v_tmp
           N_v_mean += 1
       else:
           v_mean = np.copy(v_tmp)
           N_v_mean = 1
       #plt.plot(t_tmp, v_tmp, c='lightgray')
       #plt.plot(t_tmp, v_mean, c='red', linewidth=2)
       #plt.show()

v_mean = v_mean / (1.0 * N_v_mean)

#plt.plot(t_tmp, v_mean, linewidth = 2)
#plt.plot(t_tmp, [v_mean[-600:].mean()]*t_tmp.size, linewidth = 2)
#plt.show()

# We can plot v_mean on the log scale and see where contributions from the fast time constants mostly end, and the
# contribution from the slower time constant remains.  This happens to be on the time scale of approximately 5 to 20 ms.
# This portion of the response can then be used to do the exponential fit.
#plt.plot(t_tmp, -v_mean + 3.0, linewidth = 2)
#plt.yscale('log')
#plt.show()

t_fit_start = 5.0
t_fit_end = 20.0
tau_start = 1.0
t0 = t_fit_start
t_fit_array, v0, dv, ((tau), tmp) = exp_fit_within_t_bounds(t_tmp, v_mean, t0, t_fit_start, t_fit_end, tau_start)

plt.plot(t_tmp, v_mean)
plt.plot(t_fit_array, v0 + dv * np.exp( (t0 - t_fit_array) / tau ), c='green', linewidth = 2)
plt.title('tau = %f ms' % (tau))
plt.annotate('v0 = %f, dv = %f, t0 = %f' % (v0, dv, t0), xy=(0.5, 0.8), xycoords='axes fraction', fontsize=8)
plt.xlabel('Time (ms)')
plt.ylabel('Voltage (mV)')
plt.savefig(fig_name, format='eps')
plt.show()



