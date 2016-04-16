import numpy as np
import matplotlib.pyplot as plt
from math import floor
from numpy.fft import rfft, irfft
from glob import glob

tstop = 500
f_name = 'll2_g8_8_test%dms_LGN_only_no_con_lif_syn_z' % (tstop)
num_cell_type = [0, 3700, 7000, 8500, 9300, 10000, 39750, 45000]
cell_type = ['Scnn1a', 'Rorb', 'Nr5a1', 'PV1', 'PV2']
ref_file = 'output_ll2_g8_8_test%dms_LGN_only_no_con_syn_z002/tot_f_rate.dat' % (tstop)


def running_mean(x, N):
    # fft is used here (circular convolution)
    # filter
    sfilter = np.ones(N)/float(N)
    zeropadding_sfilter = np.zeros(len(x))
    zeropadding_sfilter[int(floor((len(x)-N)/2.)):int(floor((len(x)-N)/2.)+N)] = sfilter
    fft_signal = rfft(x)
    fft_filter = rfft(zeropadding_sfilter)
    return irfft(fft_filter*fft_signal)
    # cumsum = np.cumsum(np.insert(x, 0, 0))
    # return (cumsum[N:] - cumsum[:-N])/N


def compute_mean(cell_id, tot_rate, N=100):
    sliding_tot_rate = tot_rate
    for cell_group in xrange(len(cell_type)):
        group = np.logical_and(cell_id >= num_cell_type[cell_group], cell_id < num_cell_type[cell_group+1])
        sliding_tot_rate[group] = running_mean(tot_rate[group], N)
    return sliding_tot_rate


def plot_tot_firing_rate_comparison(n_file_dir, ref_file=ref_file, N=100, ncol=2):
    series = np.genfromtxt(ref_file, delimiter=' ')
    ref_firing_rate = series[0:10000, ncol]
    cell_id = series[0:10000, 0]
    series = np.genfromtxt(n_file_dir+'/tot_f_rate.dat', delimiter=' ')
    firing_rate = series[0:10000, ncol]
    smoothed_ref_data = compute_mean(cell_id, ref_firing_rate, N=N)
    smoothed_data = compute_mean(cell_id, firing_rate, N=N)
    plt.plot(cell_id, smoothed_ref_data, '--k')
    plt.plot(cell_id, smoothed_data, '-r', alpha=0.5)
    plt.ylim(bottom=0.0)
    plt.xlabel('gid')
    plt.ylabel('Firing rate (Hz)')
    plt.title('Running average of the firing rate, N = %d' % (N))
    plt.savefig(n_file_dir+'/comparison_plot.png', dpi=150)
    plt.savefig(n_file_dir+'comparison_plot.png', dpi=150)
    # plt.show()
    plt.close('all')

if __name__ == '__main__':
    # for nfile in xrange(101, 109):
    nfile = 110
    plot_tot_firing_rate_comparison('output_ll2_g8_8_test500ms_LGN_only_no_con_lif_syn_z'+str(nfile))
