import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import matplotlib
matplotlib.rcParams.update({'font.size': 15})

import matplotlib.gridspec as gspec

def box_plot_sys(ax, data_dict, sys_order, sys_color, label):
    sys_data = []
    for sys in sys_order:
        sys_data.append(data_dict[sys])

    box = ax.boxplot(sys_data, patch_artist=True, sym='c.') # notch=True
    for patch, color in zip(box['boxes'], [sys_color[sys] for sys in sys_order]):
        patch.set_facecolor(color)
        patch.set_edgecolor(color)

    for i, sys in enumerate(sys_order):
        ax.errorbar([i+1], [sys_data[i].mean()], yerr=[sys_data[i].std() / np.sqrt(1.0 * sys_data[i].size)], marker='o', ms=8, color='k', linewidth=2, capsize=5, markeredgewidth=2, ecolor='k', elinewidth=2)
        ind = np.where(sys_data[i] > y_lim_top)[0]
        ax.annotate(u'$\u2191$'+'\n%d/%d' % (ind.size, sys_data[i].size), xy=(i+1.2, 1.0*y_lim_top), fontsize=12)

    ax.set_ylim((0.0, y_lim_top))
    ax.set_xticks(range(1, len(sys_order)+1))
    ax.set_xticklabels(sys_order)

    ax.set_ylabel(label)

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    ax.tick_params(size=10)


result_fig_fname = 'decoding_plt/decoding_plt.eps'

sys_color = {'ll': 'red', 'lr': 'tan', 'rl': 'orange', 'rr': 'black', 'exp': 'gray'}

g_decoding = {}
g_decoding['ll'] = np.load('decoding_error/ll_gratings.npy')
g_decoding['lr'] = np.load('decoding_error/lr_gratings.npy')
g_decoding['rl'] = np.load('decoding_error/rl_gratings.npy')
g_decoding['rr'] = np.load('decoding_error/rr_gratings.npy')
g_decoding['exp'] = np.load('decoding_error/exp_dg.npy')

imq_decoding = {}
imq_decoding['ll'] = np.load('decoding_error/ll_ns.npy')
imq_decoding['rr'] = np.load('decoding_error/rr_ns.npy')
imq_decoding['exp'] = np.load('decoding_error/exp_ns.npy')


# Plot the results.
y_lim_top = 1.0

fig = plt.figure(figsize = (12, 5))

gs = gspec.GridSpec(1, 2)
ax = []
ax.append(fig.add_subplot(gs[0,0]))
ax.append(fig.add_subplot(gs[0,1]))


sys_order = ['ll', 'lr', 'rl', 'rr', 'exp']
box_plot_sys(ax[0], g_decoding, sys_order, sys_color, 'Decoding error')

sys_order = ['ll', 'rr', 'exp']
box_plot_sys(ax[1], imq_decoding, sys_order, sys_color, 'Decoding error')

plt.savefig(result_fig_fname, format='eps')

plt.show()


