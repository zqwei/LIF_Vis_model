import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams.update({'font.size': 20})


tw_src_id = 552

for i in [8]: #xrange(0, 600):
    tw_f_name = '../tw_data/ll2_tw_build/2_tw_src/f_rates_%d.pkl' % (i)

    # Read the background activity trace.
    f = open(tw_f_name, 'r')
    tw_data = pickle.load(f)
    f.close()

    # Plot the magnitude of background activity.
    plt.plot(tw_data['t'], tw_data['cells'][tw_src_id])
    plt.ylabel('Bkg. activity (arb. u.)')
    plt.xlim((0, 5500.0))

    plt.title('%s' % (tw_f_name))

    plt.savefig('tw_example_plt/tw_ll2_%d.eps' % (i), format='eps')

    plt.show()


