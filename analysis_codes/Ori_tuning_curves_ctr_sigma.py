import numpy as np
import matplotlib
matplotlib.rcParams.update({'font.size': 15})
import matplotlib.pyplot as plt
import pandas as pd

cell_db_path = '/allen/aibs/mat/antona/network/14-simulations/9-network/build/'
cells_file = cell_db_path + 'll2.csv'
cells_db = pd.read_csv(cells_file, sep=' ')

# Use sigma from the pre-computed Gaussian fits for tuning curves as a measure of the tuning width.
sim_data_ctr80 = pd.read_csv('Ori/ll2_pref_stat_and_Gfit_4Hz.csv', sep=' ')
sim_data_ctrlow = pd.read_csv('Ori/ll2_ctr10_pref_stat_and_Gfit_4Hz.csv', sep=' ')

# Select those cells that have sufficiently high orientation selectivity.
# Do that based on the selectivity observed for contrast of 80%.
gids_with_high_CV_ori = sim_data_ctr80[sim_data_ctr80['CV_ori'] > 0.5]['id'].values

# Also, select only the cells for which goodness of fit was fairly high.
fit_gids_ctr80 = sim_data_ctr80[sim_data_ctr80['Gfit_goodness_r'] > 0.95]['id'].values
fit_gids_ctrlow = sim_data_ctrlow[sim_data_ctrlow['Gfit_goodness_r'] > 0.95]['id'].values

gids_sel_ctr80 = np.intersect1d(gids_with_high_CV_ori, fit_gids_ctr80)
gids_sel_ctrlow = np.intersect1d(gids_with_high_CV_ori, fit_gids_ctrlow)

gids_sel = np.intersect1d(gids_sel_ctr80, gids_sel_ctrlow)

for type in ['Scnn1a', 'Rorb', 'Nr5a1']:
    type_gids = cells_db[cells_db['type'] == type]['index'].values
    current_gids = np.intersect1d(gids_sel, type_gids)

    sel_data_ctr80 = sim_data_ctr80[sim_data_ctr80['id'].isin(current_gids)]
    sel_data_ctrlow = sim_data_ctrlow[sim_data_ctrlow['id'].isin(current_gids)]

    plt.scatter(sel_data_ctr80['Gfit_sigma'], sel_data_ctrlow['Gfit_sigma'])
    plt.plot([0.0, 100.0], [0.0, 100.0], '--', c='gray')
    plt.xlim((10.0, 50.0))
    plt.ylim((10.0, 50.0))
    plt.xlabel('Contrast 80% HHWH (Degrees)')
    plt.ylabel('Contrast 10% HHWH (Degrees)')
    plt.title('%s' % (type))
    plt.show()

    fit_sigma_dif = sel_data_ctr80['Gfit_sigma'].values - sel_data_ctrlow['Gfit_sigma'].values
    # print '%s: average HHWH difference (Degrees): %f +/- %f' % (type, fit_sigma_dif.mean(), fit_sigma_dif.std())

    plt.hist(fit_sigma_dif, bins=np.arange(-18.0, 18.0, 3.0))
    plt.xlim((-23.0, 23.0))
    #plt.ylim((10.0, 50.0))
    plt.xlabel('HHWH difference (Degrees)')
    plt.ylabel('Number of cells')
    plt.title('%s' % (type))
    plt.show()

#plt.plot(sim_data_ctr80['id'], sim_data_ctr80['Gfit_sigma'], '-o', label='ctr=80%')
#plt.plot(sim_data_ctrlow['id'], sim_data_ctrlow['Gfit_sigma'], '-o', label='ctr=10%')
#plt.legend()
#plt.xlabel('gid')
#plt.ylabel('Sigma (degrees)')
#plt.show()
#
#plt.plot(sim_data_ctr80['id'], sim_data_ctrlow['Gfit_sigma']/sim_data_ctr80['Gfit_sigma'], '-o')
#plt.xlabel('gid')
#plt.ylabel('Sigma(10%)/Sigma(80%)')
#plt.show()
#
#sigma_ratio = sim_data_ctrlow['Gfit_sigma'].values[0:8500]/sim_data_ctr80['Gfit_sigma'].values[0:8500]
#print 'Biophys. exc. cells, mean+/-std. of Sigma(10%%)/Sigma(80%%), %f+/-%f' % (sigma_ratio.mean(), sigma_ratio.std())
