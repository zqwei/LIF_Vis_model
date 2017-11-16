import f_rate
import json
import pandas as pd

sim_name_list = []
dir_prefix_list = []

#for sys_name in ['ll1', 'll2', 'll3']:
#    for trial in xrange(0, 20):
#        sim_name_list.append('%s_spont_%d_sd278' % (sys_name, trial))

sys_name = 'll1'
grating_id_list = [67] #range(6, 240, 30) + range(7, 240, 30) + range(8, 240, 30) + range(9, 240, 30)
sd_str = 'sd278'
#sys_name = 'll2'
#grating_id_list = [7] #range(7, 240, 30) + range(8, 240, 30) + range(9, 240, 30)
#sd_str = 'sd278'
#sys_name = 'll3'
#grating_id_list = range(8, 240, 30)
#sd_str = 'sd278'
#sys_name = 'lr2'
#grating_id_list = range(8, 240, 30)
#sd_str = 'sd287_cn0'
#sys_name = 'rl2'
#grating_id_list = range(218, 240, 30)
#sd_str = 'sd285'
#sys_name = 'rr2'
#grating_id_list = range(8, 240, 30)
#sd_str = 'sd282_cn0'

for grating_id in grating_id_list:
    for trial in xrange(0, 10):
        sim_name_list.append('%s_g%d_%d_%s_LGN_only_no_con' % (sys_name, grating_id, trial, sd_str))

        #if (sys_name in ['ll1', 'lr1', 'rl1', 'rr1']):
        #    dir_prefix_list.append('../simulations_%s' % (sys_name))
        #else:
        #    dir_prefix_list.append('..')
        dir_prefix_list.append('../simulations_%s/gratings' % (sys_name))


for i_sim, sim_name in enumerate(sim_name_list):

    print 'Processing spikes from simulation %s.' % (sim_name)
    dir_prefix = dir_prefix_list[i_sim]
    f = open('%s/config_%s.json' % (dir_prefix, sim_name), 'r')
    config = json.load(f)
    f.close()

    workdir_n = '%s/%s' % (dir_prefix, config['biophys'][0]['output_dir'])
    for x in config['manifest']:
        if (x['key'] == 'CELL_DB'):
            cells_fname = '../%s' % (x['spec'])
    cell_db = pd.read_csv(cells_fname, sep=' ')
    N_cells = cell_db['index'].values.size

    f_rate.tot_f_rate(workdir_n+'/spk.dat', workdir_n+'/tot_f_rate.dat', config['postprocessing']['in_t_omit'], (config['run']['tstop'] - config['postprocessing']['post_t_omit']), config['run']['tstop'], N_cells)


