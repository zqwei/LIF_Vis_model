import json
import os
from os import path, makedirs

Nnodes = 5
ppn = 24
Ncores = Nnodes * ppn

system_name = 'll3'
cell_db_path = '/allen/aibs/mat/antona/network/14-simulations/9-network/build/%s.csv' % (system_name)
con_path = '/allen/aibs/mat/antona/network/14-simulations/9-network/build/%s_connections' % (system_name)

update_tw_trial_id = 'yes'
use_vis_stim_path_only = 'no'

vis_map = '/allen/aibs/mat/antona/network/14-simulations/9-network/build/ll3_inputs_from_LGN.csv'
tw_map = '/allen/aibs/mat/antona/network/14-simulations/9-network/tw_data/ll3_tw_build/mapping_3_tw_src.csv'

syn_data_id = 'lif_z101'
syn_data_file = 'syn_data_%s.json' % (syn_data_id)

tstop = 3000.0
tw_trial_id = 180
for grating_id in xrange(8, 240, 30):
    for trial in xrange(0, 10):
        jobname = '%s_g%d_%d_sd%s' % (system_name, grating_id, trial, syn_data_id)
        vis_stim_path = '/allen/aibs/mat/antona/network/14-simulations/6-LGN_firing_rates_and_spikes/LGN_spike_trains/output3/grating_%d_LGN_spk.dat' % (grating_id)
        vis_t_shift = 0.0
        vis_trials_in_file = 10
        workdir = 'output_' + jobname
        startfile = 'run_' + jobname + '.py'
        configname = 'config_' + jobname + '.json'
        qsub_file_name = 'qsub_' + jobname + '.qsub'
        vis_dict = {'mode': 'file', 'map': vis_map, 'trial': trial, 't_shift': vis_t_shift, 'trials_in_file': vis_trials_in_file}
        tw_stim_path = '/allen/aibs/mat/antona/network/14-simulations/9-network/tw_data/%s_tw_build/3_tw_src/%d_spk.dat' % (system_name, tw_trial_id)
        tw_t_shift = 0.0
        tw_trials_in_file = 1
        tw_current_trial_within_file = 0  # This is different than tw_trial_id (the latter refers to the file name).
        tw_dict = {'mode': 'file', 'map': tw_map, 'trial': tw_current_trial_within_file, 't_shift': tw_t_shift, 'trials_in_file': tw_trials_in_file}
        if (update_tw_trial_id == 'yes'):
            tw_trial_id += 1  # Increase the ID of the traveling wave for the next grating/trial combination.
        if not path.exists(workdir):
            makedirs(workdir)
        f = open(startfile, 'w')
        f.write('import start as start\n')
        f.write('\n')
        f.write('start.run_simulation(\'' + ('%s' % (configname)) + '\')\n')
        f.write('\n')
        f.write('\n')
        f.close()
        f_config = open('config_standard.json', 'r')
        config = json.load(f_config)
        f_config.close()
        config['manifest'][2]['spec'] = cell_db_path
        config['connections'] = con_path
        config['biophys'][0]['model_file'][0] = configname
        config['biophys'][0]['output_dir'] = workdir
        if (use_vis_stim_path_only == 'yes'):
            config['ext_inputs'] = {vis_stim_path: vis_dict}
        else:
            config['ext_inputs'] = {vis_stim_path: vis_dict, tw_stim_path: tw_dict}
        config['run']['tstop'] = tstop
        config['cell_data_tracking']['SEClamp_insert_cell_gid_step'] = 200
        config['cell_data_tracking']['SEClamp_insert'] = 'yes'
        config['cell_data_tracking']['SEClamp_insert_first_cell'] = 2
        config['cell_data_tracking']['do_save_t_series'] = 'yes'
        config['syn_data_file'] = syn_data_file
        # del config['connections'] # Remove this for the case without any recurrent connections.
        f_config = open(configname, 'w')
        f_config.write(json.dumps(config, indent=2))
        f_config.close()
        f_out = open(qsub_file_name, 'w')
        f_out.write('#PBS -q mindscope\n')
        f_out.write('#PBS -l walltime=12:00:00\n')
        f_out.write('#PBS -l nodes=' + str(Nnodes) + ':ppn=' + str(ppn) + '\n')
        f_out.write('#PBS -N ' + jobname + '\n')
        f_out.write('#PBS -r n\n')
        f_out.write('#PBS -j oe\n')
        f_out.write('#PBS -o ' + workdir + '/' + jobname + '.out\n')
        f_out.write('#PBS -m a\n')
        f_out.write('cd $PBS_O_WORKDIR\n')
        f_out.write('\n')
        f_out.write('export LD_PRELOAD=/usr/lib64/libstdc++.so.6\n')
        f_out.write('export PATH=/shared/utils.x86_64/hydra-3.0.4/bin/:$PATH\n')
        f_out.write('\n')
        f_out.write('mpiexec -np ' + str(Ncores) + ' nrniv -mpi ' + startfile + ' > ' + workdir + '/log.txt\n')
        f_out.close
