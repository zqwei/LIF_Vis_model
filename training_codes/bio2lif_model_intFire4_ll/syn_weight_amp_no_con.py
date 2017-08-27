import json
from os.path import exists
from os import makedirs
# import numpy as np
import pandas as pd
from sys import argv

Nnodes = 5
ppn = 24
Ncores = Nnodes * ppn
tstop = 500
f_name = 'll2_g8_8_test500ms_inh_lif_syn_z'


def syn_weight_amp(output_folder, syn_file=''):
    all_data = pd.read_csv(output_folder + '/cell_update_stats.dat')
    amp_syn = all_data['w_curr']
    with open('syn_data_lif_z101.jsonbak') as data_file:
        data = json.load(data_file)
        data["Scnn1a"]["tw_exc"]["w"] = amp_syn[0] * data["Scnn1a"]["tw_exc"]["w"]
        data["Rorb"]["tw_exc"]["w"] = amp_syn[1] * data["Rorb"]["tw_exc"]["w"]
        data["Nr5a1"]["tw_exc"]["w"] = amp_syn[2] * data["Nr5a1"]["tw_exc"]["w"]
        data["PV1"]["tw_exc"]["w"] = amp_syn[3] * data["PV1"]["tw_exc"]["w"]
        data["PV2"]["tw_exc"]["w"] = amp_syn[4] * data["PV2"]["tw_exc"]["w"]
        with open('syn_data_lif_z' + syn_file + '.json', 'w') as outfile:  # rewrite the syn file
            json.dump(data, outfile, indent=4)
    with open('config_ll2_g8_8_sdlif_z101_test500ms_no_con.json') as data_file:  # rewrite the config file
        data = json.load(data_file)
        new_config_file = 'config_' + f_name + syn_file + '.json'
        new_output_dir = 'output_' + f_name + syn_file
        data["biophys"][0]["output_dir"] = new_output_dir
        data["run"]["tstop"] = float(tstop)
        data["postprocessing"]["in_t_omit"] = float(tstop)
        if not exists(new_output_dir):
            makedirs(new_output_dir)
        data["biophys"][0]["model_file"][0] = new_config_file
        data["syn_data_file"] = 'syn_data_lif_z' + syn_file + '.json'
        with open(new_config_file, 'w') as config_file:
            json.dump(data, config_file, indent=4)
        with open('run_' + f_name + syn_file + '.py', 'w') as run_file:
            run_file.write("import start as start\n\nstart.run_simulation('%s')\n\n" % (new_config_file))


if __name__ == '__main__':
    syn_weight_amp('output_' + f_name + argv[1], argv[1])
    jobname = f_name + argv[1]
    workdir = 'output_' + jobname
    startfile = 'run_' + jobname + '.py'
    qsub_file_name = 'qsub_' + jobname + '.qsub'
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
