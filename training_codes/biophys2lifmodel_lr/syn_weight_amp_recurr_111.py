import json
from os.path import exists
from os import makedirs
# import numpy as np
import pandas as pd
from sys import argv

Nnodes = 5
ppn = 32
Ncores = Nnodes * ppn
tstop = 500
f_name = 'lr2_g8_8_test%dms_inh_lif_syn_z' % (tstop)


def syn_weight_amp(output_folder, syn_file=''):
    with open('syn_data_lif_z101.jsonbak') as data_file:
        data = json.load(data_file)
        data["Scnn1a"]["exc"]["w"] = 0.5 * data["Scnn1a"]["exc"]["w"]
        data["Rorb"]["exc"]["w"] = 0.5 * data["Rorb"]["exc"]["w"]
        data["Nr5a1"]["exc"]["w"] = 0.65 * data["Nr5a1"]["exc"]["w"]
        data["PV1"]["exc"]["w"] = 0.85 * data["PV1"]["exc"]["w"]
        data["PV2"]["exc"]["w"] = 0.85 * data["PV2"]["exc"]["w"]
        with open('syn_data_lif_z' + syn_file + '.json', 'w') as outfile:  # rewrite the syn file
            json.dump(data, outfile, indent=4)
    with open('config_lr2_g8_8_sd287_cn0.json') as data_file:  # rewrite the config file
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
            run_file.write("import start0 as start\n\nstart.run_simulation('%s')\n\n" % (new_config_file))


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
