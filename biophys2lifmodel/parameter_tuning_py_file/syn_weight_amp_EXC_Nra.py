import json
from os.path import exists
from os import makedirs
# import numpy as np
import pandas as pd

tstop = 500
f_name = 'll2_g8_8_test%dms_inh_lif_syn_z' % (tstop)


def syn_weight_amp(output_folder, syn_file=''):
    all_data = pd.read_csv(output_folder + '/cell_update_stats_old.dat')
    amp_syn_Nra = all_data['w_curr'][0]
    with open('syn_data_278_lif_amp_100_INH_Nra.jsonbak') as data_file:
        data = json.load(data_file)
        data["Nr5a1"]["exc"]["w"] = amp_syn_Nra * data["Nr5a1"]["exc"]["w"]
        with open('syn_data_lif_z' + syn_file + '.json', 'w') as outfile:  # rewrite the syn file
            json.dump(data, outfile, indent=4)
    with open('config_ll2_g8_8_sd278_test500ms.json') as data_file:  # rewrite the config file
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
