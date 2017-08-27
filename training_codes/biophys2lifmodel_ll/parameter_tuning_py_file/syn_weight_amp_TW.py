import json
from os.path import exists
from os import makedirs
# import numpy as np
import pandas as pd

tstop = 500
f_name = 'll2_g8_8_test%dms_no_con_lif_syn_z' % (tstop)


def syn_weight_amp(output_folder, syn_file=''):
    all_data = pd.read_csv(output_folder+'/cell_update_stats_old.dat')
    amp_syn = all_data['w_curr']
    amp_Scann1a = amp_syn[0]
    amp_Rorb = amp_syn[1]
    amp_Nr5a1 = amp_syn[2]
    amp_PV1 = amp_syn[3]
    amp_PV2 = amp_syn[4]
    with open('syn_data_278_lif_amp_100_LGN_PV1x.jsonbak') as data_file:
        data = json.load(data_file)
        data["Scnn1a"]["tw_exc"]["w"] = amp_Scann1a * data["Scnn1a"]["tw_exc"]["w"]
        data["Rorb"]["tw_exc"]["w"] = amp_Rorb * data["Rorb"]["tw_exc"]["w"]
        data["Nr5a1"]["tw_exc"]["w"] = amp_Nr5a1 * data["Nr5a1"]["tw_exc"]["w"]
        data["PV1"]["tw_exc"]["w"] = amp_PV1 * data["PV1"]["tw_exc"]["w"]
        data["PV2"]["tw_exc"]["w"] = amp_PV2 * data["PV2"]["tw_exc"]["w"]
        with open('syn_data_lif_z'+syn_file+'.json', 'w') as outfile:  # rewrite the syn file
            json.dump(data, outfile, indent=4)
    with open('config_ll2_g8_8_sd278_test500ms_no_con.json') as data_file:  # rewrite the config file
        data = json.load(data_file)
        new_config_file = 'config_'+f_name+syn_file+'.json'
        new_output_dir = 'output_'+f_name+syn_file
        data["biophys"][0]["output_dir"] = new_output_dir
        data["run"]["tstop"] = float(tstop)
        data["postprocessing"]["in_t_omit"] = float(tstop)
        if not exists(new_output_dir):
            makedirs(new_output_dir)
        data["biophys"][0]["model_file"][0] = new_config_file
        data["syn_data_file"] = 'syn_data_lif_z'+syn_file+'.json'
        with open(new_config_file, 'w') as config_file:
            json.dump(data, config_file, indent=4)
        with open('run_'+f_name+syn_file+'.py', 'w') as run_file:
            run_file.write("import start as start\n\nstart.run_simulation('%s')\n\n" % (new_config_file))
