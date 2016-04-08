import json
from os.path import exists
from os import makedirs
import numpy as np

f_name = 'll2_g8_8_test500ms_LGN_only_no_con_lif_syn_z'


def syn_weight_amp(amp_syn_output, syn_file=''):
    amp_syn = np.load(amp_syn_output+'.npy')
    amp_Scann1a = amp_syn[0, 2]
    amp_Rorb = amp_syn[1, 2]
    amp_Nr5a1 = amp_syn[2, 2]
    amp_PV1 = amp_syn[3, 2]
    amp_PV2 = amp_syn[4, 2]
    with open('syn_data_278_lif_amp_100.jsonbak') as data_file:
        data = json.load(data_file)
        data["Scnn1a"]["LGN_exc"]["w"] = amp_Scann1a * data["Scnn1a"]["LGN_exc"]["w"]
        data["Rorb"]["LGN_exc"]["w"] = amp_Rorb * data["Rorb"]["LGN_exc"]["w"]
        data["Nr5a1"]["LGN_exc"]["w"] = amp_Nr5a1 * data["Nr5a1"]["LGN_exc"]["w"]
        data["PV1"]["LGN_exc"]["w"] = amp_PV1 * data["PV1"]["LGN_exc"]["w"]
        data["PV2"]["LGN_exc"]["w"] = amp_PV2 * data["PV2"]["LGN_exc"]["w"]
        with open(syn_file+'.json', 'w') as outfile:  # rewrite the syn file
            json.dump(data, outfile, indent=4)
    with open('config_ll2_g8_8_sd278_test500ms_LGN_only_no_con.json') as data_file:  # rewrite the config file
        data = json.load(data_file)
        new_config_file = 'config_'+f_name+syn_file+'.json'
        new_output_dir = 'output_'+f_name+syn_file
        data["biophys"][0]["output_dir"] = new_output_dir
        if not exists(new_output_dir):
            makedirs(new_output_dir)
        data["biophys"][0]["model_file"][0] = new_config_file
        data["syn_data_file"] = syn_file+'.json'
        with open(new_config_file, 'w') as config_file:
            json.dump(data, config_file, indent=4)
        with open('run_'+f_name+syn_file+'.py', 'w') as run_file:
            run_file.write("import start as start\n\nstart.run_simulation('%s')\n\n" % (new_config_file))