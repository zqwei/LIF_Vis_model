from glob import glob
from os.path import splitext, exists
from os import makedirs
# from os import system
import json
# import numpy as np

if len(glob('anton_config_test/config_*.json')) > 0:
    for file_name in glob('anton_config_test/config_*.json'):
            with open(file_name, 'r') as input_file:
                new_file_name = splitext(file_name)[0][18:]
                data = json.load(input_file)
                data["biophys"][0]["output_dir"] = data["biophys"][0]["output_dir"]+"_lif_amp_100"
                output_dir = data["biophys"][0]["output_dir"]
                if not exists(output_dir):
                    makedirs(output_dir)
                data["biophys"][0]["model_file"][0] = new_file_name + '_lif_amp_100.json'
                data["syn_data_file"] = "syn_data_278_lif_amp_100.json"
                old_str = [s for s in data["ext_inputs"].keys() if '/allen/aibs/mat/antona' in s][0]
                new_str = old_str.replace('/allen/aibs/mat/antona/network/14-simulations/6-LGN_firing_rates_and_spikes/', '', 2)
                data["ext_inputs"][new_str] = data["ext_inputs"][old_str]
                data["ext_inputs"].pop(old_str)
                with open(new_file_name + '_lif_amp_100.json', 'w') as output_file:
                    json.dump(data, output_file, sort_keys=True, indent=4)
                with open('run' + new_file_name[6:] + '_lif_amp_100.py', 'w') as run_file:
                    run_file.write("import start as start\n\nstart.run_simulation('%s')\n\n" % (new_file_name + '_lif_amp_100.json'))
