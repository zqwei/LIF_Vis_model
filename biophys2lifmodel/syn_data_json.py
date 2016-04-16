import json
import copy

cell_type = ["Scnn1a", "Rorb", "Nr5a1", "PV1", "PV2"]
syn_type = ["LGN_exc", "exc", "tw_exc", "inh"]

with open('syn_data_278.jsonbak') as data_file:
    data = json.load(data_file)
    for amp_value in xrange(100, 1000, 100):
        output_data = copy.deepcopy(data)
        for n_cell in cell_type:
            for n_syn in syn_type:
                output_data[n_cell][n_syn]["w"] = float(data[n_cell][n_syn]["w"])*float(amp_value)
        with open('syn_data_278_lif_amp_'+str(amp_value)+'.json', 'w') as outfile:
            json.dump(output_data, outfile, sort_keys=True, indent=4)
