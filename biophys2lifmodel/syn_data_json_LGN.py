import json
import numpy as np

cell_type = ["Scnn1a", "Rorb", "Nr5a1", "PV1", "PV2"]


def write_syn_data_json(amp_syn_output):
	amp_syn = np.load(amp_syn_output+'.npy')
	amp_arr = amp_syn[:, 2]
	with open('syn_data_278_lif_amp_100.jsonbak','r') as data_file:
		data = json.load(data_file)
		for n_cell in cell_type:
			data[n_cell]["LGN_exc"]["w"] = float(data[n_cell]["LGN_exc"]["w"])*float(amp_arr[n_cell])
		with open('syn_data_lif_z'+amp_syn_output+'.json', 'w') as outfile:
			json.dump(data, outfile, indent=4)
