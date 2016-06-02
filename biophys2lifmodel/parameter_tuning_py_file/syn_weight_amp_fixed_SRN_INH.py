# based on the results from syn_weight_amp_init_INH.py
import json
import pandas as pd


def syn_weight_amp(intput_folder, syn_file=''):
    all_data = pd.read_csv(intput_folder + '/cell_update_stats_old.dat')
    amp_syn = all_data['w_curr'][0]
    with open('syn_data_278_lif_amp_100_init_INH.jsonbak') as data_file:
        data = json.load(data_file)
        data["Scnn1a"]["inh"]["w"] = amp_syn * data["Scnn1a"]["inh"]["w"]
        data["Rorb"]["inh"]["w"] = amp_syn * data["Rorb"]["inh"]["w"]
        data["Nr5a1"]["inh"]["w"] = amp_syn * data["Nr5a1"]["inh"]["w"]
        with open('syn_data_278_lif_amp_100_INH_SRN.jsonbak', 'w') as outfile:  # rewrite the syn file
            json.dump(data, outfile, indent=4)

if __name__ == '__main__':
    syn_weight_amp('results/test500ms_all, bio taus, strategy 1/output_ll2_g8_8_test500ms_inh_lif_syn_z176')
