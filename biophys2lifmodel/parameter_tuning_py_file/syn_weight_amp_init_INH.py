import json
import pandas as pd

tw_file = 'results/test500ms_no_con, bio taus/output_ll2_g8_8_test500ms_no_con_lif_syn_z119'
tw_data = pd.read_csv(tw_file + '/cell_update_stats_old.dat')
LGN_file = 'results/test500ms_LGN_only_no_con, bio taus/output_ll2_g8_8_test500ms_LGN_only_no_con_lif_syn_z129/'
LGN_data = pd.read_csv(tw_file + '/cell_update_stats_old.dat')

amp_syn = (tw_data['w_curr'] + LGN_data['w_curr']) / 2.
amp_Scann1a = amp_syn[0]
amp_Rorb = amp_syn[1]
amp_Nr5a1 = amp_syn[2]
amp_PV1 = amp_syn[3]
amp_PV2 = amp_syn[4]
with open('syn_data_278_lif_amp_100_TW.jsonbak') as data_file:
    data = json.load(data_file)
    data["Scnn1a"]["exc"]["w"] = amp_Scann1a * data["Scnn1a"]["exc"]["w"]
    data["Rorb"]["exc"]["w"] = amp_Rorb * data["Rorb"]["exc"]["w"]
    data["Nr5a1"]["exc"]["w"] = amp_Nr5a1 * data["Nr5a1"]["exc"]["w"]
    data["PV1"]["exc"]["w"] = amp_PV1 * data["PV1"]["exc"]["w"]
    data["PV2"]["exc"]["w"] = amp_PV2 * data["PV2"]["exc"]["w"]
    data["Scnn1a"]["inh"]["w"] = amp_Scann1a * data["Scnn1a"]["inh"]["w"]
    data["Rorb"]["inh"]["w"] = amp_Rorb * data["Rorb"]["inh"]["w"]
    data["Nr5a1"]["inh"]["w"] = amp_Nr5a1 * data["Nr5a1"]["inh"]["w"]
    data["PV1"]["inh"]["w"] = amp_PV1 * data["PV1"]["inh"]["w"]
    data["PV2"]["inh"]["w"] = amp_PV2 * data["PV2"]["inh"]["w"]
    with open('syn_data_278_lif_amp_100_init_INH.jsonbak', 'w') as outfile:  # rewrite the syn file
        json.dump(data, outfile, indent=4)
