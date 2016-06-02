import json
import pandas as pd

tstop = 500
f_name = 'll2_g8_8_test%dms_LGN_only_no_con_lif_syn_z' % (tstop)


def syn_weight_amp(intput_folder, syn_file=''):
    all_data = pd.read_csv(intput_folder + '/cell_update_stats_old.dat')
    amp_syn = all_data['w_curr']
    amp_Scann1a = amp_syn[0]
    amp_Rorb = amp_syn[1]
    amp_Nr5a1 = amp_syn[2]
    amp_PV1 = amp_syn[3]
    amp_PV2 = amp_syn[4]
    with open('syn_data_278_lif_amp_100.jsonbak') as data_file:
        data = json.load(data_file)
        data["Scnn1a"]["LGN_exc"]["w"] = amp_Scann1a * data["Scnn1a"]["LGN_exc"]["w"]
        data["Rorb"]["LGN_exc"]["w"] = amp_Rorb * data["Rorb"]["LGN_exc"]["w"]
        data["Nr5a1"]["LGN_exc"]["w"] = amp_Nr5a1 * data["Nr5a1"]["LGN_exc"]["w"]
        data["PV1"]["LGN_exc"]["w"] = amp_PV1 * data["PV1"]["LGN_exc"]["w"] / 2.
        data["PV2"]["LGN_exc"]["w"] = amp_PV2 * data["PV2"]["LGN_exc"]["w"] / 2.
        # print(data["Scnn1a"]["tw_exc"]["w"]*1000.)
        # print(data["Rorb"]["tw_exc"]["w"]*1000.)
        # print(data["Nr5a1"]["tw_exc"]["w"]*1000.)
        # print(data["PV1"]["tw_exc"]["w"]*1000.)
        # print(data["PV2"]["tw_exc"]["w"]*1000.)
        with open('syn_data_278_lif_amp_100_LGN_PV1x.jsonbak', 'w') as outfile:  # rewrite the syn file
            json.dump(data, outfile, indent=4)

if __name__ == '__main__':
    syn_weight_amp('output_ll2_g8_8_test500ms_LGN_only_no_con_lif_syn_z129')
