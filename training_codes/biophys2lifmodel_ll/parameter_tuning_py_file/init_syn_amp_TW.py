import json
from os.path import exists
from os import makedirs
import numpy as np
import pandas as pd


tstop = 500
f_name = 'll2_g8_8_test%dms_no_con_lif_syn_z' % (tstop)
num_core = 30
num_cell_type = [0, 3700, 7000, 8500, 9300, 10000, 39750, 45000]
cell_type = ['Scnn1a', 'Rorb', 'Nr5a1', 'PV1', 'PV2']
ref_file = 'output_ll2_g8_8_test%dms_no_con_syn_z001/tot_f_rate.dat' % (tstop)

def syn_weight_amp_pd(syn_file=' '):
    n_file_dir = 'output_'+f_name+syn_file
    amp_syn = pd.read_csv(n_file_dir+"/cell_update_stats_old.dat")['w_curr']
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
        sfile = f_name + syn_file
        with open('run_compile.bat', 'w') as output_file:
            output_file.write('mpiexec -np %d nrniv -mpi run_%s.py > output_%s/log.txt\n' % (num_core, sfile, sfile))


def estimate_initial_error(syn_file, ncol=2, ref_file=ref_file):
    series = np.genfromtxt(ref_file, delimiter=' ')
    ref_firing_rate = series[0:10000, ncol]
    cell_id = series[0:10000, 0]
    n_file_dir = 'output_'+f_name+str(syn_file)
    series = np.genfromtxt(n_file_dir+'/tot_f_rate.dat', delimiter=' ')
    firing_rate = series[0:10000, ncol]
    diff_rate = firing_rate-ref_firing_rate
    pdUpdateNew = pd.DataFrame(data=np.random.randn(5, 8), columns=['Cell_type', 'E_old', 'E_curr', 'grad_old', 'grad_curr', 'w_old', 'w_curr', 'dw'])
    pdUpdateNew['Cell_type'] = pdUpdateNew['Cell_type'].astype('str')
    amp_syn = pd.read_csv(n_file_dir+"/cell_update_stats_old.dat")['w_curr']
    for cell_group in xrange(len(cell_type)):
        group = np.logical_and(cell_id >= num_cell_type[cell_group], cell_id < num_cell_type[cell_group+1])
        E_curr = np.mean(diff_rate[group]**2)  # mean
        grad_curr = np.mean(diff_rate[group])  # mean
        pdUpdateNew['Cell_type'][cell_group] = cell_type[cell_group]
        pdUpdateNew['E_old'][cell_group] = E_curr
        pdUpdateNew['grad_old'][cell_group] = grad_curr
        pdUpdateNew['w_old'][cell_group] = amp_syn[cell_group]
        w_old = pdUpdateNew['w_old'][cell_group]
        pdUpdateNew['E_curr'][cell_group] = E_curr
        pdUpdateNew['grad_curr'][cell_group] = grad_curr
        dw = abs(w_old/grad_curr*0.1)
        pdUpdateNew['dw'][cell_group] = dw
        w_curr = w_old - grad_curr * dw
        pdUpdateNew['w_curr'][cell_group] = w_curr
    pdUpdateNew.to_csv(n_file_dir+"/cell_update_stats_new.dat", mode='w', index=False)


def main(starting_idx):
    if not exists('output_'+f_name+str(starting_idx)+'/tot_f_rate.dat'):
        syn_weight_amp_pd(str(starting_idx))
    else:
        estimate_initial_error(str(starting_idx))

if __name__ == '__main__':
    main(100)

