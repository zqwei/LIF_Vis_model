import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import json
import nest
import nest.raster_plot
from numpy import exp
import time
from external_inputs import *

cellNickName = {318808427: 'Nr5a1', 395830185: 'Scnn1a', 314804042: 'Rorb', 330080937: 'PV1', 318331342: 'PV2'}

LIFModelDF = np.load('LIFModel.npy')
cellData = pd.read_csv('V1_L4.csv', sep=' ', index_col='index')

with open('syn_data.json') as syn_data_file:
    syn_params = json.load(syn_data_file)


nest.ResetKernel()
nest.SetStatus([0], [{'resolution':0.001}])

neuron_node = nest.Create('iaf_cond_exp', len(cellData))
multimeter = nest.Create("multimeter", len(cellData))
nest.SetStatus(multimeter, {"withtime":True, "record_from":["V_m", "g_ex", "g_in"]})
nest.Connect(multimeter, neuron_node, 'one_to_one')

t_stop = 5000.

inputs_node_list = [];

for ncell in xrange(len(cellData)):
    f_ext_in_name = 'external_inputs_0_0/cell-%d.dat' % (ncell)
    input_trains = external_inputs(f_ext_in_name, t_stop)
    inputs_node = nest.Create("spike_generator", len(input_trains))
    input_train_dict = []
    for n_node in xrange(len(input_trains)):
        input_train_dict.append({"spike_times": input_trains[n_node]})

    nest.SetStatus(inputs_node, input_train_dict)
    inputs_node_list.append(inputs_node)

    n_cell_type = cellData['type'][0]
    index_array = LIFModelDF['type']==n_cell_type
    neuronparams = { 'g_L':LIFModelDF[index_array]['C_m'][0]/LIFModelDF[index_array]['tau_m'][0],
                     'V_th':LIFModelDF[index_array]['V_th'][0],
                     'E_L':LIFModelDF[index_array]['E_L'][0],
                     't_ref':LIFModelDF[index_array]['t_ref'][0],
                     'V_reset':LIFModelDF[index_array]['V_reset'][0],
                     'C_m':LIFModelDF[index_array]['C_m'][0],
                     'V_m':LIFModelDF[index_array]['E_L'][0]} #initial membrane potential
    synapseparams = {'E_ex': syn_params[n_cell_type]['exc']['e'],
                     'E_in': syn_params[n_cell_type]['inh']['e'],
                     'tau_syn_ex': syn_params[n_cell_type]['exc']['tau2'],
                     'tau_syn_in': syn_params[n_cell_type]['inh']['tau2']}
    nest.SetStatus([neuron_node[ncell]], neuronparams);
    nest.SetStatus([neuron_node[ncell]], synapseparams);

    nest.Connect(inputs_node_list[ncell], [neuron_node[ncell]], 'all_to_all',
                 {"weight": syn_params[n_cell_type]['tw_exc']['w']*1000.,
                  "delay": syn_params[n_cell_type]['tw_exc']['delay']})


# simulation
nest.Simulate(t_stop)

dmm = nest.GetStatus(multimeter)[0]
Vms = dmm["events"]["V_m"]
ts = dmm["events"]["times"]
g_exs = dmm["events"]["g_ex"]
g_ins = dmm["events"]["g_in"]

fig, ax = plt.subplots()
ax.plot(ts-ts.min(), Vms,'-k',label='Vm')
plt.show()
