from common import *
from syn_uniform import *

# Function for establishing connections; it is assumed that all connections are of the same type and project to the same target cell;
# ideally, this implies connections from one source to one target cell, with an arbitrary number of synapses.
def con_src_tar(src_obj_list, src_type, tar_gid, target, external_flag, utils_obj, d_f_dict):
  target_type = cell_types[type_index(tar_gid)]
  syn_data_types_tmp = utils_obj.description.data['syn_data_types'][target_type][src_type]
  N_syn = len(src_obj_list)

  # Distribute synapses on the dendritic tree of the target neuron.
  # Do not include LIF neurons in this process.
  if ( target_type not in ['LIF_exc', 'LIF_inh'] ):
    sec_labels = syn_data_types_tmp['sec']
    dcutoff = syn_data_types_tmp['dcutoff']
    rand_t = h.Randstream(tar_gid, len(common_rand_stream_dict[tar_gid]))
    dumSynList = syn_uniform(N_syn, tar_gid, target, sec_labels, dcutoff, rand_t, d_f_dict)
    common_rand_stream_dict[tar_gid].append(rand_t)

    for j in xrange(N_syn):
      common_syn_list.append(dumSynList[j])
      dumSynList[j].e = syn_data_types_tmp['e']
      dumSynList[j].tau1 = syn_data_types_tmp['tau1']
      dumSynList[j].tau2 = syn_data_types_tmp['tau2']

  # Establish connections.
  for j in xrange(N_syn):
    if ( target_type not in ['LIF_exc', 'LIF_inh'] ):
      tar_obj = dumSynList[j]
      syn_weight = syn_data_types_tmp['w']  # Mean synaptic conductance in uS (for Exp2Syn).
    else:
      tar_obj = target.ac
      # For LIF neurons, use positive or negative weights for excitatory and inhibitory neurons, respectively.
      if (syn_data_types_tmp['e'] < -55.0) :
        syn_weight = -1.0 * syn_data_types_tmp['w']
      else:
        syn_weight = syn_data_types_tmp['w']

    if (external_flag == 'external'):
      nc = h.NetCon(src_obj_list[j], tar_obj)
    else:
      # Here, src_obj_list should be the list of source gids.
      nc = pc.gid_connect(src_obj_list[j], tar_obj)

    nc.weight[0] = syn_weight
    nc.delay = syn_data_types_tmp['delay']

    common_nc_list.append(nc)

