from common import *
import linecache
from syn_uniform import *
from con_src_tar import *
from build_all_d_distributions import *

import matplotlib.pyplot as plt

ext_in_netstim_list = []
ext_in_vecstim_list = []
ext_in_train_vec_list = []
ext_in_vecstim = h.VecStim()


def external_inputs(ext_in_tar_cell_gid, f_ext_in_name, t_stop, utils_obj):
  target = cells[ext_in_tar_cell_gid]
  target_type = cell_types[type_index(ext_in_tar_cell_gid)]

  # Read information about external inputs from a file.
  f_train_name = []
  f_train_line_pos = []
  source_type = []
  N_syn = []
  random_ISI_file_t_shift = []
  f_ext_in = open(f_ext_in_name, 'r')
  for line in f_ext_in:
    tmp_l = line.split()
    f_train_name.append(tmp_l[0]) # File from which we should read the spike train; if it is 'random', use randomly generated Poisson spike trains.
    f_train_line_pos.append(int(tmp_l[1])) # Read the file and find the line number f_train_line_pos; do not use it if f_train_name = 'random'.
    source_type.append(tmp_l[2]) # Presynaptic type. 
    N_syn.append(int(tmp_l[3])) # Number of synapses to produce with inputs from this line of f_ext_in.
    random_ISI_file_t_shift.append(float(tmp_l[4])) # If f_train_name = 'random', use this as ISI; otherwise, use it as uniform time shift for all spikes in the train (both in ms).
  f_ext_in.close()

  # Create a dictionary with distributions of distances from the soma for all sections.  The dictionary
  # contains this information for ALL combinations of section labels ('basal', 'apical', etc.) that are
  # found for this cell given the inputs from the file.
  d_f_dict = {}
  if (target_type not in ['LIF_exc', 'LIF_inh']):
    all_sec_label_lists = []
    for src_type_tmp in source_type:
      all_sec_label_lists.append(utils_obj.description.data['syn_data_types'][target_type][src_type_tmp]['sec'])
    d_f_dict = build_all_d_distributions(all_sec_label_lists, ext_in_tar_cell_gid)

  # For each line in the input file, establish the appropriate number of connections.
  for i_line in xrange(len(f_train_name)):
    src_obj_list = []

    # Read the spike times for the spike train (if f_train_name[i_line] != 'random').
    if (f_train_name[i_line] != 'random'):
      spike_t = [float(x) for x in linecache.getline(f_train_name[i_line], f_train_line_pos[i_line]).split()] # For linecache, numbering of lines seems to be starting from 1, which is the convention we use in external_inputs files as well.

    if (f_train_name == 'random'): # Initiate random stream.
      rand_t = h.Randstream(ext_in_tar_cell_gid, len(common_rand_stream_dict[ext_in_tar_cell_gid]))
      rand_t.r.negexp(1.0)
      common_rand_stream_dict[ext_in_tar_cell_gid].append(rand_t)

    # Generate appropriate sources for each input.
    for j in xrange(N_syn[i_line]):

      if (f_train_name[i_line] == 'random'):
        dumNetStim = h.NetStim(.5)
        dumNetStim.noiseFromRandom(rand_t.r)
        dumNetStim.interval = random_ISI_file_t_shift[i_line] # mean ISI (ms)
        dumNetStim.number = 5 + int(t_stop / random_ISI_file_t_shift[i_line]) # Pad the number of events just a little bit, because otherwise, if random_ISI_file_t_shift > t_stop, no events are generated at all; the padding number would be the maximum number of events that may be generated within t-stop in that situation, but in most cases there will be fewer generated, as the event times are on average separated by delta_t = random_ISI_file_t_shift.
        dumNetStim.noise = 1.0
        dumNetStim.start = 0.0
        ext_in_netstim_list.append(dumNetStim)
        src_obj_list.append(dumNetStim)

      else:
        ext_in_train_vec = h.Vector()
        if (random_ISI_file_t_shift[i_line] != 0.0):
          spike_t = [(x + random_ISI_file_t_shift[i_line]) for x in spike_t]
        for x in spike_t:
          ext_in_train_vec.append(x)
        ext_in_train_vec_list.append(ext_in_train_vec)
        ext_in_vecstim = h.VecStim()
        ext_in_vecstim_list.append(ext_in_vecstim)
        ext_in_vecstim.play(ext_in_train_vec)
        src_obj_list.append(ext_in_vecstim)

    con_src_tar(src_obj_list, source_type[i_line], ext_in_tar_cell_gid, target, 'external', utils_obj, d_f_dict)

