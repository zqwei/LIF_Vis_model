import linecache
import random

'''
Generate external inputs from files e.g. 'external_inputs_0_0/cell-0.dat'
Each line in input file:

f_train_name:     File from which we should read the spike train; 
                  if it is 'random', use randomly generated Poisson spike trains.

f_train_line_pos: Read the file and find the line number f_train_line_pos;
                  do not use it if f_train_name = 'random'.
                  
source_type:      Presynaptic type: tw_exc by default

N_syn:            Number of synapses to produce with inputs from this line of f_ext_in;
                  do not use this when using single synapses

random_ISI_file_t_shift: If f_train_name = 'random', use this as ISI; 
                   otherwise, use it as uniform time shift for all spikes in the train (both in ms).

f_ext_in_name   : input file name
t_stop          : end time of simulation (if f_train_name = 'random')
'''

def external_inputs(f_ext_in_name, t_stop): 
    ext_in_train_vec_list = []
    f_train_name = []
    f_train_line_pos = []
    source_type = []
    N_syn = []
    random_ISI_file_t_shift = []
    
    f_ext_in = open(f_ext_in_name, 'r')
    for line in f_ext_in:
        tmp_l = line.split()
        f_train_name.append(tmp_l[0])
        f_train_line_pos.append(int(tmp_l[1])) 
        source_type.append(tmp_l[2])  
        N_syn.append(int(tmp_l[3]))
        random_ISI_file_t_shift.append(float(tmp_l[4])) 
    f_ext_in.close()
    
    for i_line in xrange(len(f_train_name)):
        # Read the spike times for the spike train (if f_train_name[i_line] != 'random').
        if (f_train_name[i_line] != 'random'):
            spike_t = [float(x) for x in linecache.getline(f_train_name[i_line], f_train_line_pos[i_line]).split()] 
        if (f_train_name == 'random'): # generated Poisson spike trains.
        	spike_t = poisson_spike_train(random_ISI_file_t_shift[i_line], t_stop)
    
        # Generate appropriate sources for each input -- single synapse
        # for j in xrange(N_syn[i_line]):
        ext_in_train_vec = []
        
        # Add time delay to spike time
        if (f_train_name[i_line] != 'random') and (random_ISI_file_t_shift[i_line] != 0.0):                
        	spike_t = [(x + random_ISI_file_t_shift[i_line]) for x in spike_t]
        	
        for x in spike_t:
            ext_in_train_vec.append(x)
        ext_in_train_vec_list.append(ext_in_train_vec)
        
    return ext_in_train_vec_list
                
def poisson_spike_train(t_ISI, t_stop): # in ms
	t_total = 0.
	t_spike_train = []
	while t_total<t_stop:
		t_spike = random.expovariate(1./t_ISI)
		t_total = t_total + t_spike
		if t_total<t_stop:
			t_spike_train.append(t_total)
	return t_spike_train