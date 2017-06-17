import pandas as pd
import numpy as np
from glob import glob
from os.path import splitext
from os import system

LIFModelDF = np.load('LIFModel.npy')

if len(glob('LIF_*_bak.hoc')) == 0:
	for file_name in glob('LIF_*.hoc'):
		if file_name not in ['LIF_pyramid.hoc', 'LIF_interneuron.hoc']:
			input_file = open(file_name, 'r')
			cell_type = splitext(file_name)[0][4:]
			output_file = open(file_name+'ss', 'w')
			index_array = LIFModelDF['type'] == cell_type
			for line in input_file:
				if 'ac.tau' in line:
					outputString = ''
					for s in line.split():
						if s.replace(".","",1).isdigit():
							outputString = outputString + str(LIFModelDF[index_array]['tau_m'][0])
						else:
							outputString = outputString + s + '  '
					output_file.write('\t' + outputString + '\n')
				elif 'ac.refrac' in line:
					outputString = ''
					for s in line.split():
						if s.replace(".","",1).isdigit():
							outputString = outputString + str(LIFModelDF[index_array]['t_ref'][0])
						else:
							outputString = outputString + s + '  '
					output_file.write('\t' + outputString + '\n')
				else:
					output_file.write(line + '\n')
			input_file.close()
			output_file.close()
			system('mv ' + file_name + ' LIF_' + cell_type + '.hoc.bak')
			system('mv ' + file_name+'ss ' + file_name)
