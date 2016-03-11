import numpy as np
import matplotlib.pyplot as plt
from glob import glob

output_list = glob('output_*')

for base_dir in output_list:
  filename = 'spk.dat'
  full_f_name = base_dir +'/' + filename
  print 'Processing file %s.' % (full_f_name)
  series = np.genfromtxt(full_f_name, delimiter=' ')

  if (series.size > 2):
    plt.scatter(series[:, 0], series[:, 1], s=1, c='k')
    plt.title('%s' % (base_dir))
    plt.show()
