import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

#npy_movie_fname = 'natural_movies/TouchOfEvil_frames_3600_to_3690.npy'
#npy_movie_fname = 'natural_movies/TouchOfEvil_frames_5550_to_5640.npy'
#npy_movie_fname = 'natural_movies/TouchOfEvil_frames_1530_to_1680.npy'
#npy_movie_fname = 'natural_movies/TouchOfEvil_frames_3600_to_3750.npy'
#npy_movie_fname = 'natural_movies/TouchOfEvil_frames_5550_to_5700.npy'

npy_movie_fname = 'natural_movies/Protector2_frames_3050_to_3140.npy'

m_tmp = np.load(npy_movie_fname)

#NOTE that at this point axis 0 correponds to t, axis 1 to y, and axis 2 to x.
for i in xrange(0, m_tmp.shape[0], 10):
  print i
  plt.imshow(m_tmp[i], cmap=plt.cm.Greys_r); plt.show()


