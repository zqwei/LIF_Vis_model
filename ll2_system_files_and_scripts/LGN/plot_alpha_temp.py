import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import factorial

#An alpha-shaped temporal filter. Flag=True builds filter for Transient cells
def alpha_temp_fil(dt, t_max, k_alpha, beta, n_filter):
    t_vec = np.arange(0,t_max,dt)
    f_t = (k_alpha * t_vec) ** n_filter * np.exp(-k_alpha * t_vec) * (1 / factorial(n_filter) - beta * ((k_alpha * t_vec) ** 2) / factorial(n_filter + 2))
    return f_t, t_vec


def alpha_temp_shifted(dt, t_max, t_shift, k_alpha, beta, n_filter):
    t_vec = np.arange(0,t_max,dt)
    t_vec_shifted = t_vec + t_shift
    f_t = (k_alpha * t_vec_shifted) ** n_filter * np.exp(-k_alpha * t_vec_shifted) * (1 / factorial(n_filter) - beta * ((k_alpha * t_vec_shifted) ** 2) / factorial(n_filter + 2))
    return f_t, t_vec




dt = 1.0
t_max = 500.0
beta = 1.0

c_list = ['k', 'r', 'b']

for i, k_alpha in enumerate([0.035, 0.05]): #0.038, 0.103
    for j, n_filter in enumerate([1, 2, 8]): #2, 8

        f, t = alpha_temp_fil(dt, t_max, k_alpha, beta, n_filter)

        #if ((i == 0) and (j == 0)):
        #    f = 0.3 * f

        plt.plot(t, f, c=c_list[i], label=('k_alpha = %g, n_filter = %d' % (k_alpha, n_filter)))

t_shift = 100.0
k_alpha = 0.038
n_filter = 8
#f, t = alpha_temp_shifted(dt, t_max, t_shift, k_alpha, beta, n_filter)
#plt.plot(t, f, label='alpha_temp_shifted')

plt.plot(t, 0.0*t)
plt.legend()
plt.show()

