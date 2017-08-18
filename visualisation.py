import utility_functions as util
import numpy as np
import matplotlib.pyplot as plt
import MCMC_nicelog as nice
import MCMC_simple as simple

root_dir = '/Users/perandersen/Data/SNR-AB/'

theta_nice = nice.run_grid()
theta_simple = simple.run_emcee()
util.plot_combined(root_dir, ['nicelog','simple'], [theta_nice,theta_simple], [nice.nicelog_snr,simple.simple_snr], [False,False])
plt.show()