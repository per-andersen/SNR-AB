import utility_functions as util
import numpy as np
import matplotlib.pyplot as plt
import MCMC_nicelog as nice
import MCMC_simple as simple

root_dir = '/Users/perandersen/Data/SNR-AB/'

theta_nice = nice.run_grid()
theta_simple = simple.
util.plot_combined(root_dir, ['nicelog'], [theta_nice], [nice.nicelog_snr], [False])
plt.show()