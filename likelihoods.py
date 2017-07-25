import MCMC_abnew_log as MCMC
import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import quad
import matplotlib.pyplot as plt
import time

t0 = time.time()
theta_pass = MCMC.run_scipy()

samples = MCMC.read_pickled_samples()
print np.shape(samples)

hist, edges = np.histogram(samples[:,0],bins=300,density=True)
func = interp1d(edges[:-1],hist)
print np.trapz(edges[:-1],hist)

print quad(func,-12.78,-11.8,maxp1=100)

integrated_prob = 0.
lower_limit = theta_pass[0]
while integrated_prob < 0.34:
	lower_limit -= 0.1
	integrated_prob = quad(func,lower_limit,theta_pass[0],maxp1=100)[0]

print lower_limit
print quad(func,theta_pass[0],-11.8,maxp1=100)[0]


plt.figure()
plt.plot(edges[:-1],hist)
plt.plot(edges[:-1],func(edges[:-1]),'.')
plt.axvline(theta_pass[0],c='k')

plt.figure()
plt.hist(samples[:,0],bins=300,cumulative=True,normed=True)
plt.axvline(theta_pass[0],c='k')

plt.figure()
plt.hist(samples[:,0],bins=300,normed=True)
plt.axvline(theta_pass[0],c='k')
print "Done in", time.time() - t0, "seconds"

plt.show()