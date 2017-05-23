import numpy as np
import emcee
import matplotlib.pyplot as plt
from chainconsumer import ChainConsumer
import time

t0 = time.time()
gp13 = np.genfromtxt('GP13.txt')
m05 = np.genfromtxt('M05.txt')
s06 = np.genfromtxt('S06.txt')
sm12 = np.genfromtxt('Sm12.txt') 

sm12 = np.delete(sm12,1, axis=0) #!!! This needs to be justified

logssfr_data = np.concatenate((gp13[:,0],m05[:,0],s06[:,0],sm12[:,0]))
ssfr_data = 10**logssfr_data
snr_data = np.concatenate((gp13[:,1],m05[:,1],s06[:,1],sm12[:,1]))
snr_err_data = np.concatenate((gp13[:,2],m05[:,2],np.ones(len(s06[:,1]))*2e-14,np.ones(len(sm12[:,1]))*2e-14))
#snr_err_data = 1.

def split_snr(logssfr,theta):
	a, k, ssfr0 = theta
	logssfr0 = np.log10(ssfr0)
	ssfr = 10**logssfr

	snr_return = np.zeros(len(logssfr))
	for ii in np.arange(len(logssfr)):
		if ssfr[ii] > ssfr0:
			snr_return[ii] = a + a * np.log(ssfr[ii]/ssfr0) / k
		else:
			snr_return[ii] =  a
	return snr_return

def lnprior(theta):
	a, k, ssfr0 = theta
	if 5e-16 < a < 5e-13 and 0. < k < 2. and 1e-13 < ssfr0 < 1e-9:
		return 0.
	return -np.inf

def lnlike(theta, ssfr, snr, snr_err):
	a, k, ssfr0 = theta
	logssfr = np.log10(ssfr)
	snr_model = split_snr(logssfr, theta)
	return -0.5*(np.sum( ((snr-snr_model)/snr_err)**2.  ))

def lnprob(theta, ssfr, snr, snr_err):
	lp = lnprior(theta)
	if not np.isfinite(lp):
		return -np.inf
	return lp + lnlike(theta, ssfr, snr, snr_err)



ndim = 3
#nwalkers = 500
nwalkers = 500
pos_min = np.array([5e-16, 0., 1e-13])
pos_max = np.array([5e-13, 2., 1e-9])
psize = pos_max - pos_min
pos = [pos_min + psize*np.random.rand(ndim) for ii in range(nwalkers)]

sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(ssfr_data, snr_data, snr_err_data), threads=1)
pos, prob, state = sampler.run_mcmc(pos, 200)
sampler.reset()

pos, prob, state = sampler.run_mcmc(pos, 20000)

samples = sampler.flatchain
print np.shape(samples)

#plt.figure()
#plt.hist(samples[:,0], bins=20)
#plt.figure()
#plt.hist(samples[:,1], bins=20)
#plt.figure()
#plt.hist(samples[:,2], bins=20)


c = ChainConsumer()
c.add_chain(samples, parameters=["$A$", "$k$", "$sSFR_0$"])
#fig = c.plot(figsize="column")
fig = c.plot(figsize=(8.5,7))
#fig.set_size_inches(14.5 + fig.get_size_inches())
print "Done in", time.time() - t0, "seconds"

plt.show()



logssfr_values = np.linspace(-13,-8,100)
snr_values_1 = np.zeros(len(logssfr_values))
snr_values_2 = np.zeros(len(logssfr_values))
snr_values_1 = split_snr(logssfr_values,[4.18e-14, 0.272,3.82e-11])
snr_values_2 = split_snr(logssfr_values,[4.0e-14, 0.266,3.11e-11])
ax = plt.subplot()
ax.set_yscale("log")
plt.xlim([-13,-8])
#plt.ylim([-13,-8])
plt.errorbar(gp13[:,0],gp13[:,1],yerr=[gp13[:,2],gp13[:,3]])
plt.errorbar(m05[:,0],m05[:,1],yerr=[m05[:,2],m05[:,3]])
#plt.errorbar(s06[:,0],s06[:,1],yerr=2e-14)
plt.errorbar(sm12[:,0],sm12[:,1],yerr=2e-14)
plt.plot(logssfr_values, snr_values_1,c='k',lw=3)
plt.plot(logssfr_values, snr_values_2,c='k',lw=3,ls='--')
plt.show()
