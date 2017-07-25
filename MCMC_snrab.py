import numpy as np
import emcee
import matplotlib.pyplot as plt
from chainconsumer import ChainConsumer
import time

t0 = time.time()
gp13 = np.genfromtxt('Teddy/GP13.txt')
m05 = np.genfromtxt('Teddy/M05.txt')
s06 = np.genfromtxt('Teddy/S06.txt')
sm12 = np.genfromtxt('Teddy/Sm12.txt') 

sm12 = np.delete(sm12,1, axis=0) #!!! This needs to be justified

logssfr_data = np.concatenate((gp13[:,0],m05[:,0],s06[:,0],sm12[:,0]))
snr_data = np.concatenate((gp13[:,1],m05[:,1],s06[:,1],sm12[:,1]))
snr_err_data = np.concatenate((gp13[:,2],m05[:,2],np.ones(len(s06[:,1]))*5e-14,np.ones(len(sm12[:,1]))*5e-14))

#logssfr_data = np.concatenate((gp13[:,0],m05[:,0],s06[:,0]))
#snr_data = np.concatenate((gp13[:,1],m05[:,1],s06[:,1]))
#snr_err_data = np.concatenate((gp13[:,2],m05[:,2],np.ones(len(s06[:,1]))*2e-14))

ssfr_data = 10**logssfr_data

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


def simple_snr(logssfr,theta):
	a, b = theta
	ssfr = 10**logssfr
	return a + b * ssfr

def new_snr(logssfr,theta):
	k1, k2, ssfr1, ssfr2, ssfra = theta
	ssfr = 10**logssfr

	snr_return = np.zeros(len(ssfr))
	
	for ii in np.arange(len(ssfr)):
		
		if ssfr1 < ssfr[ii]:
			snr_return[ii] = k1
		
		elif ssfr[ii] < ssfr2:
			snr_return[ii] = k2 * ssfra
		
		else:
			snr_return[ii] = ssfr[ii] * (k1 / ssfr1 + k2*np.log(ssfr1 / ssfr2))

	return snr_return

def lnprior(theta):
	if len(theta) == 3:
		a, k, ssfr0 = theta
		if 5e-16 < a < 5e-13 and 0. < k < 2. and 1e-13 < ssfr0 < 1e-9:
			return 0.
		return -np.inf

	elif len(theta) == 2:
		a, b = theta
		if 1e-16 < a < 1e-12 and 1e-5 < b < 1e-2:
			return 0.
		return -np.inf

	elif len(theta) == 5:
		k1, k2, ssfr1, ssfr2, ssfra = theta
		if 1e-14 < k1 < 1e-12 and 1e-14 < k2 < 1e-10 and 1e-12 < ssfr1 < 1e-9 and 1e-13 < ssfr2 < 1e-10 and 1e-6 < ssfra < 1e-2:
			return 0.
		return -np.inf

def lnlike(theta, ssfr, snr, snr_err):
	logssfr = np.log10(ssfr)
	if len(theta) == 3:
		a, k, ssfr0 = theta
		snr_model = split_snr(logssfr, theta)
		return -0.5*(np.sum( ((snr-snr_model)/snr_err)**2.  ))
	if len(theta) == 2:
		a, b = theta
		snr_model = simple_snr(logssfr, theta)
		return -0.5*(np.sum( ((snr-snr_model)/snr_err)**2.  ))
	if len(theta) == 5:
		k1, k2, ssfr1, ssfr2, ssfra = theta
		snr_model = new_snr(logssfr, theta)
		return -0.5*(np.sum( ((snr-snr_model)/snr_err)**2.  ))

def lnprob(theta, ssfr, snr, snr_err):
	lp = lnprior(theta)
	if not np.isfinite(lp):
		return -np.inf
	return lp + lnlike(theta, ssfr, snr, snr_err)

def run_plot_mcmc_new():
	ndim = 5
	#nwalkers = 500
	nwalkers = 500
	#k1, k2, ssfr1, ssfr2, ssfra = theta
	#5e-13, 1e-11, 5e-10, 5e-11, 4e-3
	pos_min = np.array([1e-14, 1e-12, 1e-11, 1e-12, 1e-4])
	pos_max = np.array([1e-12, 1e-10, 1e-9, 1e-10, 1e-2])
	psize = pos_max - pos_min
	pos = [pos_min + psize*np.random.rand(ndim) for ii in range(nwalkers)]

	sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(ssfr_data, snr_data, snr_err_data), threads=1)
	pos, prob, state = sampler.run_mcmc(pos, 700)
	sampler.reset()

	pos, prob, state = sampler.run_mcmc(pos, 6000)

	samples = sampler.flatchain
	print np.shape(samples)
	c = ChainConsumer()
	c.add_chain(samples, parameters=["$k_1$", "$k_2$", "$sSFR_1$", "$sSFR_2$", "$sSFR_a$"])
	#fig = c.plot(figsize="column")
	fig = c.plotter.plot(figsize=(8.5,7))
	#fig.set_size_inches(14.5 + fig.get_size_inches())
	summary =  c.analysis.get_summary()
	print summary["$k_1$"][1]
	print summary["$k_2$"][1]
	print summary["$sSFR_1$"][1]
	print summary["$sSFR_2$"][1]
	print summary["$sSFR_a$"][1]
	print "Done in", time.time() - t0, "seconds"

	plt.show()

	'''
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
	'''

def run_plot_mcmc_split():
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

	pos, prob, state = sampler.run_mcmc(pos, 1000)

	samples = sampler.flatchain
	print np.shape(samples)
	c = ChainConsumer()
	c.add_chain(samples, parameters=["$A$", "$k$", "$sSFR_0$"])
	#fig = c.plot(figsize="column")
	fig = c.plotter.plot(figsize=(8.5,7))
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

def run_plot_mcmc_simple():
	ndim = 2
	#nwalkers = 500
	nwalkers = 500
	pos_min = np.array([1e-16, 1e-5])
	pos_max = np.array([1e-12, 1e-2])
	psize = pos_max - pos_min
	pos = [pos_min + psize*np.random.rand(ndim) for ii in range(nwalkers)]

	sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(ssfr_data, snr_data, snr_err_data), threads=1)
	pos, prob, state = sampler.run_mcmc(pos, 200)
	sampler.reset()

	pos, prob, state = sampler.run_mcmc(pos, 1000)

	samples = sampler.flatchain
	print np.shape(samples)
	c = ChainConsumer()
	c.add_chain(samples, parameters=["$A$", "$B$"])
	#fig = c.plot(figsize="column")
	fig = c.plotter.plot(figsize=(8.5,7))
	#fig.set_size_inches(14.5 + fig.get_size_inches())
	print "Done in", time.time() - t0, "seconds"

	plt.figure()

	logssfr_values = np.linspace(-13,-8,100)
	snr_values_1 = split_snr(logssfr_values,[4.18e-14, 0.272,3.82e-11])
	snr_values_2 = simple_snr(logssfr_values,[91.9e-15, 345.4e-6])
	ax = plt.subplot()
	ax.set_yscale("log")
	plt.xlim([-13,-8])
	#plt.ylim([-13,-8])
	plt.errorbar(gp13[:,0],gp13[:,1],yerr=[gp13[:,2],gp13[:,3]],label='GP13')
	plt.errorbar(m05[:,0],m05[:,1],yerr=[m05[:,2],m05[:,3]],label='M05')
	plt.errorbar(s06[:,0],s06[:,1],yerr=2e-14,label='S06')
	plt.errorbar(sm12[:,0],sm12[:,1],yerr=2e-14,label='SM12')
	plt.plot(logssfr_values, snr_values_1,c='k',lw=3)
	plt.plot(logssfr_values, snr_values_2,c='k',lw=3,ls='--')
	plt.legend()
	plt.show()

def plot_new():
	
	#k1, k2, ssfr1, ssfr2, ssfra = theta
	theta = 5e-13, 1e-11, 5e-10, 4e-11, 4e-3
	logssfr_values = np.linspace(-13,-8,100000)
	snr_values = new_snr(logssfr_values, theta)
	ax = plt.subplot()
	ax.set_yscale("log")
	plt.xlim([-13,-8])
	#plt.ylim([-13,-8])
	plt.errorbar(gp13[:,0],gp13[:,1],yerr=[gp13[:,2],gp13[:,3]])
	plt.errorbar(m05[:,0],m05[:,1],yerr=[m05[:,2],m05[:,3]])
	plt.errorbar(s06[:,0],s06[:,1],yerr=2e-14)
	plt.errorbar(sm12[:,0],sm12[:,1],yerr=2e-14)
	plt.plot(logssfr_values, snr_values,c='k',lw=3)
	#plt.plot(logssfr_values, snr_values,'ko')
	plt.show()

run_plot_mcmc_simple()
#run_plot_mcmc_split()
#run_plot_mcmc_new()
#plot_new()