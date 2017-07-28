import utility_functions as util
import os
import numpy as np
import emcee
import matplotlib.pyplot as plt
from chainconsumer import ChainConsumer
import pickle as pick
import time

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
	a, c, d = theta
	#theta_pass = 4.2e-14, 0.272, 3.8e-11
	if (5e-16 < a < 5e-13) and (0.001 < c < 1.5) and (5e-13 < d < 5e-10):
		return 0.
	return -np.inf

def lnlike(theta, ssfr, snr, snr_err):
	logssfr = np.log10(ssfr)
	a, c, d = theta

	snr_model = split_snr(logssfr, theta)
	return -0.5*(np.sum( ((snr-snr_model)/snr_err)**2.  ))

def lnprob(theta, ssfr, snr, snr_err):
	lp = lnprior(theta)
	if not np.isfinite(lp):
		return -np.inf
	return lp + lnlike(theta, ssfr, snr, snr_err)

if __name__ == '__main__':
	t0 = time.time()

	root_dir = '/Users/perandersen/Data/SNR-AB/'
	model_name = 'split'
	
	if os.path.isfile(root_dir + 'Data/MCMC_split.pkl'):
		print 'Chains already exist, using existing chains...'
		pkl_data_file = open(root_dir + 'Data/MCMC_split.pkl','rb')
		samples = pick.load(pkl_data_file)
		pkl_data_file.close()
		print np.shape(samples)
	else:
		print 'Chains do not exist, computing chains...'
		logssfr, ssfr, snr, snr_err = util.read_data_with_log()

		ndim = 3	
		nwalkers = 500
		pos_min = np.array([5e-16, 0.001, 5e-13])
		pos_max = np.array([5e-13, 1.5, 5e-10])
		psize = pos_max - pos_min
		pos = [pos_min + psize*np.random.rand(ndim) for ii in range(nwalkers)]

		sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(ssfr, snr, snr_err), threads=1)
		pos, prob, state = sampler.run_mcmc(pos, 2000)
		sampler.reset()

		pos, prob, state = sampler.run_mcmc(pos, 10000)

		samples = sampler.flatchain
		output = open(root_dir + 'Data/MCMC_split.pkl','wb')
 		pick.dump(samples,output)
 		output.close()
		print np.shape(samples)

	#plt.figure()
	#plt.hist(samples[:,0],bins=300)
	#plt.xlabel('A')

	#plt.figure()
	#plt.hist(samples[:,1],bins=300)
	#plt.xlabel('k')

	#plt.figure()
	#plt.hist(samples[:,2],bins=300)
	#plt.xlabel('sSFR_0')


	c = ChainConsumer()
	c.add_chain(samples, parameters=["$A$", "$k$", "$sSFR_0$"])
	#figw = c.plotter.plot_walks()
	fig = c.plotter.plot(figsize=(8,6))
	fig.savefig(root_dir + 'Plots/marginals_split.pdf')
	summary =  c.analysis.get_summary()

	a_fit = summary["$A$"][1]
	c_fit = summary["$k$"][1]
	d_fit = summary["$sSFR_0$"][1]
	
	print 'A', a_fit
	print 'k', c_fit
	print 'sSFR_0', d_fit
	

	logssfr, ssfr, snr, snr_err = util.read_data_with_log()
	theta_pass = a_fit, c_fit, d_fit
	#theta_pass = 4.2e-14, 0.272, 3.8e-11
	chi2 = np.sum( ((snr-split_snr(logssfr, theta_pass))/snr_err)**2.  )
	bic = chi2 + 3.*np.log(len(logssfr))
	aic = chi2 + 3.*2.

	print "Done in", time.time() - t0, "seconds"
	print ""
	print "BIC", bic
	print "AIC", aic
	print "chi2", chi2
	print "r.chi2", chi2 / (len(logssfr)-3.)

	util.plot_data(root_dir, model_name, theta_pass, split_snr)

	plt.show()

