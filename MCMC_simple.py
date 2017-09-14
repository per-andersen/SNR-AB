import utility_functions as util
import os
import numpy as np
import emcee
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from chainconsumer import ChainConsumer
import pickle as pick
import time

def simple_snr(logssfr,theta):
	a, b = theta
	ssfr = 10**logssfr
	return a + b * ssfr

def lnprior(theta):
	a, c = theta
	#theta_pass = 4.2e-14, 0.272, 3.8e-11
	if (1e-15 < a < 1e-13) and (1e-6 < c < 5e-2):
		return 0.
	return -np.inf

def lnlike(theta, ssfr, snr, snr_err):
	logssfr = np.log10(ssfr)
	a, c = theta

	snr_model = simple_snr(logssfr, theta)
	return -0.5*(np.sum( ((snr-snr_model)/snr_err)**2.  ))

def lnprob(theta, ssfr, snr, snr_err):
	lp = lnprior(theta)
	if not np.isfinite(lp):
		return -np.inf
	return lp + lnlike(theta, ssfr, snr, snr_err)

def run_grid():
	resolution = 300

	#ML parameters: [  1.20103392 -13.38221532 -10.07802213   2.61730236]

	a_min, a_max = 2e-14, 8e-14
	b_min, b_max = 0.0002, 0.0008

	# Reading in data
	logssfr, ssfr,  snr, snr_err = util.read_data_with_log()

	likelihoods = np.ones((resolution,resolution))

	a_par = np.linspace(a_min,a_max,resolution)
	b_par = np.linspace(b_min,b_max,resolution)

	max_like = 0.

	for ii in np.arange(resolution):
		for jj in np.arange(resolution):
			theta = a_par[ii], b_par[jj]
			likelihoods[ii,jj] = np.exp(lnlike(theta,ssfr,snr,snr_err))
			if likelihoods[ii,jj] > max_like:
				max_like = likelihoods[ii,jj]
				theta_max = a_par[ii], b_par[jj]
	
	#plt.figure()
	#plt.xlabel("a")
	#plt.ylabel("b")
	#im = plt.imshow(likelihoods[:,:],interpolation='none',origin='lower',cmap=cm.Greys,extent=(a_min, a_max, b_min, b_max))
	#plt.colorbar()

	a_like = np.ones(resolution)
	b_like = np.ones(resolution)
	for ii in np.arange(resolution):
		a_like[ii] = np.sum(likelihoods[ii,:])
		b_like[ii] = np.sum(likelihoods[:,ii])

	yes_chainconsumer = True
	if yes_chainconsumer:
		print "Defining chainconsumer"
		c = ChainConsumer()
		print "Adding chain"
		c.add_chain([a_par, b_par], parameters=["A","B"],weights=likelihoods,grid=True)
		print "Doing plot"
		fig = c.plotter.plot()
	
	a_fit = a_par[np.argmax(a_like)]
	b_fit = b_par[np.argmax(b_like)]

	plt.figure()
	plt.plot(a_par,a_like)
	plt.xlabel('a')

	plt.figure()
	plt.plot(b_par,b_like)
	plt.xlabel('b')

	theta_pass = a_fit, b_fit
	return theta_pass

def run_emcee():
	if util.do_chains_exist(model_name,root_dir):
		print 'Chains already exist, using existing chains...'
		samples = util.read_chains(model_name,root_dir)
		print np.shape(samples)
	else:
		print 'Chains do not exist, computing chains...'
		logssfr, ssfr, snr, snr_err = util.read_data_with_log()

		ndim = 2	
		nwalkers = 700
		pos_min = np.array([1e-15, 5e-2])
		pos_max = np.array([1e-13, 1e-3])
		psize = pos_max - pos_min
		pos = [pos_min + psize*np.random.rand(ndim) for ii in range(nwalkers)]

		sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(ssfr, snr, snr_err), threads=1)
		pos, prob, state = sampler.run_mcmc(pos, 300)
		sampler.reset()

		pos, prob, state = sampler.run_mcmc(pos, 5000)

		samples = sampler.flatchain
		output = open(root_dir + 'Data/MCMC_simple.pkl','wb')
 		pick.dump(samples,output)
 		output.close()
		print np.shape(samples)

	#plt.figure()
	#plt.hist(samples[:,0],bins=300)
	#plt.xlabel('A')

	#plt.figure()
	#plt.hist(samples[:,1],bins=300)
	#plt.xlabel('B')


	c = ChainConsumer()
	c.add_chain(samples, parameters=["$A$", "$B$"])
	#figw = c.plotter.plot_walks()
	fig = c.plotter.plot(figsize=(6.5,5))
	fig.savefig(root_dir + 'Plots/marginals_simple.pdf')
	summary =  c.analysis.get_summary()

	a_fit = summary["$A$"][1]
	c_fit = summary["$B$"][1]

	theta_pass = a_fit, c_fit
	
	print 'A', a_fit
	print 'B', c_fit

	return theta_pass
	
root_dir = '/Users/perandersen/Data/SNR-AB/'
model_name = 'simple'

if __name__ == '__main__':
	t0 = time.time()
	
	logssfr, ssfr, snr, snr_err = util.read_data_with_log()
	
	theta_pass = run_emcee()
	#theta_pass = run_grid()
	chi2 = np.sum( ((snr-simple_snr(logssfr, theta_pass))/snr_err)**2.  )
	bic = chi2 + 2.*np.log(len(logssfr))
	aic = chi2 + 2.*2.
	ks_test = util.ks_test(np.log10(ssfr),snr,simple_snr,theta_pass,visualise=False)

	print "Done in", time.time() - t0, "seconds"
	print ""
	print "BIC", bic
	print "AIC", aic
	print "chi2", chi2
	print "r.chi2", chi2 / (len(logssfr)-2.)
	print "KS", ks_test
	#theta_pass = 5e-14, 3e-4
	util.plot_data(root_dir, model_name, theta_pass, simple_snr)
	
	plt.show()

	#run_grid()

