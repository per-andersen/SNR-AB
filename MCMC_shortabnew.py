import utility_functions as util
import os
import numpy as np
import emcee
import matplotlib.pyplot as plt
from chainconsumer import ChainConsumer
import scipy.optimize as opt
import pickle as pick
import time

def new_snr(ssfr,theta):
	k2, ssfr1, ssfr2, ssfra = theta

	snr_return = np.zeros(len(ssfr))
	
	for ii in np.arange(len(ssfr)):
		
		if ssfr[ii] < ssfr2:
			snr_return[ii] = k2 * ssfra
		
		else:
			snr_return[ii] = ssfr[ii] * k2*np.log(ssfr1 / ssfr2)

	return snr_return

def lnlike(theta, ssfr, snr, snr_err):
	k2, ssfr1, ssfr2, ssfra = theta

	snr_model = new_snr(ssfr, theta)
	return -0.5*(np.sum( ((snr-snr_model)/snr_err)**2.  ))

def run_emcee():
	if os.path.isfile(root_dir + 'Data/MCMC_shortabnew.pkl'):
		print 'Chains already exist, using existing chains'
		pkl_data_file = open(root_dir + 'Data/MCMC_shortabnew.pkl','rb')
		samples = pick.load(pkl_data_file)
		pkl_data_file.close()
		print np.shape(samples)
	else:
		print 'Chains do not exist, computing chains...'

		# Setting parameter top hat priors
		#4.628e-13, 6.105e-11, 2.885e-10, 1.008e-11, 6.100e-4
		k2_min, k2_max = 1e-13, 1e-9
		ssfr1_min, ssfr1_max = 1e-11, 1e-7
		ssfr2_min, ssfr2_max = 1e-13, 1e-9
		ssfra_min, ssfra_max = 1e-7, 1e-1

		# These priors force the chain to only check this local extrema
		#k1_min, k1_max = 1e-13, 9e-13
		#k2_min, k2_max = 1e-5, 7e-4
		#ssfr1_min, ssfr1_max = 1e-10, 8e-10
		#ssfr2_min, ssfr2_max = 1e-11, 8e-11
		#ssfra_min, ssfra_max = 2e-11, 9e-10

		ndim = 4	
		nwalkers = 300
		nburn = 20
		nsample = 150

		# These functions define the prior and the function to apply prior to likelihood
		def lnprior(theta):
			k2, ssfr1, ssfr2, ssfra = theta
			
			if (k2_min < k2 < k2_max) and (ssfr1_min < ssfr1 < ssfr1_max) and (ssfr2_min < ssfr2 < ssfr2_max) and (ssfra_min < ssfra < ssfra_max):
				
				slope_term = k1 / ssfr1 + k2*np.log(ssfr1 / ssfr2)
				if not np.isclose(k1,ssfr1*slope_term):
					return -np.inf
				if not np.isclose(k2*ssfra,ssfr2*slope_term):
					return -np.inf

				return 0.
			return -np.inf

		def lnprob(theta, logssfr, logsnr, snr_err):
			lp = lnprior(theta)
			if not np.isfinite(lp):
				return -np.inf
			return lp + lnlike(theta, logssfr, logsnr, snr_err)

		# Reading in data
		ssfr, snr, snr_err = util.read_data()

		# Setting initial position of walkers
		pos_min = np.array([k2_min, ssfr1_min, ssfr2_min, ssfra_min])
		pos_max = np.array([k2_max, ssfr1_max, ssfr2_max, ssfra_max])
		psize = pos_max - pos_min

		psize = pos_max - pos_min
		pos = [pos_min + psize*np.random.rand(ndim) for ii in range(nwalkers)]
		#pos = [np.array([4.628e-13, 6.105e-11, 2.885e-10, 1.008e-11, 6.100e-4]) + 1e-4*psize*np.random.randn(ndim) for i in range(nwalkers)]

		# Defining sampler
		sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(ssfr, snr, snr_err), threads=1)

		# Performing burn in
		pos, prob, state = sampler.run_mcmc(pos, nburn)
		sampler.reset()

		# Main sampling run
		pos, prob, state = sampler.run_mcmc(pos, nsample)

		# These plots are for diagnostics use
		plt.figure()
		ax = plt.subplot()
		ax.set_yscale("log")
		#ax.set_xscale("log")
		plt.plot(sampler.chain[:,:,0].T,'b',alpha=0.05)
		plt.xlabel('par0')
		
		plt.figure()
		ax = plt.subplot()
		ax.set_yscale("log")
		#ax.set_xscale("log")
		plt.plot(sampler.chain[:,:,3].T,'b',alpha=0.05)
		plt.xlabel('par3')

		plt.figure()
		ax = plt.subplot()
		#ax.set_yscale("log")
		#ax.set_xscale("log")
		plt.plot(sampler.lnprobability[:,:].T,'b',alpha=0.05)
		plt.xlabel('lnprob')

		# Formatting and saving output
		samples = sampler.flatchain
		output = open(root_dir + 'Data/MCMC_shortabnew.pkl','wb')
 		pick.dump(samples,output)
 		output.close()
		print np.shape(samples)

	plt.figure()
	ax = plt.subplot()
	ax.set_xscale("log")
	plt.hist(samples[:,0],bins=100)
	plt.xlabel('k2')

	plt.figure()
	ax = plt.subplot()
	ax.set_xscale("log")
	plt.hist(samples[:,1],bins=100)
	plt.xlabel('ssfr1')

	plt.figure()
	ax = plt.subplot()
	ax.set_xscale("log")
	plt.hist(samples[:,2],bins=100)
	plt.xlabel('ssfr2')

	plt.figure()
	ax = plt.subplot()
	ax.set_xscale("log")
	plt.hist(samples[:,3],bins=100)
	plt.xlabel('ssfra')

	c = ChainConsumer()
	c.add_chain(samples, parameters=["$k_2$", "$sSFR_1$", "$sSFR_2$", "$sSFR_a$"])
	c.configure(smooth=False,bins=300,sigmas=[0,1,2,3])
	#figw = c.plotter.plot_walks()
	fig = c.plotter.plot()
	summary =  c.analysis.get_summary()

	k2_fit = summary["$k_2$"][1]
	ssfr1_fit = summary["$sSFR_1$"][1]
	ssfr2_fit = summary["$sSFR_2$"][1]
	ssfra_fit = summary["$sSFR_a$"][1]

	theta_pass = k2_fit, ssfr1_fit, ssfr2_fit, ssfra_fit
	
	print 'k2', k2_fit
	print 'ssfr1', ssfr1_fit
	print 'ssfr2', ssfr2_fit
	print 'ssfra', ssfra_fit
	return theta_pass

def run_grid():
	if os.path.isfile(root_dir + 'Data/MCMC_shortabnew_grid.pkl'):
		print 'Grid already exists, using existing grid...'
		pkl_data_file = open(root_dir + 'Data/MCMC_shortabnew_grid.pkl','rb')
		resolution, likelihoods, k2_par, s1_par, s2_par, sa_par, theta_max = pick.load(pkl_data_file)
		pkl_data_file.close()
	else:
		print 'Grid does not exist, computing grid...'
	
		resolution = 100

		#theta_pass = 4.628e-13, 2e-4, 1./(0.5e9), 1.008e-11, 2./(12e9)
		k2_min, k2_max = 4e-5, 1.2e-4
		ssfr1_min, ssfr1_max = 1e-8, 5e-7
		ssfr2_min, ssfr2_max = 7e-12, 2.e-10
		ssfra_min, ssfra_max = 2e-10, 1.e-9

		# Reading in data
		ssfr, snr, snr_err = util.read_data()

		k2_par = np.linspace(k2_min,k2_max,resolution)
		s1_par = np.linspace(ssfr1_min,ssfr1_max,resolution)
		s2_par = np.linspace(ssfr2_min,ssfr2_max,resolution)
		sa_par = np.linspace(ssfra_min,ssfra_max,resolution)

		likelihoods = np.ones((resolution,resolution,resolution,resolution))
		max_like = 0.

		for ii in np.arange(resolution):
			if ii%2 == 0:
				print np.round((float(ii) / resolution) * 100.,2), "% Done"
			for jj in np.arange(resolution):
				for kk in np.arange(resolution):
					for ll in np.arange(resolution):
						theta = k2_par[ii], s1_par[jj], s2_par[kk], sa_par[ll]
						likelihoods[ii,jj,kk,ll] = np.exp(lnlike(theta,ssfr,snr,snr_err))
						if likelihoods[ii,jj,kk,ll] > max_like:
							max_like = likelihoods[ii,jj,kk,ll]
							theta_max = k2_par[ii], s1_par[jj], s2_par[kk], sa_par[ll]
							#print "New max like:", max_like
							#print theta_max, "\n"
		likelihoods /= np.sum(likelihoods)
		output = open(root_dir + 'Data/MCMC_shortabnew_grid.pkl','wb')
		result = resolution, likelihoods, k2_par, s1_par, s2_par, sa_par, theta_max
 		pick.dump(result,output)
 		output.close()

	k2_like = np.zeros(resolution)
	s1_like = np.zeros(resolution)
	s2_like = np.zeros(resolution)
	sa_like = np.zeros(resolution)
	for ii in np.arange(resolution):
		k2_like[ii] = np.sum(likelihoods[ii,:,:,:])
		s1_like[ii] = np.sum(likelihoods[:,ii,:,:])
		s2_like[ii] = np.sum(likelihoods[:,:,ii,:])
		sa_like[ii] = np.sum(likelihoods[:,:,:,ii])
	
	'''
	plt.figure()
	ax = plt.subplot()
	ax.set_xscale("log")
	plt.plot(k2_par,k2_like,'x')
	plt.xlabel('k2')

	plt.figure()
	ax = plt.subplot()
	ax.set_xscale("log")
	plt.plot(s1_par,s1_like,'x')
	plt.xlabel('ssfr1')

	plt.figure()
	ax = plt.subplot()
	ax.set_xscale("log")
	plt.plot(s2_par,s2_like,'x')
	plt.xlabel('ssfr2')

	plt.figure()
	ax = plt.subplot()
	ax.set_xscale("log")
	plt.plot(sa_par,sa_like,'x')
	plt.xlabel('ssfra')
	'''

	# These are the marginalised maximum likelihood parameters
	k2_fit = k2_par[np.argmax(k2_like)]
	s1_fit = s1_par[np.argmax(s1_like)]
	s2_fit = s2_par[np.argmax(s2_like)]
	sa_fit = sa_par[np.argmax(sa_like)]

	print "ML parameters:"
	#theta_pass = k2_fit, s1_fit, s2_fit, sa_fit
	theta_pass = theta_max
	print theta_pass
	return theta_pass

if __name__ == '__main__':
	t0 = time.time()

	root_dir = '/Users/perandersen/Data/SNR-AB/'
	model_name = 'shortabnew'

	#theta_pass = run_emcee()
	theta_pass = run_grid()
	#theta_pass = 8.7179e-05, 5.92e-01, 1.23846e-11, 5.076923e-10

	ssfr, snr, snr_err = util.read_data()
	chi2 = np.sum( ((snr-new_snr(ssfr, theta_pass))/snr_err)**2.  )
	bic = chi2 + 4.*np.log(len(ssfr))
	aic = chi2 + 4.*2.

	print "Done in", time.time() - t0, "seconds"
	print ""
	print "BIC", bic
	print "AIC", aic
	print "chi2", chi2
	print "r.chi2", chi2 / (len(ssfr)-4.)
	
	util.plot_data_log(root_dir,model_name,theta_pass,new_snr)

	plt.show()

