import utility_functions as util
import os
import numpy as np
import emcee
import matplotlib.pyplot as plt
from chainconsumer import ChainConsumer
import scipy.optimize as opt
import pickle as pick
import time

def new_contin_snr(ssfr,theta):
	k1, k2, ssfr1, ssfr2 = theta

	snr_return = np.zeros(len(ssfr))
	
	for ii in np.arange(len(ssfr)):
		
		if ssfr1 < ssfr[ii]:
			snr_return[ii] = ssfr1 * (k1 / ssfr1 + k2*np.log(ssfr1 / ssfr2))
		
		elif ssfr[ii] < ssfr2:
			snr_return[ii] = ssfr2 * (k1 / ssfr1 + k2*np.log(ssfr1 / ssfr2))
		
		else:
			snr_return[ii] = ssfr[ii] * (k1 / ssfr1 + k2*np.log(ssfr1 / ssfr2))

	return snr_return

def lnlike(theta, ssfr, snr, snr_err):
	k1, k2, ssfr1, ssfr2 = theta

	snr_model = new_contin_snr(ssfr, theta)
	return -0.5*(np.sum( ((snr-snr_model)/snr_err)**2.  ))

def run_emcee():
	if os.path.isfile(root_dir + 'Data/MCMC_abnew.pkl'):
		print 'Chains already exist, using existing chains'
		pkl_data_file = open(root_dir + 'Data/MCMC_abnew.pkl','rb')
		samples = pick.load(pkl_data_file)
		pkl_data_file.close()
		print np.shape(samples)
	else:
		print 'Chains do not exist, computing chains...'

		# Setting parameter top hat priors
		#4.628e-13, 6.105e-11, 2.885e-10, 1.008e-11, 6.100e-4
		#k1_min, k1_max = 1e-14, 5e-11
		#k2_min, k2_max = 1e-13, 1e-9
		#ssfr1_min, ssfr1_max = 1e-11, 1e-7
		#ssfr2_min, ssfr2_max = 1e-13, 1e-9
		#ssfra_min, ssfra_max = 1e-7, 1e-1

		# These priors force the chain to only check this local extrema
		k1_min, k1_max = 1e-13, 9e-13
		k2_min, k2_max = 1e-5, 7e-4
		ssfr1_min, ssfr1_max = 1e-10, 8e-10
		ssfr2_min, ssfr2_max = 1e-11, 8e-11
		ssfra_min, ssfra_max = 2e-11, 9e-10

		ndim = 5	
		nwalkers = 300
		nburn = 20
		nsample = 150

		# These functions define the prior and the function to apply prior to likelihood
		def lnprior(theta):
			k1, k2, ssfr1, ssfr2, ssfra = theta
			
			if (k1_min < k1 < k1_max) and (k2_min < k2 < k2_max) and (ssfr1_min < ssfr1 < ssfr1_max) and (ssfr2_min < ssfr2 < ssfr2_max) and (ssfra_min < ssfra < ssfra_max):
				
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
		pos_min = np.array([k1_min, k2_min, ssfr1_min, ssfr2_min, ssfra_min])
		pos_max = np.array([k1_max, k2_max, ssfr1_max, ssfr2_max, ssfra_max])
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
		ax.set_yscale("log")
		#ax.set_xscale("log")
		plt.plot(sampler.chain[:,:,4].T,'b',alpha=0.05)
		plt.xlabel('par4')

		plt.figure()
		ax = plt.subplot()
		#ax.set_yscale("log")
		#ax.set_xscale("log")
		plt.plot(sampler.lnprobability[:,:].T,'b',alpha=0.05)
		plt.xlabel('lnprob')

		# Formatting and saving output
		samples = sampler.flatchain
		output = open(root_dir + 'Data/MCMC_abnew.pkl','wb')
 		pick.dump(samples,output)
 		output.close()
		print np.shape(samples)

	plt.figure()
	ax = plt.subplot()
	ax.set_xscale("log")
	plt.hist(samples[:,0],bins=100)
	plt.xlabel('k1')

	plt.figure()
	ax = plt.subplot()
	ax.set_xscale("log")
	plt.hist(samples[:,1],bins=100)
	plt.xlabel('k2')

	plt.figure()
	ax = plt.subplot()
	ax.set_xscale("log")
	plt.hist(samples[:,2],bins=100)
	plt.xlabel('ssfr1')

	plt.figure()
	ax = plt.subplot()
	ax.set_xscale("log")
	plt.hist(samples[:,3],bins=100)
	plt.xlabel('ssfr2')

	plt.figure()
	ax = plt.subplot()
	ax.set_xscale("log")
	plt.hist(samples[:,4],bins=100)
	plt.xlabel('ssfra')

	c = ChainConsumer()
	c.add_chain(samples, parameters=["$k_1$", "$k_2$", "$sSFR_1$", "$sSFR_2$", "$sSFR_a$"])
	c.configure(smooth=False,bins=300,sigmas=[0,1,2,3])
	#figw = c.plotter.plot_walks()
	fig = c.plotter.plot()
	summary =  c.analysis.get_summary()

	k1_fit = summary["$k_1$"][1]
	k2_fit = summary["$k_2$"][1]
	ssfr1_fit = summary["$sSFR_1$"][1]
	ssfr2_fit = summary["$sSFR_2$"][1]
	ssfra_fit = summary["$sSFR_a$"][1]

	theta_pass = k1_fit, k2_fit, ssfr1_fit, ssfr2_fit, ssfra_fit
	
	print 'k1', k1_fit
	print 'k2', k2_fit
	print 'ssfr1', ssfr1_fit
	print 'ssfr2', ssfr2_fit
	print 'ssfra', ssfra_fit
	return theta_pass

def run_grid():
	if util.does_grid_exist(model_name,root_dir):
		print 'Grid already exists, using existing grid...'
		resolution, likelihoods, parameters, theta_max = util.read_grid(model_name,root_dir)
		k1_par, k2_par, s1_par, s2_par = parameters
	else:
		print 'Grid does not exist, computing grid...'
	
		resolution = 100

		#theta_pass = 4.628e-13, 2e-4, 1./(0.5e9), 1.008e-11, 2./(12e9)
		k1_min, k1_max = 3e-13, 6e-13
		k2_min, k2_max = 3e-5, 4e-4
		ssfr1_min, ssfr1_max = 2e-10, 6e-10
		ssfr2_min, ssfr2_max = 1.5e-11, 5.8e-11

		# Reading in data
		ssfr, snr, snr_err = util.read_data()

		k1_par = np.linspace(k1_min,k1_max,resolution)
		k2_par = np.linspace(k2_min,k2_max,resolution)
		s1_par = np.linspace(ssfr1_min,ssfr1_max,resolution)
		s2_par = np.linspace(ssfr2_min,ssfr2_max,resolution)

		likelihoods = np.ones((resolution,resolution,resolution,resolution))
		max_like = 0.

		for ii in np.arange(resolution):
			if ii%2 == 0:
				print np.round((float(ii) / resolution) * 100.,2), "% Done"
			for jj in np.arange(resolution):
				for kk in np.arange(resolution):
					for ll in np.arange(resolution):
						theta = k1_par[ii], k2_par[jj], s1_par[kk], s2_par[ll]
						likelihoods[ii,jj,kk,ll] = np.exp(lnlike(theta,ssfr,snr,snr_err))
						if likelihoods[ii,jj,kk,ll] > max_like:
							max_like = likelihoods[ii,jj,kk,ll]
							theta_max = k1_par[ii], k2_par[jj], s1_par[kk], s2_par[ll]
							#print "New max like:", max_like
							#print theta_max, "\n"
		likelihoods /= np.sum(likelihoods)
		output = open(root_dir + 'Data/MCMC_abnewcontin_grid.pkl','wb')
		parameters = k1_par, k2_par, s1_par, s2_par
		result = resolution, likelihoods, parameters, theta_max
 		pick.dump(result,output)
 		output.close()

	k1_like = np.zeros(resolution)
	k2_like = np.zeros(resolution)
	s1_like = np.zeros(resolution)
	s2_like = np.zeros(resolution)
	for ii in np.arange(resolution):
		k1_like[ii] = np.sum(likelihoods[ii,:,:,:])
		k2_like[ii] = np.sum(likelihoods[:,ii,:,:])
		s1_like[ii] = np.sum(likelihoods[:,:,ii,:])
		s2_like[ii] = np.sum(likelihoods[:,:,:,ii])
	
	'''
	plt.figure()
	ax = plt.subplot()
	ax.set_xscale("log")
	plt.plot(k1_par,k1_like,'x')
	plt.xlabel('k1')

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
	'''

	# These are the marginalised maximum likelihood parameters
	k1_fit = k1_par[np.argmax(k1_like)]
	k2_fit = k2_par[np.argmax(k2_like)]
	s1_fit = s1_par[np.argmax(s1_like)]
	s2_fit = s2_par[np.argmax(s2_like)]

	print "ML parameters:"
	#theta_pass = k1_fit, k2_fit, s1_fit, s2_fit
	theta_pass = theta_max
	print theta_pass
	return theta_pass

root_dir = '/Users/perandersen/Data/SNR-AB/'
model_name = 'abnewcontin'

if __name__ == '__main__':
	t0 = time.time()


	#theta_pass = run_emcee()
	theta_pass = run_grid()


	ssfr, snr, snr_err = util.read_data()
	chi2 = np.sum( ((snr-new_contin_snr(ssfr, theta_pass))/snr_err)**2.  )
	bic = chi2 + 5.*np.log(len(ssfr))
	aic = chi2 + 5.*2.
	ks_test = util.ks_test(ssfr,snr,new_contin_snr,theta_pass)

	print "Done in", time.time() - t0, "seconds"
	print ""
	print "BIC", bic
	print "AIC", aic
	print "chi2", chi2
	print "r.chi2", chi2 / (len(ssfr)-5.)
	print "KS", ks_test
	
	util.plot_data_log(root_dir,model_name,theta_pass,new_contin_snr)

	plt.show()

