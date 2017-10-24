import utility_functions as util
import os
import numpy as np
import emcee
import matplotlib.pyplot as plt
from chainconsumer import ChainConsumer
import pickle as pick
import time
import scipy.optimize as opt

def nicelog_snr(logssfr,theta):
	a, k, ssfr0, alpha = theta
	logssfr0 = np.log10(ssfr0)
	ssfr = 10**logssfr
	return a + a * np.log(ssfr/ssfr0 + alpha) / k

def lnprior(theta):
	a, k, ssfr0, alpha = theta
	#theta_pass = 4.2e-14, 0.272, 3.8e-11
	if (0.05e-14 < a < 120e-14) and (0.001 < k < 5.5) and (0.1e-11 < ssfr0 < 1000e-11) and (0.01 < alpha < 1.6):
		return 0.
	return -np.inf

def lnlike(theta, logssfr, snr, snr_err):
	snr_model = nicelog_snr(logssfr, theta)
	return -0.5*(np.sum( ((snr-snr_model)/snr_err)**2.  ))

def run_grid():
	if util.does_grid_exist(model_name,root_dir):
		print 'Grid already exists, using existing grid...'
		resolution, likelihoods, parameters, theta_max = util.read_grid(model_name,root_dir)
		a_par, k_par, s0_par, alpha_par = parameters
	else:
		print 'Grid does not exist, computing grid...'
	
		resolution = 20
		#theta_pass = 4.2e-14, 0.272, 3.8e-11, 0.9
		a_min, a_max = 2e-14, 20e-14
		k_min, k_max = 0.05, 1.4
		s0_min, s0_max = 1e-11, 60e-11
		alpha_min, alpha_max = 0.3, 1.4

		#a_min, a_max = 2e-14, 13e-14
		#k_min, k_max = 0.05, 2
		#s0_min, s0_max = 1e-11, 20e-11
		#alpha_min, alpha_max = 0.55, 1.4

		# Reading in data
		snr_illu, logssfr_illu, times_all = util.get_illustris_ssfr_ssnr()
		logssfr = np.array([])
		snr = np.array([])
		for ii in range(len(logssfr_illu)):
			if logssfr_illu[ii] > -20.:
				snr = np.append(snr,snr_illu[ii])
				logssfr = np.append(logssfr,logssfr_illu[ii])
		snr_err = 1.

		a_par = np.linspace(a_min,a_max,resolution)
		k_par = np.linspace(k_min,k_max,resolution)
		s0_par = np.linspace(s0_min,s0_max,resolution)
		alpha_par = np.linspace(alpha_min,alpha_max,resolution)

		likelihoods = np.ones((resolution,resolution,resolution,resolution))
		max_like = 0.

		for ii in np.arange(resolution):
			if ii%2 == 0:
				print np.round((float(ii) / resolution) * 100.,2), "% Done"
			for jj in np.arange(resolution):
				for kk in np.arange(resolution):
					for ll in np.arange(resolution):
						theta = a_par[ii], k_par[jj], s0_par[kk], alpha_par[ll]
						likelihoods[ii,jj,kk,ll] = np.exp(lnlike(theta,logssfr,snr,snr_err))
						if likelihoods[ii,jj,kk,ll] > max_like:
							max_like = likelihoods[ii,jj,kk,ll]
							theta_max = a_par[ii], k_par[jj], s0_par[kk], alpha_par[ll]
							print "New max like:", max_like
							print theta_max, "\n"
		likelihoods /= np.sum(likelihoods)
		output = open(root_dir + 'Data/MCMC_illustris_nicelog_grid.pkl','wb')
		parameters = a_par, k_par, s0_par, alpha_par
		result = resolution, likelihoods, parameters, theta_max
 		pick.dump(result,output)
 		output.close()

	a_like = np.zeros(resolution)
	k_like = np.zeros(resolution)
	s0_like = np.zeros(resolution)
	alpha_like = np.zeros(resolution)
	for ii in np.arange(resolution):
		a_like[ii]    = np.sum(likelihoods[ii,:,:,:])
		k_like[ii]    = np.sum(likelihoods[:,ii,:,:])
		s0_like[ii]    = np.sum(likelihoods[:,:,ii,:])
		alpha_like[ii]    = np.sum(likelihoods[:,:,:,ii])
	
	yes_chainconsumer = False
	if yes_chainconsumer:
		print "Defining chainconsumer"
		c = ChainConsumer()
		print "Adding chain"
		c.add_chain([a_par, k_par, s0_par, alpha_par], parameters=["a","k","s0","alpha"],weights=likelihoods,grid=True)
		print "Doing plot"
		fig = c.plotter.plot()

	'''
	plt.figure()
	ax = plt.subplot()
	ax.set_xscale("log")
	plt.plot(a_par,a_like,'x')
	plt.xlabel('a')

	plt.figure()
	ax = plt.subplot()
	ax.set_xscale("log")
	plt.plot(k_par,k_like,'x')
	plt.xlabel('k')

	plt.figure()
	ax = plt.subplot()
	ax.set_xscale("log")
	plt.plot(s0_par,s0_like,'x')
	plt.xlabel('ssfr0')

	plt.figure()
	ax = plt.subplot()
	ax.set_xscale("log")
	plt.plot(alpha_par,alpha_like,'x')
	plt.xlabel('alpha')
	'''
	
	# These are the marginalised maximum likelihood parameters
	a_fit = a_par[np.argmax(a_like)]
	k_fit = k_par[np.argmax(k_like)]
	s0_fit = s0_par[np.argmax(s0_like)]
	alpha_fit = alpha_par[np.argmax(alpha_like)]

	print "ML parameters:"
	#theta_pass = a_fit, k_fit, s0_fit, alpha_fit
	theta_pass = theta_max
	print theta_pass
	return theta_pass

root_dir = '/Users/perandersen/Data/SNR-AB/'
model_name = 'illustris_nicelog'

if __name__ == '__main__':
	t0 = time.time()
	
	snr_illu, logssfr_illu, times_all = util.get_illustris_ssfr_ssnr()
	logssfr = np.array([])
	snr = np.array([])
	for ii in range(len(logssfr_illu)):
		if logssfr_illu[ii] > -20.:
			snr = np.append(snr,snr_illu[ii])
			logssfr = np.append(logssfr,logssfr_illu[ii])
	snr_err = 1.


	#theta_pass = run_grid()
	theta_pass = 1.12e-13, 0.49, 6e-11, 0.7

	
	chi2 = np.sum( ((snr-nicelog_snr(logssfr, theta_pass))/snr_err)**2.  )
	bic = chi2 + 4.*np.log(len(logssfr))
	aic = chi2 + 4.*2.
	ks_test = util.ks_test(logssfr,snr,nicelog_snr,theta_pass,visualise=False)

	print ""
	print "BIC", bic
	print "AIC", aic
	print "chi2", chi2
	print "r.chi2", chi2 / (len(logssfr)-4.)
	print "KS", ks_test
	
	logssfr_values = np.linspace(-15,-8,100000)
	snr_values = nicelog_snr(logssfr_values, theta_pass)
	plt.figure()
	ax = plt.subplot()
	plt.xlabel('log(sSFR)',size='large')
	plt.ylabel('sSNR',size='large')
	ax.set_yscale("log")
	plt.xlim((-15,-8))
	plt.ylim((2e-14,2e-12))
	plt.plot(logssfr_values, snr_values,c='k',lw=3)
	plt.plot(logssfr,snr,'r.')
	

	

	print "Done in", time.time() - t0, "seconds"

	plt.show()

