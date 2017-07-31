import utility_functions as util
import os
import numpy as np
import emcee
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from chainconsumer import ChainConsumer
import scipy.optimize as opt
import pickle as pick
import time

def sigmoid_snr(ssfr,theta):
	a, b, c, d = theta

	return b + a / (1. + np.exp( (-ssfr+c)*d ))

def lnlike(theta, ssfr, snr, snr_err):

	snr_model = sigmoid_snr(ssfr, theta)
	return -0.5*(np.sum( ((snr-snr_model)/snr_err)**2.))

def run_scipy():
	theta_pass = 6e-13, 4.4e-14, 1e-9, 1e10
	ssfr, snr, snr_err = util.read_data()

	nll = lambda *args: -lnlike(*args)
	result = opt.minimize(nll, [theta_pass],args=(ssfr,snr,snr_err),options={'disp': True})
	print "ML parameters:", result.x
	#a_fit, b_fit, c_fit, d_fit = result.x[0], result.x[1], result.x[2], result.x[3]
	#theta_pass = a_fit, b_fit, c_fit, d_fit
	theta_pass = result.x
	return theta_pass

def run_grid():
	if util.does_grid_exist(model_name,root_dir):
		print 'Grid already exists, using existing chains...'
		resolution, likelihoods, parameters, theta_max = util.read_grid(model_name,root_dir)
		a_par, b_par, c_par, d_par = parameters 
	else:
		print 'Grid does not exist, computing grid...'
	
		resolution = 100

		'''
		# These limits give a pretty decent overview of this local extrema
		a_min, a_max = 1e-13, 9e-13
		b_min, b_max = 1e-17, 6e-14
		c_min, c_max = 0.1e-10, 1e-9
		d_min, d_max = 1e9, 50e9
		'''

		# These limits focus in on the local extrema above
		a_min, a_max = 2e-13, 6e-13
		b_min, b_max = 6e-15, 6e-14
		c_min, c_max = 6e-11, 3e-10
		d_min, d_max = 7e9, 80e9

		# Reading in data
		logssfr, logsnr, snr_err = util.read_data()

		a_par = np.linspace(a_min,a_max,resolution)
		b_par = np.linspace(b_min,b_max,resolution)
		c_par = np.linspace(c_min,c_max,resolution)
		d_par = np.linspace(d_min,d_max,resolution)

		likelihoods = np.ones((resolution,resolution,resolution,resolution))
		max_like = 0.

		for ii in np.arange(resolution):
			if ii%5 == 0:
				print np.round((float(ii) / resolution) * 100.,2), "% Done"
			for jj in np.arange(resolution):
				for kk in np.arange(resolution):
					for ll in np.arange(resolution):
						theta = a_par[ii], b_par[jj], c_par[kk], d_par[ll]
						likelihoods[ii,jj,kk,ll] = np.exp(lnlike(theta,logssfr,logsnr,snr_err))
						if likelihoods[ii,jj,kk,ll] > max_like:
							max_like = likelihoods[ii,jj,kk,ll]
							theta_max = a_par[ii], b_par[jj], c_par[kk], d_par[ll]
							#print "New max like:", max_like
							#print theta_max, "\n"
		likelihoods /= np.sum(likelihoods)
		output = open(root_dir + 'Data/MCMC_sigmoid_grid.pkl','wb')
		parameters = a_par, b_par, c_par, d_par
		result = resolution, likelihoods, parameters, theta_max
 		pick.dump(result,output)
 		output.close()

	a_like = np.zeros(resolution)
	b_like = np.zeros(resolution)
	c_like = np.zeros(resolution)
	d_like = np.zeros(resolution)
	for ii in np.arange(resolution):
		a_like[ii] = np.sum(likelihoods[ii,:,:,:])
		b_like[ii] = np.sum(likelihoods[:,ii,:,:])
		c_like[ii] = np.sum(likelihoods[:,:,ii,:])
		d_like[ii] = np.sum(likelihoods[:,:,:,ii])
	
	'''
	plt.figure()
	ax = plt.subplot()
	ax.set_xscale("log")
	plt.plot(a_par,a_like)
	plt.xlabel('a')

	plt.figure()
	ax = plt.subplot()
	ax.set_xscale("log")
	plt.plot(b_par,b_like)
	plt.xlabel('b')

	plt.figure()
	ax = plt.subplot()
	ax.set_xscale("log")
	plt.plot(c_par,c_like)
	plt.xlabel('c')

	plt.figure()
	ax = plt.subplot()
	ax.set_xscale("log")
	plt.plot(d_par,d_like)
	plt.xlabel('d')
	'''

	a_fit = a_par[np.argmax(a_like)]
	b_fit = b_par[np.argmax(b_like)]
	c_fit = c_par[np.argmax(c_like)]
	d_fit = d_par[np.argmax(d_like)]

	print "ML parameters:"
	theta_pass = a_fit, b_fit, c_fit, d_fit
	print theta_pass
	return theta_max


if __name__ == '__main__':
	t0 = time.time()
	
	root_dir = '/Users/perandersen/Data/SNR-AB/'
	model_name = 'sigmoid'

	#theta_pass = run_scipy()
	theta_pass = run_grid()
	
	ssfr, snr, snr_err = util.read_data()
	chi2 = np.sum( ((snr-sigmoid_snr(ssfr, theta_pass))/snr_err)**2.  )
	bic = chi2 + 4.*np.log(len(ssfr))
	aic = chi2 + 4.*2.
	ks_test = util.ks_test(ssfr,snr,sigmoid_snr,theta_pass)

	print "Done in", time.time() - t0, "seconds"
	print ""
	print "BIC", bic
	print "AIC", aic
	print "chi2", chi2
	print "r.chi2", chi2 / (len(ssfr)-4.)
	print "KS", ks_test

	

	util.plot_data_log(root_dir,model_name, theta_pass, sigmoid_snr)

	plt.show()

