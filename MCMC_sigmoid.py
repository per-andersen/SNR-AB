import os
import numpy as np
import emcee
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from chainconsumer import ChainConsumer
import scipy.optimize as opt
import pickle as pick
import time

def read_data_names():
	mathew = np.genfromtxt('Mathew/Smith_2012_Figure5_Results.txt')
	sullivan = np.genfromtxt('Mathew/Smith_2012_Figure5_Sullivan_Results.txt')
	mathew = np.delete(mathew,1, axis=0) #!!! Justify this

	logssfr_mat = mathew[:,0]
	snr_mat = mathew[:,1]
	snr_err_upp_mat = mathew[:,2]
	snr_err_low_mat = mathew[:,3]
	snr_err_mat = np.sqrt(snr_err_low_mat**2 + snr_err_upp_mat**2) #!!! Check this is ok

	logssfr_sul = sullivan[:,0]
	snr_sul = sullivan[:,1]
	snr_err_sul = sullivan[:,2]

	logssfr = np.concatenate((logssfr_mat,logssfr_sul))
	ssfr = 10**logssfr
	snr = np.concatenate((snr_mat,snr_sul))
	snr_err = np.concatenate((snr_err_mat,snr_err_sul))

	return logssfr_sul, snr_sul, snr_err_sul, logssfr_mat, snr_mat, snr_err_mat

def read_data():
	mathew = np.genfromtxt('Mathew/Smith_2012_Figure5_Results.txt')
	sullivan = np.genfromtxt('Mathew/Smith_2012_Figure5_Sullivan_Results.txt')
	mathew = np.delete(mathew,1, axis=0) #!!! Justify this

	logssfr_mat = mathew[:,0]
	snr_mat = mathew[:,1]
	snr_err_upp_mat = mathew[:,2]
	snr_err_low_mat = mathew[:,3]
	snr_err_mat = np.sqrt(snr_err_low_mat**2 + snr_err_upp_mat**2) #!!! Check this is ok

	logssfr_sul = sullivan[:,0]
	snr_sul = sullivan[:,1]
	snr_err_sul = sullivan[:,2]

	logssfr = np.concatenate((logssfr_mat,logssfr_sul))
	ssfr = 10**logssfr
	snr = np.concatenate((snr_mat,snr_sul))
	snr_err = np.concatenate((snr_err_mat,snr_err_sul))

	return ssfr, snr, snr_err

def plot_data(theta):
	logssfr_sul, snr_sul, snr_err_sul, logssfr_mat, snr_mat, snr_err_mat = read_data_names()

	ssfr_values = np.logspace(-13,-8,10000)
	snr_values = sigmoid_snr(ssfr_values, theta)
	plt.figure()
	ax = plt.subplot()
	plt.xlabel('log(sSFR)',size='large')
	plt.ylabel('sSNR',size='large')
	plt.xlim((-13,-8))
	plt.ylim((2e-14,1e-12))
	ax.set_yscale("log")
	plt.plot(np.log10(ssfr_values), snr_values,c='k',lw=3)
	plt.errorbar(logssfr_sul,snr_sul,yerr=snr_err_sul,fmt='o',label='Sullivan et al. (2006)')
	plt.errorbar(logssfr_mat,snr_mat,yerr=snr_err_mat,fmt='x',label='Smith et al. (2012)')
	plt.legend(frameon=False, loc=2, fontsize=16)

	plt.savefig(root_dir + 'Plots/model_sigmoid.pdf')

def sigmoid_snr(ssfr,theta):
	a, b, c, d = theta

	return b + a / (1. + np.exp( (-ssfr+c)*d ))

def lnlike(theta, ssfr, snr, snr_err):

	snr_model = sigmoid_snr(ssfr, theta)
	return -0.5*(np.sum( ((snr-snr_model)/snr_err)**2.))

def run_scipy():
	theta_pass = 6e-13, 4.4e-14, 1e-9, 1e10
	ssfr, snr, snr_err = read_data()

	nll = lambda *args: -lnlike(*args)
	result = opt.minimize(nll, [theta_pass],args=(ssfr,snr,snr_err),options={'disp': True})
	print "ML parameters:", result.x
	#a_fit, b_fit, c_fit, d_fit = result.x[0], result.x[1], result.x[2], result.x[3]
	#theta_pass = a_fit, b_fit, c_fit, d_fit
	theta_pass = result.x
	return theta_pass

def run_grid():
	if os.path.isfile(root_dir + 'Data/MCMC_sigmoid_grid.pkl'):
		print 'Grid already exists, using existing chains...'
		pkl_data_file = open(root_dir + 'Data/MCMC_sigmoid_grid.pkl','rb')
		resolution, likelihoods, a_par, b_par, c_par, d_par, theta_max = pick.load(pkl_data_file)
		pkl_data_file.close()
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
		logssfr, logsnr, snr_err = read_data()

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
							print "New max like:", max_like
							print theta_max, "\n"
		likelihoods /= np.sum(likelihoods)
		output = open(root_dir + 'Data/MCMC_sigmoid_grid.pkl','wb')
		result = resolution, likelihoods, a_par, b_par, c_par, d_par, theta_max
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

	#theta_pass = run_scipy()
	theta_pass = run_grid()

	
	ssfr, snr, snr_err = read_data()
	chi2 = np.sum( ((snr-sigmoid_snr(ssfr, theta_pass))/snr_err)**2.  )
	bic = chi2 + 4.*np.log(len(ssfr))
	aic = chi2 + 4.*2.

	print "Done in", time.time() - t0, "seconds"
	print ""
	print "BIC", bic
	print "AIC", aic
	print "chi2", chi2
	print "r.chi2", chi2 / (len(ssfr)-4.)

	
	#theta_pass = 6e-13, 4.4e-14, 0.4e-9, 1.1e10
	plot_data(theta_pass)

	plt.show()

