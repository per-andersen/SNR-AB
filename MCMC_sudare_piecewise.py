import utility_functions as util
import os
import numpy as np
import emcee
import matplotlib.pyplot as plt
from chainconsumer import ChainConsumer
import scipy.optimize as opt
import pickle as pick
import time

def piecewise_snr(ssfr,theta):
	slope, offset, x1, x2 = theta

	snr_return = np.zeros(len(ssfr))
	for ii in np.arange(len(ssfr)):
		if ssfr[ii] < x1:
			snr_return[ii] = x1**(slope) * offset
		
		elif ssfr[ii] > x2:
			snr_return[ii] = x2**(slope) * offset
		
		else:
			snr_return[ii] = ssfr[ii]**(slope) * offset
	return snr_return

def lnlike(theta, ssfr, snr, snr_err):
	snr_model = piecewise_snr(ssfr, theta)
	return -0.5*(np.sum( ((snr-snr_model)/snr_err)**2.  ))

def run_grid():
	if util.does_grid_exist(model_name,root_dir):
		print 'Grid already exists, using existing grid...'
		resolution, likelihoods, parameters, theta_max = util.read_grid(model_name,root_dir)
		k1_par, k2_par, x1_par, x2_par = parameters
	else:
		print 'Grid does not exist, computing grid...'
	
		resolution = 100

		k1_min, k1_max = 0.4, 0.9
		k2_min, k2_max = 0.5e-7, 9e-7
		x1_min, x1_max = 1e-12, 5e-11
		x2_min, x2_max = 0.1e-9, 3e-9

		# Reading in data
		'''
		ssfr, snr, snr_err = util.read_data()
		
		logssfr_sudare = np.array([-12.2, -10.5, -9.7, -9.])
		ssfr_sudare = 10**logssfr_sudare
		ssnr_sudare = np.array([0.5,1.2,3.2,6.5])*1e-13
		sudare_ssnr_uncertainty = np.array([[0.305,0.45,1.1,2.05]])*1e-13

		ssfr = np.append(ssfr,ssfr_sudare)
		snr = np.append(snr,ssnr_sudare)
		snr_err = np.append(snr_err,sudare_ssnr_uncertainty)
		'''

		ssfr = np.array([])
		snr = np.array([])
		snr_err = np.array([])

		logssfr_sul, snr_sul, snr_err_sul, logssfr_mat, snr_mat, snr_err_mat = util.read_data_names()
		logssfr_sudare = np.array([-12.2, -10.5, -9.7, -9.])
		ssfr_sudare = 10**logssfr_sudare
		ssnr_sudare = np.array([0.5,1.2,3.2,6.5])*1e-13
		sudare_ssnr_uncertainty = np.array([[0.305,0.45,1.1,2.05]])*1e-13

		#ssfr = np.append(ssfr,ssfr_sudare)
		#snr = np.append(snr,ssnr_sudare)
		#snr_err = np.append(snr_err,sudare_ssnr_uncertainty)

		ssfr = np.append(ssfr,10**logssfr_mat)
		snr = np.append(snr,snr_mat)
		snr_err = np.append(snr_err,snr_err_mat)

		ssfr = np.append(ssfr,10**np.delete(logssfr_sul,[1]))
		snr = np.append(snr,np.delete(snr_sul,[1]))
		snr_err = np.append(snr_err,np.delete(snr_err_sul,[1]))


		k1_par = np.linspace(k1_min,k1_max,resolution)
		k2_par = np.linspace(k2_min,k2_max,resolution)
		x1_par = np.linspace(x1_min,x1_max,resolution)
		x2_par = np.linspace(x2_min,x2_max,resolution)

		likelihoods = np.ones((resolution,resolution,resolution,resolution))
		max_like = 0.

		for ii in np.arange(resolution):
			if ii%2 == 0:
				print np.round((float(ii) / resolution) * 100.,2), "% Done"
			for jj in np.arange(resolution):
				for kk in np.arange(resolution):
					for ll in np.arange(resolution):
						theta = k1_par[ii], k2_par[jj], x1_par[kk], x2_par[ll]
						likelihoods[ii,jj,kk,ll] = np.exp(lnlike(theta,ssfr,snr,snr_err))
						if likelihoods[ii,jj,kk,ll] > max_like:
							max_like = likelihoods[ii,jj,kk,ll]
							theta_max = k1_par[ii], k2_par[jj], x1_par[kk], x2_par[ll]
							#print "New max like:", max_like
							#print theta_max, "\n"
		likelihoods /= np.sum(likelihoods)
		output = open(root_dir + 'Data/MCMC_sudare_piecewise_grid.pkl','wb')
		parameters = k1_par, k2_par, x1_par, x2_par
		result = resolution, likelihoods, parameters, theta_max
 		pick.dump(result,output)
 		output.close()

	k1_like = np.zeros(resolution)
	k2_like = np.zeros(resolution)
	x1_like = np.zeros(resolution)
	x2_like = np.zeros(resolution)
	for ii in np.arange(resolution):
		k1_like[ii]    = np.sum(likelihoods[ii,:,:,:])
		k2_like[ii]    = np.sum(likelihoods[:,ii,:,:])
		x1_like[ii]    = np.sum(likelihoods[:,:,ii,:])
		x2_like[ii]    = np.sum(likelihoods[:,:,:,ii])
	
	
	plt.figure()
	ax = plt.subplot()
	ax.set_xscale("log")
	plt.plot(k1_par,k1_like,'x')
	plt.xlabel('slope')

	plt.figure()
	ax = plt.subplot()
	ax.set_xscale("log")
	plt.plot(k2_par,k2_like,'x')
	plt.xlabel('offset')

	plt.figure()
	ax = plt.subplot()
	ax.set_xscale("log")
	plt.plot(x1_par,x1_like,'x')
	plt.xlabel('x1')

	plt.figure()
	ax = plt.subplot()
	ax.set_xscale("log")
	plt.plot(x2_par,x2_like,'x')
	plt.xlabel('x2')
	
	
	# These are the marginalised maximum likelihood parameters
	k1_fit = k1_par[np.argmax(k1_like)]
	k2_fit = k2_par[np.argmax(k2_like)]
	x1_fit = x1_par[np.argmax(x1_like)]
	x2_fit = x2_par[np.argmax(x2_like)]

	print "ML parameters:"
	#theta_pass = k1_fit, k2_fit, x1_fit, x2_fit
	theta_pass = theta_max
	print theta_pass
	return theta_pass

def prompt_fraction(theta):
	slope, offset, s1, s2 = theta
	k1 = (s1**slope) * offset
	k2 = ( (s2**slope) * offset / s2 - k1/s1) / (np.log(s1/s2))
	return (k1/s1) / (k1/s1 + k2*np.log(s1/s2))

root_dir = '/Users/perandersen/Data/SNR-AB/'
model_name = 'sudare_piecewise'

if __name__ == '__main__':
	t0 = time.time()


	#theta_pass = run_emcee()
	theta_pass = run_grid()
	#theta_pass = 0.58585858585858586, 1.1905050505050503e-07, 1.0060606060606059e-11, 1.0373737373737374e-09

	
	ssfr, snr, snr_err = util.read_data()
	chi2 = np.sum( ((snr-piecewise_snr(ssfr, theta_pass))/snr_err)**2.  )
	bic = chi2 + 4.*np.log(len(ssfr))
	aic = chi2 + 4.*2.
	ks_test = util.ks_test(ssfr,snr,piecewise_snr,theta_pass)

	print "Done in", time.time() - t0, "seconds"
	print ""
	print "BIC", bic
	print "AIC", aic
	print "chi2", chi2
	print "r.chi2", chi2 / (len(ssfr)-4.)
	print "KS", ks_test
	
	
	plt.figure()
	ax = plt.subplot()
	plt.xlabel('log(sSFR)',size='large')
	plt.ylabel('sSNR',size='large')
	ax.set_yscale("log")
	plt.xlim((-13,-8))
	plt.ylim((2e-14,2e-12))
	ssfr_values = np.logspace(-13,-8,10000)
	snr_values = piecewise_snr(ssfr_values, theta_pass)

	plt.plot(np.log10(ssfr_values), snr_values,c='k',lw=3)

	logssfr_sudare = np.array([-12.2, -10.5, -9.7, -9.])
	ssnr_sudare = np.array([0.5,1.2,3.2,6.5])*1e-13
	sudare_ssnr_uncertainty = np.array([[0.28,0.4,1.1,1.9],[0.33,0.5,1.1,2.2]])*1e-13
	
	#plt.errorbar(logssfr_sudare,ssnr_sudare,yerr=sudare_ssnr_uncertainty,fmt='s',ls='',c='g',label='SUDARE')

	logssfr_sul, snr_sul, snr_err_sul, logssfr_mat, snr_mat, snr_err_mat = util.read_data_names()

	plt.errorbar(logssfr_mat,snr_mat,yerr=snr_err_mat,label='Smith et al. (2012)',fmt='x',ls='',c='orange')

	plt.errorbar(np.delete(logssfr_sul,[1]),np.delete(snr_sul,[1]),yerr=np.delete(snr_err_sul,[1]),label='Sullivan et al. (2006)',fmt='o',ls='',c='b')

	plt.legend(frameon=False, loc=2, fontsize=16)

	print "Done in", time.time() - t0, "seconds"

	plt.show()

