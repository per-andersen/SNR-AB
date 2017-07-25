import os
import numpy as np
import emcee
import matplotlib.pyplot as plt
from chainconsumer import ChainConsumer
import scipy.optimize as opt
import pickle as pick
import time

def read_data_simple():
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

	return logssfr, ssfr, snr, snr_err

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
	# The uncertainty is really asymmetric but it is a sub percent effect, so is ignored
	snr_err_mat_log = np.log10(snr_mat) - np.log10(snr_mat - snr_err_low_mat)

	logssfr_sul = sullivan[:,0]
	snr_sul = sullivan[:,1]
	snr_err_sul = sullivan[:,2]
	snr_err_sul_log = np.log10(snr_sul) - np.log10(snr_sul - snr_err_sul)


	logssfr = np.concatenate((logssfr_mat,logssfr_sul))
	ssfr = 10**logssfr
	snr = np.concatenate((snr_mat,snr_sul))
	logsnr = np.log10(snr)
	snr_err = np.concatenate((snr_err_mat_log,snr_err_sul_log))

	return logssfr, logsnr, snr_err

def plot_data(theta):
	logssfr_sul, snr_sul, snr_err_sul, logssfr_mat, snr_mat, snr_err_mat = read_data_names()

	logssfr_values = np.linspace(-13,-8,10000)
	logsnr_values = new_snr(logssfr_values, theta)
	snr_values = 10.**logsnr_values
	plt.figure()
	ax = plt.subplot()
	plt.xlim((-13,-8))
	plt.xlabel('log(sSFR)')
	plt.ylabel('sSNR')
	ax.set_yscale("log")
	plt.plot(logssfr_values, snr_values,c='k',lw=3)
	plt.errorbar(logssfr_sul,snr_sul,yerr=snr_err_sul,fmt='o',label='Sullivan et al. (2006)')
	plt.errorbar(logssfr_mat,snr_mat,yerr=snr_err_mat,fmt='x',label='Smith et al. (2012)')
	plt.legend(frameon=False, loc=2, fontsize=16)

	plt.savefig(root_dir + 'Plots/model_abnew_log.pdf')


def new_snr(logssfr,theta):
	k1, k2, logssfr1, logssfr2, logssfra = theta
	

	snr_return = np.zeros(len(logssfr))
	
	for ii in np.arange(len(logssfr)):
		
		if logssfr1 < logssfr[ii]:
			snr_return[ii] = k1
		
		elif logssfr[ii] < logssfr2:
			snr_return[ii] = k2 * logssfra
		
		else:
			snr_return[ii] = logssfr[ii] * (k1 / logssfr1 + k2*np.log(logssfr1 / logssfr2))

	return snr_return



def lnlike(theta, logssfr, logsnr, snr_err):
	k1, k2, ssfr1, ssfr2, ssfra = theta

	snr_model = new_snr(logssfr, theta)
	return -0.5*(np.sum( ((logsnr-snr_model)/0.2)**2.  ))

def read_pickled_samples():
	pkl_data_file = open(root_dir + 'Data/MCMC_abnew_log.pkl','rb')
	samples = pick.load(pkl_data_file)
	pkl_data_file.close()
	return samples

def run_scipy():
	logssfr, logsnr, snr_err = read_data()

	theta_pass = -12.4, 0.009, -9.5, -10.9, -1500.
	nll = lambda *args: -lnlike(*args)
	result = opt.minimize(nll, [theta_pass],args=(logssfr,logsnr,snr_err),options={'disp': False})
	
	return result.x

def run_emcee():
	if os.path.isfile(root_dir + 'Data/MCMC_abnew_log.pkl'):
		print 'Chains already exist, using existing chains'
		samples = read_pickled_samples()
		print np.shape(samples)
	else:
		print 'Chains do not exist, computing chains...'

		# Setting parameter top hat priors
		k1_min, k1_max = -13., -11.6
		k2_min, k2_max = 0.00001, 0.25
		ssfr1_min, ssfr1_max = -11., -9.
		ssfr2_min, ssfr2_max = -12.5, -0.01
		ssfra_min, ssfra_max = -13000., -50.

		ndim = 5	
		nwalkers = 700
		nburn = 4000
		nsample = 30000

		# These functions define the prior and the function to apply prior to likelihood
		def lnprior(theta):
			k1, k2, ssfr1, ssfr2, ssfra = theta
			#theta_pass = -12.4, 0.009, -9.5, -10.9, -1500.

			if (k1_min < k1 < k1_max) and (k2_min < k2 < k2_max) and (ssfr1_min < ssfr1 < ssfr1_max) and (ssfr2_min < ssfr2 < ssfr2_max) and (ssfra_min < ssfra < ssfra_max):
				return 0.
			return -np.inf

		def lnprob(theta, logssfr, logsnr, snr_err):
			lp = lnprior(theta)
			if not np.isfinite(lp):
				return -np.inf
			return lp + lnlike(theta, logssfr, logsnr, snr_err)

		# Reading in data
		logssfr, logsnr, snr_err = read_data()

		# Setting initial position of walkers
		pos_min = np.array([k1_min, k2_min, ssfr1_min, ssfr2_min, ssfra_min])
		pos_max = np.array([k1_max, k2_max, ssfr1_max, ssfr2_max, ssfra_max])
		psize = pos_max - pos_min
		#pos = [pos_min + psize*np.random.rand(ndim) for ii in range(nwalkers)]
		pos = [np.array([ -12.2870,0.00889,-9.722,-10.9162,-1500.]) + 2e-5*np.random.randn(ndim) for i in range(nwalkers)]

		# Defining sampler
		sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(logssfr, logsnr, snr_err), threads=1)
		
		# Performing burn in
		#pos, prob, state = sampler.run_mcmc(pos, nburn)
		#sampler.reset()

		# Main sampling run
		pos, prob, state = sampler.run_mcmc(pos, nsample)

		# These plots are for diagnostics use
		plt.figure()
		plt.plot(sampler.chain[:,:,0].T,'b',alpha=0.05)
		plt.xlabel('par0')
		plt.figure()
		plt.plot(sampler.chain[:,:,3].T,'b',alpha=0.05)
		plt.xlabel('par3')
		plt.figure()
		plt.plot(sampler.lnprobability[:,:].T,'b',alpha=0.05)
		plt.xlabel('lnprob')

		# Formatting and saving output
		samples = sampler.flatchain
		output = open(root_dir + 'Data/MCMC_abnew_log.pkl','wb')
 		pick.dump(samples,output)
 		output.close()
		print np.shape(samples)

	plt.figure()
	plt.hist(samples[:,0],bins=200)
	plt.xlabel('k1')

	plt.figure()
	plt.hist(samples[:,1],bins=200)
	plt.xlabel('k2')

	plt.figure()
	plt.hist(samples[:,2],bins=200)
	plt.xlabel('ssfr1')

	plt.figure()
	plt.hist(samples[:,3],bins=200)
	plt.xlabel('ssfr2')

	plt.figure()
	plt.hist(samples[:,4],bins=200)
	plt.xlabel('ssfra')

	c = ChainConsumer()
	c.add_chain(samples, parameters=["$k_1$", "$k_2$", "$sSFR_1$", "$sSFR_2$", "$sSFR_a$"])
	c.configure(smooth=False,bins=100,sigmas=[0,1,2,3])
	#figw = c.plotter.plot_walks()
	fig = c.plotter.plot()
	fig.savefig(root_dir + 'Plots/marginals_abnew_log.pdf')
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
	
if __name__ == '__main__':
	t0 = time.time()
	
	root_dir = '/Users/perandersen/Data/SNR-AB/'
	theta_pass = run_scipy()
	#theta_pass = run_emcee()
	
	print theta_pass
	#logssfr, logsnr, snr_err = read_data()
	logssfr, ssfr, snr, snr_err = read_data_simple()
	chi2 = np.sum( ((snr - 10.**new_snr(logssfr, theta_pass))/snr_err)**2.  )
	bic = chi2 + 5.*np.log(len(logssfr))
	aic = chi2 + 5.*2.

	print "Done in", time.time() - t0, "seconds"
	print ""
	print "BIC", bic
	print "AIC", aic
	print "chi2", chi2
	print "r.chi2", chi2 / (len(logssfr)-5.)
	
	plot_data(theta_pass)

	#plt.show()

