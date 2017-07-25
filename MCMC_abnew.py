import os
import numpy as np
import emcee
import matplotlib.pyplot as plt
from chainconsumer import ChainConsumer
import scipy.optimize as opt
import pickle as pick
import time

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

	return logssfr, ssfr, snr, snr_err

def plot_data(theta):
	logssfr, ssfr, snr, snr_err = read_data()

	logssfr_values = np.linspace(-13,-8,10000)
	snr_values = new_snr(logssfr_values, theta)
	plt.figure()
	ax = plt.subplot()
	ax.set_yscale("log")
	plt.plot(logssfr_values, snr_values,c='k',lw=3)
	plt.errorbar(logssfr,snr,yerr=snr_err,fmt='o')

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
	k1, k2, ssfr1, ssfr2, ssfra = theta
	if (1e-13 < k1 < 1e-12) and (5e-13 < k2 < 1e-10) and (1e-11 < ssfr1 < 1e-9) and (5e-13 < ssfr2 < 1e-10) and (1e-5 < ssfra < 1e-2):
		return 0.
	return -np.inf

def lnlike(theta, ssfr, snr, snr_err):
	logssfr = np.log10(ssfr)
	k1, k2, ssfr1, ssfr2, ssfra = theta

	snr_model = new_snr(logssfr, theta)
	return -0.5*(np.sum( ((snr-snr_model)/snr_err)**2.  ))

def lnprob(theta, ssfr, snr, snr_err):
	lp = lnprior(theta)
	if not np.isfinite(lp):
		return -np.inf
	return lp + lnlike(theta, ssfr, snr, snr_err)

if __name__ == '__main__':
	t0 = time.time()

	
	if os.path.isfile('MCMC_abnew.pkl'):
		print 'Chains already exist, using existing chains'
		pkl_data_file = open('MCMC_abnew.pkl','rb')
		samples = pick.load(pkl_data_file)
		pkl_data_file.close()
		print np.shape(samples)
	else:
		logssfr, ssfr, snr, snr_err = read_data()

		ndim = 5	
		nwalkers = 500
		pos_min = np.array([1e-13, 5e-13, 1e-11, 5e-13, 1e-5])
		pos_max = np.array([1e-12, 1e-10, 1e-9, 1e-10, 1e-2])
		psize = pos_max - pos_min
		pos = [pos_min + psize*np.random.rand(ndim) for ii in range(nwalkers)]

		sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(ssfr, snr, snr_err), threads=1)
		pos, prob, state = sampler.run_mcmc(pos, 3000)
		sampler.reset()

		pos, prob, state = sampler.run_mcmc(pos, 30000)

		samples = sampler.flatchain
		output = open('MCMC_abnew.pkl','wb')
 		pick.dump(samples,output)
 		output.close()
		print np.shape(samples)

	plt.figure()
	plt.hist(samples[:,0],bins=100)
	plt.xlabel('k1')

	plt.figure()
	plt.hist(samples[:,1],bins=100)
	plt.xlabel('k2')

	plt.figure()
	plt.hist(samples[:,2],bins=100)
	plt.xlabel('ssfr1')

	plt.figure()
	plt.hist(samples[:,3],bins=100)
	plt.xlabel('ssfr2')

	plt.figure()
	plt.hist(samples[:,4],bins=100)
	plt.xlabel('ssfra')

	c = ChainConsumer()
	c.add_chain(samples, parameters=["$k_1$", "$k_2$", "$sSFR_1$", "$sSFR_2$", "$sSFR_a$"])
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
	
	logssfr, ssfr, snr, snr_err = read_data()
	theta_pass = 4.63367747971e-13, 6.2105042757e-12, 2.87220202454e-10, 3.12145638807e-11, 0.00623198967077
	#theta_pass = 4.628e-13, 6.105e-11, 2.885e-10, 1.008e-11, 6.100e-4
	#print lnprob(theta_pass, ssfr, snr, snr_err)
	nll = lambda *args: -lnlike(*args)
	bnds = ((None, None), (None, None),(0., None),(0., None),(0., None))
	#result = opt.minimize(nll, [theta_pass],args=(ssfr,snr,snr_err),bounds=bnds,options={'disp': True},method='L-BFGS-B')
	result = opt.fmin(nll, [theta_pass],args=(ssfr,snr,snr_err),disp=0,xtol=0.000000001)
	#print result
	

	k1_ml, k2_ml, ssfr1_ml, ssfr2_ml, ssfra_ml = result
	#theta_pass = k1_ml, k2_ml, ssfr1_ml, ssfr2_ml, ssfra_ml
	#print lnprob(theta_pass, ssfr, snr, snr_err)
	chi2 = np.sum( ((snr-new_snr(logssfr, theta_pass))/snr_err)**2.  )
	bic = chi2 + 5.*np.log(len(logssfr))
	aic = chi2 + 5.*2.

	print "Done in", time.time() - t0, "seconds"
	print ""
	print "BIC", bic
	print "AIC", aic
	print "chi2", chi2
	print "r.chi2", chi2 / (len(logssfr)-5.)
	
	plot_data(theta_pass)

	plt.show()

