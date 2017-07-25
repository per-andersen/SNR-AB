import os
import numpy as np
import emcee
import matplotlib.pyplot as plt
from chainconsumer import ChainConsumer
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

	logssfr_values = np.linspace(-13,-8,100000)
	snr_values = sigmoid_snr(logssfr_values, theta)
	plt.figure()
	ax = plt.subplot()
	plt.xlim((-13,-8))
	#ax.set_yscale("log")
	plt.plot(logssfr_values, snr_values,c='k',lw=3)
	plt.errorbar(logssfr,snr,yerr=snr_err,fmt='o')

def sigmoid_snr(logssfr,theta):
	a, c, d = theta
	ssfr = 10**logssfr

	return a / (1. + np.exp( (-logssfr+c)*d ))

def lnprior(theta):
	a, c, d = theta
	if (1e-15 < a < 1e-8) and (-12. < c < -1.) and (0. < d < 10.):
		return 0.
	return -np.inf

def lnlike(theta, ssfr, snr, snr_err):
	logssfr = np.log10(ssfr)
	a, c, d = theta

	snr_model = sigmoid_snr(logssfr, theta)
	return -0.5*(np.sum( ((snr-snr_model)/snr_err)**2.  ))

def lnprob(theta, ssfr, snr, snr_err):
	lp = lnprior(theta)
	if not np.isfinite(lp):
		return -np.inf
	return lp + lnlike(theta, ssfr, snr, snr_err)

if __name__ == '__main__':
	t0 = time.time()

	if os.path.isfile('MCMC_sigmoid.pkl'):
		print 'Chains already exist, using existing chains...'
		pkl_data_file = open('MCMC_sigmoid.pkl','rb')
		samples = pick.load(pkl_data_file)
		pkl_data_file.close()
		print np.shape(samples)
	else:
		print 'Chains do not exist, computing chains...'
		logssfr, ssfr, snr, snr_err = read_data()

		ndim = 3	
		nwalkers = 500
		pos_min = np.array([1e-15, -12., 0.])
		pos_max = np.array([1e-8, -1., 10.])
		psize = pos_max - pos_min
		pos = [pos_min + psize*np.random.rand(ndim) for ii in range(nwalkers)]

		sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(ssfr, snr, snr_err), threads=1)
		pos, prob, state = sampler.run_mcmc(pos, 500)
		sampler.reset()

		pos, prob, state = sampler.run_mcmc(pos, 5000)

		samples = sampler.flatchain
		output = open('MCMC_sigmoid.pkl','wb')
 		pick.dump(samples,output)
 		output.close()
		print np.shape(samples)

	plt.figure()
	plt.hist(samples[:,0],bins=300)
	plt.xlabel('a')

	plt.figure()
	plt.hist(samples[:,1],bins=300)
	plt.xlabel('c')

	plt.figure()
	plt.hist(samples[:,2],bins=300)
	plt.xlabel('d')


	c = ChainConsumer()
	c.add_chain(samples, parameters=["$a$", "$c$", "$d$"])
	#figw = c.plotter.plot_walks()
	fig = c.plotter.plot()
	summary =  c.analysis.get_summary()

	a_fit = summary["$a$"][1]
	c_fit = summary["$c$"][1]
	d_fit = summary["$d$"][1]

	theta_pass = a_fit, c_fit, d_fit
	
	print 'a', a_fit
	print 'c', c_fit
	print 'd', d_fit

	
	print "Done in", time.time() - t0, "seconds"
	theta_pass = a_fit, c_fit, d_fit
	plot_data(theta_pass)

	plt.show()

