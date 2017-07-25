import os
import numpy as np
import emcee
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from chainconsumer import ChainConsumer
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

def plot_data(theta):
	logssfr_sul, snr_sul, snr_err_sul, logssfr_mat, snr_mat, snr_err_mat = read_data_names()

	logssfr_values = np.linspace(-13,-8,100000)
	snr_values = simple_snr(logssfr_values, theta)
	plt.figure()
	ax = plt.subplot()
	plt.xlabel('log(sSFR)',size='large')
	plt.ylabel('sSNR',size='large')
	plt.xlim((-13,-8))
	ax.set_yscale("log")
	plt.plot(logssfr_values, snr_values,c='k',lw=3)
	plt.errorbar(logssfr_sul,snr_sul,yerr=snr_err_sul,fmt='o',label='Sullivan et al. (2006)')
	plt.errorbar(logssfr_mat,snr_mat,yerr=snr_err_mat,fmt='x',label='Smith et al. (2012)')
	plt.legend(frameon=False, loc=2, fontsize=17)

	plt.savefig(root_dir + 'Plots/model_simple.pdf')


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
	resolution = 20

	#ML parameters: [  1.20103392 -13.38221532 -10.07802213   2.61730236]

	a_min, a_max = 2e-14, 8e-14
	b_min, b_max = 0.0002, 0.0008

	# Reading in data
	logssfr, ssfr,  snr, snr_err = read_data()

	likelihoods = np.ones((resolution,resolution))

	a_par = np.linspace(a_min,a_max,resolution)
	b_par = np.linspace(b_min,b_max,resolution)

	for ii in np.arange(resolution):
		print ii
		for jj in np.arange(resolution):
			theta = a_par[ii], b_par[jj]
			likelihoods[ii,jj] = -lnlike(theta,ssfr,snr,snr_err)*2.
	
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
	
	plt.figure()
	plt.plot(a_par,a_like)
	plt.xlabel('a')

	plt.figure()
	plt.plot(b_par,b_like)
	plt.xlabel('b')

	plt.show()

if __name__ == '__main__':
	t0 = time.time()
	
	root_dir = '/Users/perandersen/Data/SNR-AB/'
	
	if os.path.isfile(root_dir + 'Data/MCMC_simple.pkl'):
		print 'Chains already exist, using existing chains...'
		pkl_data_file = open(root_dir + 'Data/MCMC_simple.pkl','rb')
		samples = pick.load(pkl_data_file)
		pkl_data_file.close()
		print np.shape(samples)
	else:
		print 'Chains do not exist, computing chains...'
		logssfr, ssfr, snr, snr_err = read_data()

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
	fig = c.plotter.plot(figsize=(8,6))
	fig.savefig(root_dir + 'Plots/marginals_simple.pdf')
	summary =  c.analysis.get_summary()

	a_fit = summary["$A$"][1]
	c_fit = summary["$B$"][1]

	theta_pass = a_fit, c_fit
	
	print 'A', a_fit
	print 'B', c_fit
	
	logssfr, ssfr, snr, snr_err = read_data_simple()
	
	theta_pass = a_fit, c_fit
	chi2 = np.sum( ((snr-simple_snr(logssfr, theta_pass))/snr_err)**2.  )
	bic = chi2 + 2.*np.log(len(logssfr))
	aic = chi2 + 2.*2.

	print "Done in", time.time() - t0, "seconds"
	print ""
	print "BIC", bic
	print "AIC", aic
	print "chi2", chi2
	print "r.chi2", chi2 / (len(logssfr)-2.)
	#theta_pass = 5e-14, 3e-4
	plot_data(theta_pass)
	
	plt.show()

	#run_grid()

