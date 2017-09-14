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

def lnlike(theta, ssfr, snr, snr_err):
	logssfr = np.log10(ssfr)
	snr_model = nicelog_snr(logssfr, theta)
	return -0.5*(np.sum( ((snr-snr_model)/snr_err)**2.  ))

def lnprob(theta, ssfr, snr, snr_err):
	lp = lnprior(theta)
	if not np.isfinite(lp):
		return -np.inf
	return lp + lnlike(theta, ssfr, snr, snr_err)

def run_emcee():
	if util.do_chains_exist(model_name,root_dir):
		print 'Chains already exist, using existing chains...'
		samples = util.read_chains(model_name,root_dir)
		print np.shape(samples)
	else:
		print 'Chains do not exist, computing chains...'
		logssfr, ssfr, snr, snr_err = util.read_data_with_log()

		ndim = 4
		nwalkers = 300

		pos_min = np.array([0.05e-14, 0.001, 0.1e-11, 0.01])
		pos_max = np.array([120e-14, 5.5, 1000e-11, 1.6])
		psize = pos_max - pos_min
		pos = [pos_min + psize*np.random.rand(ndim) for ii in range(nwalkers)]

		sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(ssfr, snr, snr_err), threads=1)
		pos, prob, state = sampler.run_mcmc(pos, 500)
		sampler.reset()

		pos, prob, state = sampler.run_mcmc(pos, 4000)

		samples = sampler.flatchain
		output = open(root_dir + 'Data/MCMC_nicelog.pkl','wb')
 		pick.dump(samples,output)
 		output.close()
		print np.shape(samples)

	plt.figure()
	plt.hist(samples[:,0],bins=300)
	plt.xlabel('A')

	plt.figure()
	plt.hist(samples[:,1],bins=300)
	plt.xlabel('k')

	plt.figure()
	plt.hist(samples[:,2],bins=300)
	plt.xlabel('sSFR_0')


	c = ChainConsumer()
	c.add_chain(samples, parameters=["$A$", "$k$", "$sSFR_0$", "$alpha$"])
	#figw = c.plotter.plot_walks()
	fig = c.plotter.plot(figsize=(8,6))
	fig.savefig(root_dir + 'Plots/marginals_nicelog.pdf')
	summary =  c.analysis.get_summary()

	a_fit = summary["$A$"][1]
	c_fit = summary["$k$"][1]
	d_fit = summary["$sSFR_0$"][1]
	alpha_fit = summary["$alpha$"][1]
	
	print 'A', a_fit
	print 'k', c_fit
	print 'sSFR_0', d_fit
	print 'alpha', alpha_fit
	
	theta_pass = a_fit, c_fit, d_fit, alpha_fit
	return theta_pass

def run_grid():
	if util.does_grid_exist(model_name,root_dir):
		print 'Grid already exists, using existing grid...'
		resolution, likelihoods, parameters, theta_max = util.read_grid(model_name,root_dir)
		a_par, k_par, s0_par, alpha_par = parameters
	else:
		print 'Grid does not exist, computing grid...'
	
		resolution = 50
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
		ssfr, snr, snr_err = util.read_data()

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
						likelihoods[ii,jj,kk,ll] = np.exp(lnlike(theta,ssfr,snr,snr_err))
						if likelihoods[ii,jj,kk,ll] > max_like:
							max_like = likelihoods[ii,jj,kk,ll]
							theta_max = a_par[ii], k_par[jj], s0_par[kk], alpha_par[ll]
							#print "New max like:", max_like
							#print theta_max, "\n"
		likelihoods /= np.sum(likelihoods)
		output = open(root_dir + 'Data/MCMC_nicelog_grid.pkl','wb')
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

def run_grid_fast(ssfr, snr, snr_err):
	#a, k, ssfr0, alpha
	#1.1183673469387756e-13, 0.49081632653061219, 1.6653061224489797e-10, 0.7265306122448979

	# We do the first very quick run
	resolution_1, resolution_2 = 40, 10
	second_run_offset = 0.6

	a_min, a_max = 0.1e-13, 4.0e-13
	k_min, k_max = 0.1, 2.0
	s0_min, s0_max = 0.05e-10, 7e-10
	alpha_min, alpha_max = 0.1, 1.2

	warnings = True

	a_par = np.linspace(a_min,a_max,resolution_1)
	k_par = np.linspace(k_min,k_max,resolution_1)
	s0_par = np.linspace(s0_min,s0_max,resolution_1)
	alpha_par = np.linspace(alpha_min,alpha_max,resolution_1)

	likelihoods = np.ones((resolution_1,resolution_1,resolution_1,resolution_1))
	max_like = 0.

	for ii in np.arange(resolution_1):
		for jj in np.arange(resolution_1):
			for kk in np.arange(resolution_1):
				for ll in np.arange(resolution_1):
					theta = a_par[ii], k_par[jj], s0_par[kk], alpha_par[ll]
					likelihoods[ii,jj,kk,ll] = np.exp(lnlike(theta,ssfr,snr,snr_err))

	i,j,k,l = np.unravel_index(likelihoods.argmax(), likelihoods.shape)
	theta_max = a_par[i], k_par[j], s0_par[k], alpha_par[l]
	print theta_max

	if warnings:
		if i == 0:
			print "Warning, best fit a is at lower limit!"
			#util.plot_data(root_dir, model_name, theta_max, nicelog_snr)
			#plt.show()
		if i == (resolution_1-1):
			print "Warning, best fit a is at upper limit!"
			#util.plot_data(root_dir, model_name, theta_max, nicelog_snr)
			#plt.show()

		if j == 0:
			print "Warning, best fit k is at lower limit!"
			#util.plot_data(root_dir, model_name, theta_max, nicelog_snr)
			#plt.show()
		if j == (resolution_1-1):
			print "Warning, best fit k is at upper limit!"
			#util.plot_data(root_dir, model_name, theta_max, nicelog_snr)
			#plt.show()

		if k == 0:
			print "Warning, best fit s0 is at lower limit!"
			#util.plot_data(root_dir, model_name, theta_max, nicelog_snr)
			#plt.show()
		if k == (resolution_1-1):
			print "Warning, best fit s0 is at upper limit!"
			#util.plot_data(root_dir, model_name, theta_max, nicelog_snr)
			#plt.show()

		if l == 0:
			print "Warning, best fit alpha is at lower limit!"
			#util.plot_data(root_dir, model_name, theta_max, nicelog_snr)
			#plt.show()
		if l == (resolution_1-1):
			print "Warning, best fit alpha is at upper limit!"
			#util.plot_data(root_dir, model_name, theta_max, nicelog_snr)
			#plt.show()

	theta_pass = theta_max
	
	#print "ML parameters:"
	
	return theta_pass

def run_scipy(ssfr, snr, snr_err):
	logssfr, ssfr, snr, snr_err = util.read_data_with_log()

	theta_pass = 1.1183673469387756e-13, 0.49081632653061219, 1.6653061224489797e-10, 0.7265306122448979
	a_min, a_max = 1e-13, 7.e-13
	k_min, k_max = 0.1, 1.
	s0_min, s0_max = 1e-10, 2e-10
	alpha_min, alpha_max = 0.6, 9.
	bnds = ((a_min,a_max),(k_min,k_max),(s0_min,s0_max),(alpha_min,alpha_max))
	ssfr, snr, snr_err = util.read_data()
	nll = lambda *args: -lnlike(*args)
	
	#result = opt.minimize(nll, [theta_pass],args=(ssfr,snr,snr_err),options={'disp': True},bounds=bnds,method='SLSQP')
	result = opt.fmin(nll, [theta_pass],args=(ssfr,snr,snr_err),ftol=0.000001,xtol=0.00001)
	result = opt.fmin_cg(nll, [theta_pass],args=(ssfr,snr,snr_err))
	print "ML parameters:", result
	#a_fit, b_fit, c_fit, d_fit = result.x[0], result.x[1], result.x[2], result.x[3]
	#theta_pass = a_fit, b_fit, c_fit, d_fit
	theta_pass = result
	return theta_pass

def bootstrap_uncertainties():
	
	time_start = time.time()
	
	a_min, a_max = 0.1e-13, 4.0e-13
	k_min, k_max = 0.1, 2.0
	s0_min, s0_max = 0.05e-10, 7e-10
	alpha_min, alpha_max = 0.1, 1.2

	n_runs = 400
	pars = np.zeros((4,n_runs))

	ssfr, snr, snr_err = util.read_data()

	ndata = len(ssfr)
	index_array = np.arange(ndata)

	#plt.figure()
	#ax = plt.subplot()
	counter = 0
	while counter < n_runs:
		ii = counter
		print ii
		#snr_shift = snr + np.random.normal(size=len(snr)) * snr_err

		indices = np.random.choice(index_array,size=ndata,replace=True)

		ssfr_sub = ssfr[indices]
		snr_sub = snr[indices]
		snr_err_sub = snr_err[indices]
		pars[:,ii] = run_grid_fast(ssfr_sub, snr_sub, snr_err_sub)
		if np.isclose(pars[:,ii][3]-alpha_min,0.):
			counter -= 1

		#util.plot_data(root_dir, model_name, pars[:,ii], nicelog_snr)
		#plt.show()
		#plt.plot(np.log10(ssfr_sub),snr_sub,'bo',alpha=0.05)
		counter += 1
	#plt.errorbar(np.log10(ssfr),snr,yerr=snr_err,fmt='o',color='k')
	#plt.xlabel('log(sSFR)',size='large')
	#plt.ylabel('sSNR',size='large')
	#ax.set_yscale("log")
	#plt.xlim((-13,-8))
	#plt.ylim((5e-15,6e-12))
		
	
	plt.figure()
	plt.hist(pars[0,:],bins=10,range=(a_min,a_max))
	plt.xlabel('a')

	plt.figure()
	plt.hist(pars[1,:],bins=10,range=(k_min,k_max))
	plt.xlabel('k')

	plt.figure()
	plt.hist(pars[2,:],bins=10,range=(s0_min,s0_max))
	plt.xlabel('ssfr0')

	plt.figure()
	plt.hist(pars[3,:],bins=10,range=(alpha_min,alpha_max))
	plt.xlabel('alpha')
	
	runtime = time.time() - time_start
	np.savetxt('bootstrap_parameters.txt',pars.T)
	np.savetxt('runtime.txt',np.array([runtime]),fmt='%4.2f')

root_dir = '/Users/perandersen/Data/SNR-AB/'
model_name = 'nicelog'

if __name__ == '__main__':
	t0 = time.time()

	ssfr, snr, snr_err = util.read_data()
	#run_scipy(ssfr, snr, snr_err)

	#bootstrap_uncertainties()
	#numerical_uncertainties()

	
	logssfr, ssfr, snr, snr_err = util.read_data_with_log()

	#theta_pass = run_emcee()
	#theta_pass = run_grid()
	theta_pass = 4.2e-14, 0.272, 3.8e-11, 0.9

	
	chi2 = np.sum( ((snr-nicelog_snr(logssfr, theta_pass))/snr_err)**2.  )
	bic = chi2 + 4.*np.log(len(logssfr))
	aic = chi2 + 4.*2.
	ks_test = util.ks_test(np.log10(ssfr),snr,nicelog_snr,theta_pass,visualise=True)

	print ""
	print "BIC", bic
	print "AIC", aic
	print "chi2", chi2
	print "r.chi2", chi2 / (len(logssfr)-4.)
	print "KS", ks_test
	
	util.plot_data(root_dir, model_name, theta_pass, nicelog_snr)
	

	

	print "Done in", time.time() - t0, "seconds"

	plt.show()

