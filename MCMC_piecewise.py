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
		k1_min, k1_max = 1e-13, 9e-13
		k2_min, k2_max = 1e-5, 7e-4
		x1_min, x1_max = 1e-10, 8e-10
		x2_min, x2_max = 1e-11, 8e-11

		ndim = 4	
		nwalkers = 300
		nburn = 20
		nsample = 150

		# These functions define the prior and the function to apply prior to likelihood
		def lnprior(theta):
			k1, k2, x1, x2 = theta
			if (k1_min < k1 < k1_max) and (k2_min < k2 < k2_max) and (x1_min < x1 < x1_max) and (x2_min < x2 < x2_max):
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
		pos_min = np.array([k1_min, k2_min, x1_min, x2_min])
		pos_max = np.array([k1_max, k2_max, x1_max, x2_max])
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
		#ax.set_yscale("log")
		#ax.set_xscale("log")
		plt.plot(sampler.lnprobability[:,:].T,'b',alpha=0.05)
		plt.xlabel('lnprob')

		# Formatting and saving output
		samples = sampler.flatchain
		output = open(root_dir + 'Data/MCMC_piecewise.pkl','wb')
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
	plt.xlabel('x1')

	plt.figure()
	ax = plt.subplot()
	ax.set_xscale("log")
	plt.hist(samples[:,3],bins=100)
	plt.xlabel('x2')

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

	theta_pass = k1_fit, k2_fit, x1_fit, x2_fit
	
	print 'k1', k1_fit
	print 'k2', k2_fit
	print 'x1', x1_fit
	print 'x2', x2_fit
	return theta_pass

def run_grid():
	if util.does_grid_exist(model_name,root_dir):
		print 'Grid already exists, using existing grid...'
		resolution, likelihoods, parameters, theta_max = util.read_grid(model_name,root_dir)
		k1_par, k2_par, x1_par, x2_par = parameters
	else:
		print 'Grid does not exist, computing grid...'
	
		resolution = 100

		k1_min, k1_max = 0.4, 0.8
		k2_min, k2_max = 0.1e-8, 90e-8
		x1_min, x1_max = 0.1e-11, 4e-11
		x2_min, x2_max = 0.1e-9, 3e-9

		# Reading in data
		ssfr, snr, snr_err = util.read_data()

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
		output = open(root_dir + 'Data/MCMC_piecewise_grid.pkl','wb')
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
	
	'''
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
	'''
	
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

def numerical_uncertainties():

	theta_pass = np.array([0.5859, 1.1905e-7, 1.037e-9, 1.006e-11])
	theta_pass[2] = 0.5e-9

	'''
	# Small test
	def test_func(theta):
		a, b, c = theta
		return b*a**2 + c
	dpa = util.numerical_derivative(test_func,0.01,np.array([2.,0.1,3.]),0)
	dpb = util.numerical_derivative(test_func,0.001,np.array([2.,0.1,3.]),1)
	dpc = util.numerical_derivative(test_func,0.01,np.array([2.,0.1,3.]),2)
	print dpa, dpb, dpc
	'''

	dp1 = util.numerical_derivative(prompt_fraction,0.0001,theta_pass,0)
	dp2 = util.numerical_derivative(prompt_fraction,1e-8,theta_pass,1)
	dp3 = util.numerical_derivative(prompt_fraction,1e-12,theta_pass,2)
	dp4 = util.numerical_derivative(prompt_fraction,1e-14,theta_pass,3)

	s1, s2, s3, s4 = 0.084, 2.2e-7, 0.41e-9, 0.55e-11
	print dp1 
	print dp2
	print dp3
	print dp4

	print (dp1*s1)**2, (dp2*s2)**2, (dp3*s3)**2, (dp4*s4)**2
	spf = np.sqrt((dp1*s1)**2 + (dp2*s2)**2 + (dp3*s3)**2 + (dp4*s4)**2)
	print "f_p", prompt_fraction(theta_pass), "+/-", spf

def bootstrap_fp_uncertainties():
	theta_pass = np.array([0.5859, 1.1905e-7, 1.037e-9, 1.006e-11],dtype=np.float128)
	sigmas = np.array([0.084, 2.2e-7, 0.41e-9, 0.55e-11],dtype=np.float128)

	theta_pass[2] = 2.e-9

	prompt_fraction_max = prompt_fraction(theta_pass)

	prompt_fractions = np.array([])
	for ii in np.arange(100000):
		try:
			theta_new = theta_pass + np.random.normal(size=4) * sigmas
			theta_new[2] = 2.e-9
			fp = prompt_fraction(theta_new)
			if np.isnan(fp):
				continue
			prompt_fractions = np.append(prompt_fractions, fp)
		except:
			continue

	hist, bin_edges = np.histogram(prompt_fractions,bins=80,range=(0.,1.),density=True)
	bin_midpoints = bin_edges[:-1] + (bin_edges[1:] - bin_edges[:-1]) / 2.

	

	index_max = np.argmin( np.abs(bin_midpoints - prompt_fraction_max) )
	bin_max = bin_midpoints[index_max]

	prob = 0.
	index_right = 1
	while prob < 0.34:
		prob = np.trapz(hist[index_max:index_max+index_right],bin_edges[index_max:index_max+index_right])
		index_right += 1
	bin_right = bin_midpoints[index_max + index_right]

	prob = 0.
	index_left = 2
	while prob < 0.34:
		prob = np.trapz(hist[index_max-index_left:index_max+1],bin_edges[index_max-index_left:index_max+1])
		index_left += 1
	bin_left = bin_midpoints[index_max - index_left]

	plt.figure()
	plt.xlim((0,1))
	plt.hist(prompt_fractions,bins=80,range=(0.,1.),normed=True)
	plt.axvline(bin_left,c='r')
	plt.axvline(bin_max,c='g')
	plt.axvline(bin_right,c='b')

	print "f_p : ", prompt_fraction_max, "+", bin_right - bin_max, "/-", np.abs(bin_max - bin_left)

root_dir = '/Users/perandersen/Data/SNR-AB/'
model_name = 'piecewise'

if __name__ == '__main__':
	t0 = time.time()

	#numerical_uncertainties()
	bootstrap_fp_uncertainties()
	#theta_pass = run_emcee()
	#theta_pass = run_grid()

	'''
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
	
	util.plot_data_log(root_dir,model_name,theta_pass,piecewise_snr)
	'''

	print "Done in", time.time() - t0, "seconds"

	plt.show()

