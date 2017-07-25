import os
import numpy as np
import emcee
import matplotlib.pyplot as plt
import matplotlib.cm as cm
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
	#snr_err = 0.2

	return logssfr, logsnr, snr_err

def plot_data(theta):
	logssfr_sul, snr_sul, snr_err_sul, logssfr_mat, snr_mat, snr_err_mat = read_data_names()

	logssfr_values = np.linspace(-13,-8,100000)
	logsnr_values = sigmoid_snr(logssfr_values, theta)
	snr_values = 10.**logsnr_values
	plt.figure()
	ax = plt.subplot()
	plt.xlim((-13,-8))
	ax.set_yscale("log")
	plt.plot(logssfr_values, snr_values,c='k',lw=3)
	plt.errorbar(logssfr_sul,snr_sul,yerr=snr_err_sul,fmt='o',label='Sullivan et al. (2006)')
	plt.errorbar(logssfr_mat,snr_mat,yerr=snr_err_mat,fmt='x',label='Smith et al. (2012)')
	plt.legend(frameon=False, loc=2, fontsize=16)

	plt.savefig(root_dir + 'Plots/model_sigmoid_log.pdf')

def sigmoid_snr(logssfr,theta):
	a, b, c, d = theta

	return b + a / (1. + np.exp( (-logssfr+c)*d ))

def lnlike(theta, logssfr, logsnr, snr_err):

	snr_model = sigmoid_snr(logssfr, theta)
	return -0.5*(np.sum( ((logsnr-snr_model)/snr_err)**2.))


def run_emcee():

	if os.path.isfile(root_dir + 'Data/MCMC_sigmoid_log.pkl'):
		print 'Chains already exist, using existing chains...'
		pkl_data_file = open(root_dir + 'Data/MCMC_sigmoid_log.pkl','rb')
		samples = pick.load(pkl_data_file)
		pkl_data_file.close()
		print np.shape(samples)
	else:
		print 'Chains do not exist, computing chains...'

		# Setting parameter top hat priors
		a_min, a_max = 0.01, 50.
		b_min, b_max = -22, -10.
		c_min, c_max = -20., -2.
		d_min, d_max = 0.01, 15.

		#ML parameters: [  1.20103392 -13.38221532 -10.07802213   2.61730236]

		# Setting emcee run parameters
		ndim = 4	
		nwalkers = 500
		nburn = 1800
		nsample = 17000

		# These functions define the prior and the function to apply prior to likelihood
		def lnprior(theta):
			a, b, c, d = theta

			if (a_min < a < a_max) and (b_min < b < b_max) and (c_min < c < c_max) and (d_min < d < d_max):
				return 0.
			return -np.inf

		def lnprob(theta, ssfr, snr, snr_err):
			lp = lnprior(theta)
			if not np.isfinite(lp):
				return -np.inf
			return lp + lnlike(theta, ssfr, snr, snr_err)

		# Reading in data
		logssfr, logsnr, snr_err = read_data()

		
		# Setting initial position of walkers
		pos_min = np.array([a_min, b_min, c_min, d_min])
		pos_max = np.array([a_max, b_max, c_max, d_max])
		psize = pos_max - pos_min
		pos = [np.array([1.20103392,-13.38221532, -10.07802213, 2.61730236]) + 1e-4*np.random.randn(ndim) for i in range(nwalkers)]

		# Defining sampler
		sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(logssfr, logsnr, snr_err), threads=1)

		# Performing burn in
		pos, prob, state = sampler.run_mcmc(pos, nburn)
		sampler.reset()

		# Main sampling run
		pos, prob, state = sampler.run_mcmc(pos, nsample)

		print("Autocorr guess:", sampler.get_autocorr_time)
		#print("Maximum autocorr", np.max(sampler.acor))

		
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

		#print(sampler.lnprobability[:10,0])

		# Formatting and saving output
		samples = sampler.flatchain
		output = open(root_dir + 'Data/MCMC_sigmoid_log.pkl','wb')
 		pick.dump(samples,output)
 		output.close()
		print np.shape(samples)

	# Plotting posterior of parameters for diagnostics use
	plt.figure()
	plt.hist(samples[:,0],bins=300)
	plt.xlabel('a')

	plt.figure()
	plt.hist(samples[:,1],bins=300)
	plt.xlabel('b')

	plt.figure()
	plt.hist(samples[:,2],bins=300)
	plt.xlabel('c')

	plt.figure()
	plt.hist(samples[:,3],bins=300)
	plt.xlabel('d')

	# Plotting using chainconsumer and getting ML parameters
	c = ChainConsumer()
	c.add_chain(samples, parameters=["$a$", "$b$", "$c$", "$d$"])
	c.configure(smooth=False,bins=100,sigmas=[0,1,2,3])
	#figw = c.plotter.plot_walks()
	fig = c.plotter.plot()
	fig.savefig(root_dir + 'Plots/marginals_sigmoid_log.pdf')
	summary =  c.analysis.get_summary()
	
	a_fit = summary["$a$"][1]
	b_fit = summary["$b$"][1]
	c_fit = summary["$c$"][1]
	d_fit = summary["$d$"][1]

	theta_pass = a_fit, b_fit, c_fit, d_fit
	
	print 'a', a_fit
	print 'b', b_fit
	print 'c', c_fit
	print 'd', d_fit

	return theta_pass

def run_scipy():

	logssfr, logsnr, snr_err = read_data()
	#theta_pass = 1.5, -13.4, -9.8, 0.01 # Initial guess
	theta_pass = 1.122, -13.404, -10.08, 0.12 # Initial guess

	nll = lambda *args: -lnlike(*args)
	result = opt.minimize(nll, [theta_pass],args=(logssfr,logsnr,snr_err),options={'disp': True})
	print "ML parameters:", result.x
	a_fit, b_fit, c_fit, d_fit = result.x[0], result.x[1], result.x[2], result.x[3]
	theta_pass = a_fit, b_fit, c_fit, d_fit
	return theta_pass


def run_grid():
	if os.path.isfile(root_dir + 'Data/MCMC_sigmoid_log_grid.pkl'):
		print 'Grid already exists, using existing chains...'
		pkl_data_file = open(root_dir + 'Data/MCMC_sigmoid_log_grid.pkl','rb')
		resolution, likelihoods, a_par, b_par, c_par, d_par = pick.load(pkl_data_file)
		pkl_data_file.close()
	else:
		print 'Grid does not exist, computing grid...'
	
		resolution = 40

		#ML parameters: [  1.20103392 -13.38221532 -10.07802213   2.61730236]

		# These with a resolution of 40 yield decent results at peak likelihood
		#a_min, a_max = 1.1, 1.4
		#b_min, b_max = -13.5, -13.1
		#c_min, c_max = -10.2, -9.8
		#d_min, d_max = 1.5, 3.

		a_min, a_max = 0.1, 10.
		b_min, b_max = -15.5, -12.5
		c_min, c_max = -12., -7.
		d_min, d_max = 0.001, 10.

		# Reading in data
		logssfr, logsnr, snr_err = read_data()

		a_par = np.linspace(a_min,a_max,resolution)
		b_par = np.linspace(b_min,b_max,resolution)
		c_par = np.linspace(c_min,c_max,resolution)
		d_par = np.linspace(d_min,d_max,resolution)

		# Adding another point by hand
		a_par = np.sort(np.append(a_par,1.20103))
		b_par = np.sort(np.append(b_par,-13.382215))
		c_par = np.sort(np.append(c_par,-10.078022))
		d_par = np.sort(np.append(d_par,2.617301))
		resolution += 1

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
							print "New max like:", max_like
							print theta, "\n"
		likelihoods /= np.sum(likelihoods)
		output = open(root_dir + 'Data/MCMC_sigmoid_log_grid.pkl','wb')
		result = resolution, likelihoods, a_par, b_par, c_par, d_par
 		pick.dump(result,output)
 		output.close()

	#plt.figure()
	#plt.xlabel("a")
	#plt.ylabel("b")
	#im = plt.imshow(likelihoods[:,:,10,10],interpolation='none',origin='lower',cmap=cm.Greys,extent=(a_min, a_max, b_min, b_max))
	#plt.colorbar()

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
	plt.plot(a_par,a_like)
	plt.xlabel('a')

	plt.figure()
	plt.plot(b_par,b_like)
	plt.xlabel('b')

	plt.figure()
	plt.plot(c_par,c_like)
	plt.xlabel('c')

	plt.figure()
	plt.plot(d_par,d_like)
	plt.xlabel('d')

	a_fit = a_par[np.argmax(a_like)]
	b_fit = b_par[np.argmax(b_like)]
	c_fit = c_par[np.argmax(c_like)]
	d_fit = d_par[np.argmax(d_like)]

	print "ML parameters:"
	theta_pass = a_fit, b_fit, c_fit, d_fit
	print theta_pass
	return theta_pass


if __name__ == '__main__':
	t0 = time.time()
	
	root_dir = '/Users/perandersen/Data/SNR-AB/'

	#theta_pass = run_scipy()
	theta_pass = run_emcee()
	#theta_pass = run_grid()
	
	
	logssfr, ssfr, snr, snr_err = read_data_simple()
	chi2 = np.sum( ((snr-10.**sigmoid_snr(logssfr, theta_pass))/snr_err)**2.  )
	bic = chi2 + 4.*np.log(len(logssfr))
	aic = chi2 + 4.*2.

	print "Done in", time.time() - t0, "seconds"
	print ""
	print "BIC", bic
	print "AIC", aic
	print "chi2", chi2
	print "r.chi2", chi2 / (len(logssfr)-4.)
	#theta_pass = a_fit, b_fit, c_fit, d_fit
	plot_data(theta_pass)

	plt.show()
	
	#theta1 = 1.142105263157895, -13.382215, -10.078022000000001, 0.62105263157894741
	#print np.exp(lnlike(theta1,logssfr,logsnr,snr_err))
	#theta2 = 1.20103392, -13.38221532, -10.07802213,   2.61730236
	#print np.exp(lnlike(theta2,logssfr,logsnr,snr_err))
