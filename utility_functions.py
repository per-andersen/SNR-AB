import os
import numpy as np
import matplotlib.pyplot as plt
import pickle as pick

def do_chains_exist(model_name,root_dir):
	if os.path.isfile(root_dir + 'Data/MCMC_' + model_name + '.pkl'):
		return True
	return False

def read_chains(model_name,root_dir):
	pkl_data_file = open(root_dir + 'Data/MCMC_' + model_name + '.pkl','rb')
	samples = pick.load(pkl_data_file)
	pkl_data_file.close()
	return samples

def does_grid_exist(model_name,root_dir):
	if os.path.isfile(root_dir + 'Data/MCMC_' + model_name + '_grid.pkl'):
		return True
	return False

def read_grid(model_name,root_dir):
	pkl_data_file = open(root_dir + 'Data/MCMC_' + model_name + '_grid.pkl','rb')
	resolution, likelihoods, parameters, theta_max = pick.load(pkl_data_file)
	pkl_data_file.close()
	return resolution, likelihoods, parameters, theta_max

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

	return logssfr_sul, snr_sul, snr_err_sul, logssfr_mat, snr_mat, snr_err_mat

def read_data():
	logssfr_sul, snr_sul, snr_err_sul, logssfr_mat, snr_mat, snr_err_mat = read_data_names()

	logssfr = np.concatenate((logssfr_mat,logssfr_sul))
	ssfr = 10**logssfr
	snr = np.concatenate((snr_mat,snr_sul))
	snr_err = np.concatenate((snr_err_mat,snr_err_sul))

	return ssfr, snr, snr_err

def read_data_with_log():
	logssfr_sul, snr_sul, snr_err_sul, logssfr_mat, snr_mat, snr_err_mat = read_data_names()

	logssfr = np.concatenate((logssfr_mat,logssfr_sul))
	ssfr = 10**logssfr
	snr = np.concatenate((snr_mat,snr_sul))
	snr_err = np.concatenate((snr_err_mat,snr_err_sul))

	return logssfr, ssfr, snr, snr_err

def plot_data_log(root_dir, model_name, theta, snr_func,combined_plot=False):
	logssfr_sul, snr_sul, snr_err_sul, logssfr_mat, snr_mat, snr_err_mat = read_data_names()

	ssfr_values = np.logspace(-13,-8,10000)
	snr_values = snr_func(ssfr_values, theta)
	if combined_plot == False:
		plt.figure()
		ax = plt.subplot()
		plt.xlabel('log(sSFR)',size='large')
		plt.ylabel('sSNR',size='large')
		ax.set_yscale("log")
	plt.xlim((-13,-8))
	plt.ylim((2e-14,1e-12))
	plt.plot(np.log10(ssfr_values), snr_values,c='k',lw=3)
	plt.errorbar(logssfr_sul,snr_sul,yerr=snr_err_sul,fmt='o',label='Sullivan et al. (2006)')
	plt.errorbar(logssfr_mat,snr_mat,yerr=snr_err_mat,fmt='x',label='Smith et al. (2012)')
	if combined_plot == False:	
		plt.legend(frameon=False, loc=2, fontsize=16)
		plt.savefig(root_dir + 'Plots/model_' + model_name + '.pdf')

	

def plot_data(root_dir, model_name, theta, snr_func,combined_plot=False):
	logssfr_sul, snr_sul, snr_err_sul, logssfr_mat, snr_mat, snr_err_mat = read_data_names()

	logssfr_values = np.linspace(-13,-8,100000)
	snr_values = snr_func(logssfr_values, theta)
	if combined_plot == False:
		plt.figure()
		ax = plt.subplot()
		plt.xlabel('log(sSFR)',size='large')
		plt.ylabel('sSNR',size='large')
		ax.set_yscale("log")
	plt.xlim((-13,-8))
	plt.ylim((2e-14,1e-12))
	plt.plot(logssfr_values, snr_values,c='k',lw=3)
	plt.errorbar(logssfr_sul,snr_sul,yerr=snr_err_sul,fmt='o',label='Sullivan et al. (2006)')
	plt.errorbar(logssfr_mat,snr_mat,yerr=snr_err_mat,fmt='x',label='Smith et al. (2012)')
	if combined_plot == False:
		plt.legend(frameon=False, loc=2, fontsize=17)
		plt.savefig(root_dir + 'Plots/model_' + model_name + '.pdf')

	

def plot_combined(root_dir, model_names, thetas, snr_funcs, logornot):
	logssfr_sul, snr_sul, snr_err_sul, logssfr_mat, snr_mat, snr_err_mat = read_data_names()

	n_plots = len(model_names)
	snr = np.zeros((n_plots,10000))
	for ii in np.arange(n_plots):
		if logornot[ii] == True:
			ssfr_values = np.logspace(-13,-8,10000)
			snr_func = snr_funcs[ii]
			snr[ii,:] = snr_func(ssfr_values, thetas[ii])
		else:
			logssfr_values = np.linspace(-13,-8,10000)
			snr_func = snr_funcs[ii]
			snr[ii,:] = snr_func(logssfr_values, thetas[ii])

	logssfr_values = np.linspace(-13,-8,10000)

	plt.figure()
	ax = plt.subplot()
	plt.xlabel('log(sSFR)',size='large')
	plt.ylabel('sSNR',size='large')
	plt.xlim((-13,-8))
	plt.ylim((2e-14,1e-12))
	ax.set_yscale("log")
	plt.errorbar(logssfr_sul,snr_sul,yerr=snr_err_sul,fmt='o',label='Sullivan et al. (2006)')
	plt.errorbar(logssfr_mat,snr_mat,yerr=snr_err_mat,fmt='x',label='Smith et al. (2012)')

	for ii in np.arange(n_plots):
		plt.plot(logssfr_values, snr[ii],c='k',lw=3,label=model_names[ii])
	

	plt.legend(frameon=False, loc=2, fontsize=16)

def ks_test(ssfr,snr,snr_func,theta,visualise=False,model_name='test',plot_color='r'):

	## First we make sure everything is sorted
	snr = snr[np.argsort(ssfr)]
	ssfr = np.sort(ssfr)

	## Compute model snr values
	snr_model = snr_func(ssfr,theta)
	ks_data = np.cumsum(snr) / np.sum(snr)
	ks_model = np.cumsum(snr_model) / np.sum(snr_model)
	
	ks_result = np.abs(np.max(ks_data - ks_model))

	if visualise:
		plt.figure()
		plt.ylim((0,1))
		plt.xlim((np.min(ssfr),np.max(ssfr)))
		plt.plot(ssfr,ks_model,ls='--',color=plot_color)
		plt.plot(ssfr,ks_data,ls='-',color=plot_color,label=model_name)
		ks_index = np.argmax(np.abs(ks_data - ks_model))
		plt.axvline(ssfr[ks_index],ymin=np.min((ks_model[ks_index],ks_data[ks_index])),ymax=np.max((ks_model[ks_index],ks_data[ks_index])),color='k',lw=3)
	return ks_result









