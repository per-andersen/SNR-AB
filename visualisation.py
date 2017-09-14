import utility_functions as util
import numpy as np
import matplotlib.pyplot as plt
import MCMC_nicelog as nice
import MCMC_simple as simple
import MCMC_piecewise as piece
import MCMC_abnewcontin as contin

root_dir = '/Users/perandersen/Data/SNR-AB/'

#theta_simple = simple.run_emcee()
#util.plot_combined(root_dir, ['nicelog','simple'], [theta_nice,theta_simple], [nice.nicelog_snr,simple.simple_snr], [False,False])
#plt.show()

def plot_column_three(thetas,oplot_ab=True):
	theta_contin = contin.run_grid()
	theta_nice = nice.run_grid()
	theta_piece = piece.run_grid()

	thetas = [theta_contin,theta_piece,theta_nice]

	f, (ax1, ax2, ax3) = plt.subplots(3,1,figsize=(5,7))

	if oplot_ab:
		theta_ab = simple.run_emcee()
		logssfr_values_ab = np.linspace(-13,-8,100000)
		snr_values_ab = simple.simple_snr(logssfr_values_ab, theta_ab)
	plt.sca(ax1)
	util.plot_data_log(root_dir, 'contin', thetas[0], contin.new_contin_snr,combined_plot=True)
	if oplot_ab:
		plt.plot(logssfr_values_ab, snr_values_ab,c='r',lw=3,ls='--',alpha=0.5)
	ax1.set_yscale("log")
	ax1.set_ylabel(r'sSNR',size='x-large')
	ax1.set_xticks([0.])
	ax1.set_xticklabels([''])
	ax1.text(-12.7,3e-13,'(a) Piecewise',size='x-large')
	plt.legend(frameon=False, loc=2, fontsize=13)
	
	plt.sca(ax2)
	util.plot_data_log(root_dir, 'piece', thetas[1], piece.piecewise_snr,combined_plot=True)
	if oplot_ab:
		plt.plot(logssfr_values_ab, snr_values_ab,c='r',lw=3,ls='--',alpha=0.5)
	ax2.set_yscale("log")
	ax2.set_ylabel(r'sSNR',size='x-large')
	ax2.set_xticks([0.])
	ax2.set_xticklabels([''])
	ax2.text(-12.7,1e-12,'(b) Modified piecewise',size='x-large')

	plt.sca(ax3)
	util.plot_data(root_dir, 'nicelog', thetas[2], nice.nicelog_snr,combined_plot=True)
	if oplot_ab:
		plt.plot(logssfr_values_ab, snr_values_ab,c='r',lw=3,ls='--',alpha=0.5)
	ax3.set_yscale("log")
	ax3.set_ylabel(r'sSNR',size='x-large')
	ax3.set_xlabel(r'log(sSFR)',size='x-large')
	ax3.text(-12.7,1e-12,'(c) Smooth logarithm',size='x-large')

	plt.subplots_adjust(left=0.16,bottom=0.08,right=0.96,top=0.98,wspace=0., hspace=0.)
	plt.savefig(root_dir + 'Plots/columnplot.pdf',format='pdf')


def plot_column_two(thetas,oplot_ab=True):
	theta_contin = contin.run_grid()
	theta_nice = nice.run_grid()
	theta_piece = piece.run_grid()

	thetas = [theta_contin,theta_piece,theta_nice]

	f, (ax1, ax2) = plt.subplots(2,1,figsize=(5,7))

	if oplot_ab:
		theta_ab = simple.run_emcee()
		logssfr_values_ab = np.linspace(-13,-8,100000)
		snr_values_ab = simple.simple_snr(logssfr_values_ab, theta_ab)
	
	plt.sca(ax1)
	util.plot_data_log(root_dir, 'piece', thetas[1], piece.piecewise_snr,combined_plot=True)
	if oplot_ab:
		plt.plot(logssfr_values_ab, snr_values_ab,c='r',lw=3,ls='--',alpha=0.5)
	ax1.set_yscale("log")
	ax1.set_ylabel(r'sSNR',size='x-large')
	ax1.set_xticks([0.])
	ax1.set_xticklabels([''])
	ax1.text(-12.7,6e-13,'(a) Piecewise',size='x-large')
	plt.legend(frameon=False, loc=2, fontsize=13)

	plt.sca(ax2)
	util.plot_data(root_dir, 'nicelog', thetas[2], nice.nicelog_snr,combined_plot=True)
	if oplot_ab:
		plt.plot(logssfr_values_ab, snr_values_ab,c='r',lw=3,ls='--',alpha=0.5)
	ax2.set_yscale("log")
	ax2.set_ylabel(r'sSNR',size='x-large')
	ax2.set_xlabel(r'log(sSFR)',size='x-large')
	ax2.text(-12.7,1e-12,'(b) Smooth logarithm',size='x-large')

	plt.subplots_adjust(left=0.16,bottom=0.08,right=0.96,top=0.98,wspace=0., hspace=0.)
	plt.savefig(root_dir + 'Plots/columnplottwo.pdf',format='pdf')

def plot_bootstrap():
	parameters = np.genfromtxt(root_dir + 'Bootstrap/bootstrap_parameters_4.txt')
	
	par1 = parameters[:,0]
	par2 = parameters[:,1]
	par3 = parameters[:,2]
	par4 = parameters[:,3]

	plt.figure()
	plt.hist(par1,bins=20)
	plt.title('a')

	plt.figure()
	plt.hist(par2,bins=20)
	plt.title('k')

	plt.figure()
	plt.hist(par3,bins=20)
	plt.title('ssfr0')

	plt.figure()
	plt.hist(par4,bins=20)
	plt.title('alpha')

#plot_column_two(thetas)
plot_bootstrap()

plt.show()