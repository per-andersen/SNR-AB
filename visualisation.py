import utility_functions as util
import numpy as np
import matplotlib.pyplot as plt
import MCMC_nicelog as nice
import MCMC_simple as simple
import MCMC_piecewise as piece
import MCMC_abnewcontin as contin

root_dir = '/Users/perandersen/Data/SNR-AB/'


# theta_simple = simple.run_emcee()
# util.plot_combined(root_dir, ['nicelog','simple'], [theta_nice,theta_simple], \
# [nice.nicelog_snr,simple.simple_snr], [False,False])
# plt.show()

def plot_column_three(thetas, oplot_ab=True):
    theta_contin = contin.run_grid()
    theta_nice = nice.run_grid()
    theta_piece = piece.run_grid()

    thetas = [theta_contin, theta_piece, theta_nice]

    f, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(5, 7))

    if oplot_ab:
        theta_ab = simple.run_emcee()
        logssfr_values_ab = np.linspace(-13, -8, 100000)
        snr_values_ab = simple.simple_snr(logssfr_values_ab, theta_ab)
    plt.sca(ax1)
    util.plot_data_log(root_dir, 'contin', thetas[0], contin.new_contin_snr, combined_plot=True)
    if oplot_ab:
        plt.plot(logssfr_values_ab, snr_values_ab, c='r', lw=3, ls='--', alpha=0.5)
    ax1.set_yscale("log")
    ax1.set_ylabel(r'sSNR', size='x-large')
    ax1.set_xticks([0.])
    ax1.set_xticklabels([''])
    ax1.text(-12.7, 3e-13, '(a) Piecewise', size='x-large')
    plt.legend(frameon=False, loc=2, fontsize=13)

    plt.sca(ax2)
    util.plot_data_log(root_dir, 'piece', thetas[1], piece.piecewise_snr, combined_plot=True)
    if oplot_ab:
        plt.plot(logssfr_values_ab, snr_values_ab, c='r', lw=3, ls='--', alpha=0.5)
    ax2.set_yscale("log")
    ax2.set_ylabel(r'sSNR', size='x-large')
    ax2.set_xticks([0.])
    ax2.set_xticklabels([''])
    ax2.text(-12.7, 1e-12, '(b) Modified piecewise', size='x-large')

    plt.sca(ax3)
    util.plot_data(root_dir, 'nicelog', thetas[2], nice.nicelog_snr, combined_plot=True)
    if oplot_ab:
        plt.plot(logssfr_values_ab, snr_values_ab, c='r', lw=3, ls='--', alpha=0.5)
    ax3.set_yscale("log")
    ax3.set_ylabel(r'sSNR', size='x-large')
    ax3.set_xlabel(r'log(sSFR)', size='x-large')
    ax3.text(-12.7, 1e-12, '(c) Smooth logarithm', size='x-large')

    plt.subplots_adjust(left=0.16, bottom=0.08, right=0.96, top=0.98, wspace=0., hspace=0.)
    plt.savefig(root_dir + 'Plots/columnplot.pdf', format='pdf')


def plot_column_two(oplot_ab=True):
    theta_contin = contin.run_grid()
    theta_nice = nice.run_grid()
    theta_piece = piece.run_grid()

    thetas = [theta_contin, theta_piece, theta_nice]

    f, (ax1, ax2) = plt.subplots(2, 1, figsize=(5, 7))

    if oplot_ab:
        theta_ab = simple.run_emcee()
        logssfr_values_ab = np.linspace(-13, -8, 100000)
        snr_values_ab = simple.simple_snr(logssfr_values_ab, theta_ab)

    plt.sca(ax1)
    util.plot_data_log(root_dir, 'piece', thetas[1], piece.piecewise_snr, combined_plot=True)
    if oplot_ab:
        plt.plot(logssfr_values_ab, snr_values_ab, c='r', lw=3, ls='--', alpha=0.5)
    ax1.set_yscale("log")
    ax1.set_ylabel(r'sSNR', size='x-large')
    ax1.set_xticks([0.])
    ax1.set_xticklabels([''])
    ax1.text(-12.7, 6e-13, '(a) Piecewise', size='x-large')
    plt.legend(frameon=False, loc=2, fontsize=13)

    plt.sca(ax2)
    util.plot_data(root_dir, 'nicelog', thetas[2], nice.nicelog_snr, combined_plot=True)
    if oplot_ab:
        plt.plot(logssfr_values_ab, snr_values_ab, c='r', lw=3, ls='--', alpha=0.5)
    ax2.set_yscale("log")
    ax2.set_ylabel(r'sSNR', size='x-large')
    ax2.set_xlabel(r'log(sSFR)', size='x-large')
    ax2.text(-12.7, 1e-12, '(b) Smooth logarithm', size='x-large')

    plt.subplots_adjust(left=0.16, bottom=0.08, right=0.96, top=0.98, wspace=0., hspace=0.)
    plt.savefig(root_dir + 'Plots/columnplottwo.pdf', format='pdf')


def plot_bootstrap():
    parameters = np.genfromtxt(root_dir + 'Bootstrap/bootstrap_parameters.txt')
    print np.shape(parameters)

    par1 = parameters[:, 0]
    par2 = parameters[:, 1]
    par3 = parameters[:, 2]
    par4 = parameters[:, 3]

    plt.figure()
    plt.hist(par1, bins=10)
    plt.title('a')

    plt.figure()
    plt.hist(par2, bins=10)
    plt.title('k')

    plt.figure()
    plt.hist(par3, bins=10)
    plt.title('ssfr0')

    plt.figure()
    plt.hist(par4, bins=10)
    plt.title('alpha')


def plot_iilustris():
    f, ax = plt.subplots()
    ax.set_yscale("log")

    # theta_piece = piece.run_grid()
    theta_piece = 0.58585858585858586, 1.1905050505050503e-07, 1.0060606060606059e-11, 1.0373737373737374e-09

    ssfr_values = np.logspace(-13, -8, 10000)
    snr_piecewise = piece.piecewise_snr(ssfr_values, theta_piece)
    util.plot_data_log(root_dir, 'piece', theta_piece, piece.piecewise_snr, combined_plot=True)

    # The SUDARE results
    logssfr_sudare = np.array([-12.2, -10.5, -9.7, -9.])
    ssnr_sudare = np.array([0.5, 1.2, 3.2, 6.5]) * 1e-13
    sudare_ssnr_uncertainty = np.array([[0.28, 0.4, 1.1, 1.9], [0.33, 0.5, 1.1, 2.2]]) * 1e-13
    plt.errorbar(logssfr_sudare, ssnr_sudare, yerr=sudare_ssnr_uncertainty, fmt='s', ls='', c='g', label='SUDARE')

    # The Mannucci results
    # !!! THERE IS DISAGREEMENT BETWEEN SMITH 2012 AND BOTTICELLA 2017 ON THE VALUES OF MANNUCCI SO WE DO NOT INCLUDE THEM
    # logssfr_mannucci = np.array([-11.9, -10.45, -9.6, -8.6])
    # ssnr_mannucci = np.array([2.7e-14,8.4e-14,1.9e-13,4.5e-13])
    # mannucci_ssnr_uncertainty = np.array([[1.7e-14,2.6e-14,8e-14,4.5e-13],[1.3e-14,2.5e-14,7e-14,2.5e-13]])
    # plt.errorbar(logssfr_mannucci,ssnr_mannucci,yerr=mannucci_ssnr_uncertainty,fmt='v',ls='',c='darkblue',label='Mannucci et al. (2005)')

    # Graur 2015 results
    ssfr_graur = np.array([18, 70, 200, 0.39, 0.81, 2.2]) * 1e-12
    logssfr_graur = np.log10(ssfr_graur)
    ssnr_graur = np.array([0.3, 0.15, 0.076, 0.091, 0.126, 0.056]) * 1e-12
    # plt.plot(logssfr_graur,ssnr_graur,'cv',label='Graur et al. (2015)')

    steller_mass_loss = True

    ssnr_all, ssfr_all, times_all = util.get_illustris_ssfr_ssnr(steller_mass_loss=steller_mass_loss)

    plt.scatter((ssfr_all), (ssnr_all), c=times_all, cmap='Reds', vmin=0., vmax=13.5)
    plt.xlim((-13, -8))
    plt.ylim((2e-14, 2e-12))
    ax.set_ylabel(r'sSNR', size='x-large')
    ax.set_xlabel(r'log(sSFR)', size='x-large')
    plt.colorbar(ticks=[0, 2, 4, 6, 8, 10, 12, 13.5]).set_label('Age [Gyr]', size='x-large')
    plt.subplots_adjust(left=0.12, bottom=0.12, right=0.98, top=0.98)
    plt.legend(frameon=False, loc=2, fontsize=13)

    if steller_mass_loss:
        plt.savefig(root_dir + 'Plots/illustris_sml.pdf', format='pdf')
    else:
        plt.savefig(root_dir + 'Plots/illustris.pdf', format='pdf')


# plot_column_two()
# plot_bootstrap()
#plot_iilustris()

plt.show()
