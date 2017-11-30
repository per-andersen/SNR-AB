import os
import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import quad
import matplotlib.pyplot as plt
import pickle as pick
from copy import deepcopy
from Papaya import ReadPapayaHDF5
import time


def do_chains_exist(model_name, root_dir):
    if os.path.isfile(root_dir + 'Data/MCMC_' + model_name + '.pkl'):
        return True
    return False


def read_chains(model_name, root_dir):
    pkl_data_file = open(root_dir + 'Data/MCMC_' + model_name + '.pkl', 'rb')
    samples = pick.load(pkl_data_file)
    pkl_data_file.close()
    return samples


def does_grid_exist(model_name, root_dir):
    if os.path.isfile(root_dir + 'Data/MCMC_' + model_name + '_grid.pkl'):
        return True
    return False


def read_grid(model_name, root_dir):
    pkl_data_file = open(root_dir + 'Data/MCMC_' + model_name + '_grid.pkl', 'rb')
    resolution, likelihoods, parameters, theta_max = pick.load(pkl_data_file)
    pkl_data_file.close()
    return resolution, likelihoods, parameters, theta_max


def read_data_names():
    mathew = np.genfromtxt('Mathew/Smith_2012_Figure5_Results.txt')
    sullivan = np.genfromtxt('Mathew/Smith_2012_Figure5_Sullivan_Results.txt')
    mathew = np.delete(mathew, 1, axis=0)  # !!! Justify this

    logssfr_mat = mathew[:, 0]
    snr_mat = mathew[:, 1]
    snr_err_upp_mat = mathew[:, 2]
    snr_err_low_mat = mathew[:, 3]
    snr_err_mat = np.sqrt(snr_err_low_mat ** 2 + snr_err_upp_mat ** 2)  # !!! Check this is ok

    logssfr_sul = sullivan[:, 0]
    snr_sul = sullivan[:, 1]
    snr_err_sul = sullivan[:, 2]

    return logssfr_sul, snr_sul, snr_err_sul, logssfr_mat, snr_mat, snr_err_mat


def read_data():
    logssfr_sul, snr_sul, snr_err_sul, logssfr_mat, snr_mat, snr_err_mat = read_data_names()

    logssfr = np.concatenate((logssfr_mat, logssfr_sul))
    ssfr = 10 ** logssfr
    snr = np.concatenate((snr_mat, snr_sul))
    snr_err = np.concatenate((snr_err_mat, snr_err_sul))

    return ssfr, snr, snr_err


def read_data_with_log():
    logssfr_sul, snr_sul, snr_err_sul, logssfr_mat, snr_mat, snr_err_mat = read_data_names()

    logssfr = np.concatenate((logssfr_mat, logssfr_sul))
    ssfr = 10 ** logssfr
    snr = np.concatenate((snr_mat, snr_sul))
    snr_err = np.concatenate((snr_err_mat, snr_err_sul))

    return logssfr, ssfr, snr, snr_err


def plot_data_log(root_dir, model_name, theta, snr_func, combined_plot=False):
    logssfr_sul, snr_sul, snr_err_sul, logssfr_mat, snr_mat, snr_err_mat = read_data_names()

    ssfr_values = np.logspace(-13, -8, 10000)
    snr_values = snr_func(ssfr_values, theta)
    if not combined_plot:
        plt.figure()
        ax = plt.subplot()
        plt.xlabel('log(sSFR)', size='large')
        plt.ylabel('sSNR', size='large')
        ax.set_yscale("log")
    plt.xlim((-13, -8))
    plt.ylim((2e-14, 2e-12))
    plt.plot(np.log10(ssfr_values), snr_values, c='k', lw=3)
    plt.errorbar(logssfr_sul, snr_sul, yerr=snr_err_sul, fmt='o', label='Sullivan et al. (2006)')
    plt.errorbar(logssfr_mat, snr_mat, yerr=snr_err_mat, fmt='x', label='Smith et al. (2012)')
    if not combined_plot:
        plt.legend(frameon=False, loc=2, fontsize=16)
        plt.savefig(root_dir + 'Plots/model_' + model_name + '.pdf')


def plot_data(root_dir, model_name, theta, snr_func, combined_plot=False):
    logssfr_sul, snr_sul, snr_err_sul, logssfr_mat, snr_mat, snr_err_mat = read_data_names()

    logssfr_values = np.linspace(-13, -8, 100000)
    snr_values = snr_func(logssfr_values, theta)
    if not combined_plot:
        plt.figure()
        ax = plt.subplot()
        plt.xlabel('log(sSFR)', size='large')
        plt.ylabel('sSNR', size='large')
        ax.set_yscale("log")
    plt.xlim((-13, -8))
    plt.ylim((2e-14, 2e-12))
    plt.plot(logssfr_values, snr_values, c='k', lw=3)
    plt.errorbar(logssfr_sul, snr_sul, yerr=snr_err_sul, fmt='o', label='Sullivan et al. (2006)')
    plt.errorbar(logssfr_mat, snr_mat, yerr=snr_err_mat, fmt='x', label='Smith et al. (2012)')
    if not combined_plot:
        plt.legend(frameon=False, loc=2, fontsize=17)
        plt.savefig(root_dir + 'Plots/model_' + model_name + '.pdf')


def plot_combined(model_names, thetas, snr_funcs, logornot):
    logssfr_sul, snr_sul, snr_err_sul, logssfr_mat, snr_mat, snr_err_mat = read_data_names()

    n_plots = len(model_names)
    snr = np.zeros((n_plots, 10000))
    for ii in np.arange(n_plots):
        if logornot[ii]:
            ssfr_values = np.logspace(-13, -8, 10000)
            snr_func = snr_funcs[ii]
            snr[ii, :] = snr_func(ssfr_values, thetas[ii])
        else:
            logssfr_values = np.linspace(-13, -8, 10000)
            snr_func = snr_funcs[ii]
            snr[ii, :] = snr_func(logssfr_values, thetas[ii])

    logssfr_values = np.linspace(-13, -8, 10000)

    plt.figure()
    ax = plt.subplot()
    plt.xlabel('log(sSFR)', size='large')
    plt.ylabel('sSNR', size='large')
    plt.xlim((-13, -8))
    plt.ylim((2e-14, 1e-12))
    ax.set_yscale("log")
    plt.errorbar(logssfr_sul, snr_sul, yerr=snr_err_sul, fmt='o', label='Sullivan et al. (2006)')
    plt.errorbar(logssfr_mat, snr_mat, yerr=snr_err_mat, fmt='x', label='Smith et al. (2012)')

    for ii in np.arange(n_plots):
        plt.plot(logssfr_values, snr[ii], c='k', lw=3, label=model_names[ii])

    plt.legend(frameon=False, loc=2, fontsize=16)


def ks_test(ssfr, snr, snr_func, theta, visualise=False, model_name='test', plot_color='r'):
    # First we make sure everything is sorted
    snr = snr[np.argsort(ssfr)]
    ssfr = np.sort(ssfr)

    # Compute model snr values
    snr_model = snr_func(ssfr, theta)
    ks_data = np.cumsum(snr) / np.sum(snr)
    ks_model = np.cumsum(snr_model) / np.sum(snr_model)

    ks_result = np.abs(np.max(ks_data - ks_model))

    if visualise:
        plt.figure()
        plt.ylim((0, 1))
        plt.xlim((np.min(ssfr), np.max(ssfr)))
        plt.plot(ssfr, ks_model, ls='--', color=plot_color)
        plt.plot(ssfr, ks_data, ls='-', color=plot_color, label=model_name)
        ks_index = np.argmax(np.abs(ks_data - ks_model))
        plt.axvline(ssfr[ks_index], ymin=np.min((ks_model[ks_index], ks_data[ks_index])),
                    ymax=np.max((ks_model[ks_index], ks_data[ks_index])), color='k', lw=3)
    return ks_result


def numerical_derivative(func, step_size, base_parameters, parameter_number):
    right_parameters = deepcopy(base_parameters)
    right_parameters[parameter_number] += step_size
    right_step = func(right_parameters)

    left_parameters = deepcopy(base_parameters)
    left_parameters[parameter_number] -= step_size
    left_step = func(left_parameters)

    return (right_step - left_step) / (2. * step_size)


def dtd(tau):
    if tau < 0.964e9:
        return 6.48e-13
    else:
        return 4.0e-4 * tau ** (-1)


def remaining_stellar_mass_fraction(tt):
    if tt > 0.:
        return 1. - 0.05 * np.log(1. + tt / 1.4e6)
    else:
        raise ValueError("Time was negative in remaining_stellar_mass_fraction()!")


def get_illustris_mass_sfr_snr(steller_mass_loss, root_dir='/Users/perandersen/Data/SNR-AB/', ii=0, tt=13e9):
    data, header = ReadPapayaHDF5(root_dir + 'SFH160607A/SFR_Subhalo_%5.5d.hdf5' % ii)

    # Data['time'][0] = 0.
    # times = 14.0e9-Data['time'][::-1]
    # Data['SFR'] = Data['SFR'][::-1]

    data['time'][0] = 0.
    times = data['time'][::-1]
    data['SFR'] = data['SFR']  # [::-1]

    # plt.figure()
    # plt.plot(times,Data['SFR'])
    # plt.show()

    sfr_func = interp1d(times, data['SFR'])

    delaytimes = np.zeros(len(times))
    sfr_interpolated = sfr_func(times)

    convolution = lambda tau, time: dtd(tau) * sfr_func(time - tau)

    if steller_mass_loss:
        mass_func = lambda tau, time: sfr_func(tau) * remaining_stellar_mass_fraction(time-tau)
        mass = quad(mass_func, 0., tt, args=tt, limit=1000, full_output=1)[0]
    else:
        mass = quad(sfr_func, 0., tt, limit=1000, full_output=1)[0]

        # print mass

    snr = quad(convolution, 0., tt, args=tt, limit=1000, full_output=1)[0]

    # print 'i:',ii
    # print 'mass', mass
    # print 'sSNR', snr/mass
    # print 'log(sSFR)', np.log10(sfr_func(times[-1])/mass)
    # print ''

    return mass, sfr_func(tt), snr


def get_illustris_time_series(steller_mass_loss, ntimes=18, tstart=1.5, tend=13.9, ii=0):
    masses = np.zeros(ntimes)
    sfrs = np.zeros(ntimes)
    snrs = np.zeros(ntimes)
    times = np.linspace(tstart, tend, ntimes)
    for jj, tt in np.ndenumerate(times):
        masses[jj], sfrs[jj], snrs[jj] = get_illustris_mass_sfr_snr(tt=tt * 1e9, ii=ii,
                                                                    steller_mass_loss=steller_mass_loss)

    return np.log10(sfrs / masses), snrs / masses, times


def get_illustris_ssfr_ssnr(root_dir='/Users/perandersen/Data/SNR-AB/', steller_mass_loss=True):
    ssfr_all = np.array([])
    ssnr_all = np.array([])
    times_all = np.array([])

    print
    if steller_mass_loss:
        output_file_name = 'illustris_ssnr_ssfr_sml.pkl'
    else:
        output_file_name = 'illustris_ssnr_ssfr.pkl'

    if os.path.isfile(root_dir + output_file_name):
        pkl_data_file = open(root_dir + output_file_name, 'rb')
        result = pick.load(pkl_data_file)
        ssnr_all, ssfr_all, times_all = result
    else:
        '''
        for ii in [342,563,565,745,747,1380,1392,1531,1535,1635,1765,1864,1865,1887,2067,\
                   2068,2072,2146,2147,2211,2276,2413,2469,2589,2698,2706,2741,2780,2877,\
                   2909,2950,2988,3065,3148,3191,3236,3275,3318,3344,3351,3598,3644,3694,
                   3729,3845,3996,4122,4203,4244,4378,4473,4570,4721,0,1,2,3,3000,4500,6000,\
                   9000,10000,11000,12000,13000,14000,15000,18000,19000,20000,21000,24000,\
                   25000,26000,27000,29275]:
        '''

        for ii in [342, 563, 565, 745, 747]:
            print ii
            ssfr, ssnr, times = get_illustris_time_series(ii=ii, steller_mass_loss=steller_mass_loss)
            ssfr_all = np.append(ssfr_all, ssfr)
            ssnr_all = np.append(ssnr_all, ssnr)
            times_all = np.append(times_all, times)
        output = open(root_dir + output_file_name, 'wb')
        result = ssnr_all, ssfr_all, times_all
        pick.dump(result, output)
        output.close()
    return ssnr_all, ssfr_all, times_all


def convolution_test():
    def sfr(t):
        if 1e9 < t < 1.1e9:
            return 1.
        else:
            return 0.

    times = np.linspace(0., 10., 1000) * 1e9
    sfr_t = np.zeros(len(times))
    for ii, tt in enumerate(times):
        sfr_t[ii] = sfr(tt)
        # plt.figure()
        # plt.plot(times,sfr_t)

        # plt.figure()
        # plt.plot(times,remaining_stellar_mass_fraction(times))

    conv = lambda tau, time: remaining_stellar_mass_fraction(time - tau) * sfr(tau)

    correction = quad(conv, 0., 10e9, args=[0.], limit=10000)[0]
    print correction


if __name__ == '__main__':
    t0 = time.time()
    get_illustris_ssfr_ssnr(steller_mass_loss=True)
    #convolution_test()

    print "Done in", time.time() - t0, "seconds"
    plt.show()
