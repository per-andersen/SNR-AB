import numpy as np
import matplotlib.pyplot as plt


mannucci = np.genfromtxt('Mathew/Smith_2012_Figure5_Mannucci_Results.txt')
mathew = np.genfromtxt('Mathew/Smith_2012_Figure5_Results.txt')
sullivan = np.genfromtxt('Mathew/Smith_2012_Figure5_Sullivan_Results.txt')
data_points_manucci = np.genfromtxt('GP13/mannuccipnts.dat')

x_points_manucci = data_points_manucci[:,0]
y_points_manucci = data_points_manucci[:,1]
#print x_points_manucci
#print y_points_manucci

ssfr_man = mannucci[:,0]
snr_man = mannucci[:,1]
snr_err_upp_man = mannucci[:,2]
snr_err_low_man = mannucci[:,3]

ssfr_mat = mathew[:,0]
snr_mat = mathew[:,1]
snr_err_upp_mat = mathew[:,2]
snr_err_low_mat = mathew[:,3]

ssfr_sul = sullivan[:,0]
snr_sul = sullivan[:,1]
snr_err_sul = sullivan[:,2]


ax = plt.subplot()
#plt.xlim((-13,-8))
plt.ylim((0.5*10e-15,0.5*10e-12))
#plt.errorbar(ssfr_man, snr_man, yerr=(snr_err_low_man, snr_err_upp_man),fmt='o',label='Manucci')
plt.errorbar(10.**ssfr_mat, snr_mat, yerr=(snr_err_low_mat, snr_err_upp_mat),fmt='o',label='Smith')
plt.errorbar(10.**ssfr_sul, snr_sul, yerr=(snr_err_sul),fmt='o',label='Sullivan')
#plt.plot(x_points_manucci, 10**y_points_manucci,'s')
ax.set_yscale("log")
ax.set_xscale("log")
plt.legend(loc=2)
plt.show()


#print np.log10(snr_sul-snr_err_sul)

#print np.log10(snr_mat + snr_err_upp_mat)
#print np.log10(snr_mat - snr_err_low_mat)
#print np.log10(snr_mat) - np.log10(snr_mat - snr_err_low_mat)
#print np.log10(snr_mat) - np.log10(snr_mat + snr_err_upp_mat)
#print ""
#print np.log10(snr_sul - snr_err_sul)
#print np.log10(snr_sul + snr_err_sul)
#print np.log10(snr_sul) - np.log10(snr_sul - snr_err_sul)
#print np.log10(snr_sul) - np.log10(snr_sul + snr_err_sul)

def abnew_time(times):
	
	k1 = 7e-13
	k2 = 4.2e-4
	t1 = 1.0e9
	t2 = 10.0e9
	ta = 12e9
	snr_return = np.zeros(len(times))
	for ii in np.arange(len(times)):
		t = times[ii]
		if t < t1:
			snr_return[ii] = k1
		elif t > t2:
			snr_return[ii] =  k2 / ta
		else:
			snr_return[ii] =  (k1*t1 + k2*np.log(t2/t1)) / t
	return snr_return

def abnew_ssfr(ssfr):
	
	k1 = 7e-13
	k2 = 4.2e-4
	t1 = 1.0e9
	t2 = 10.0e9
	ta = 12e9
	times = 1. / ssfr

	snr_return = np.zeros(len(times))

	for ii in np.arange(len(times)):
		t = times[ii]
		if t < t1:
			snr_return[ii] = k1
		elif t > t2:
			snr_return[ii] =  k2 / ta
		else:
			snr_return[ii] =  (k1*t1 + k2*np.log(t2/t1)) / t
	return snr_return


#plt.figure()
#ax = plt.subplot()
#ax.set_yscale("log")
#ax.set_xscale("log")
#times = np.logspace(8,11,1000)
#plt.plot(times,abnew_time(times))

#plt.figure()
#ax = plt.subplot()
#ax.set_yscale("log")
#ax.set_xscale("log")
#ssfr = np.logspace(-8,-11,1000)
#plt.plot(ssfr,abnew_ssfr(ssfr))

plt.show()