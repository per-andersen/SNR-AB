import numpy as np
import matplotlib.pyplot as plt

data_points = np.genfromtxt('fort.25')
data_line = np.genfromtxt('fort.28')

data_line_manucci = np.genfromtxt('fort.29')
data_points_manucci = np.genfromtxt('mannuccipnts.dat')

print np.shape(data_line)

x_points = data_points[:,0]
y_points = data_points[:,1] + np.log10(2)
y_points_upper = data_points[:,2] + np.log10(2)
y_points_lower = data_points[:,3] + np.log10(2)

x_line = data_line[:,0]
y_line = data_line[:,1]

x_line_manucci = data_line_manucci[:,0]
y_line_manucci = data_line_manucci[:,1]

x_points_manucci = data_points_manucci[:,0]
y_points_manucci = data_points_manucci[:,1]


ax = plt.subplot()
plt.xlabel(r'log[sSFR]')
plt.ylabel(r'log[sSNR]')
plt.xlim((-12.5,-8))
plt.ylim((0.5*10e-15,0.5*10e-12))
#plt.ylim((-15,-11.75))
#plt.errorbar(x_points,y_points,fmt='o',yerr=[y_points-y_points_upper, y_points_lower-y_points])
plt.plot(x_points_manucci, 10**y_points_manucci,'s')
#plt.plot(x_line,y_line + np.log10(2),'-')
#plt.plot(x_line_manucci,y_line_manucci,'--')
ax.set_yscale("log")
plt.plot()
plt.show()