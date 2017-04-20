import numpy as np
import matplotlib.pyplot as plt

gp13 = np.genfromtxt('GP13.txt')
m05 = np.genfromtxt('M05.txt')
s06 = np.genfromtxt('S06.txt')
sm12 = np.genfromtxt('Sm12.txt') 

#print gp13
#print m05
#print s06
print sm12

ax = plt.subplot()
ax.set_yscale("log")
plt.errorbar(gp13[:,0],gp13[:,1],yerr=[gp13[:,2],gp13[:,3]])
plt.errorbar(m05[:,0],m05[:,1],yerr=[m05[:,2],m05[:,3]])
plt.errorbar(s06[:,0],s06[:,1],yerr=2e-14)
plt.errorbar(sm12[:,0],sm12[:,1],yerr=2e-14)
plt.show()

