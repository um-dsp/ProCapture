from email.charset import add_charset
import numpy as np
import matplotlib.pyplot as plt

            
begnign = np.loadtxt('./hamilton_indexes/begnign.csv')
adv = np.loadtxt('./hamilton_indexes/adv.csv' )


begnign = np.average(begnign, axis=1)
adv = np.average(adv,axis=1)

fig = plt.figure()
ax1 = fig.add_subplot(111)



axis = np.arange(0,len(adv))
ax1.scatter(axis, adv, s=10, c='b', marker="s", label='adv')

axis = np.arange(len(adv),len(begnign)+ len(adv))
ax1.scatter(axis,begnign, s=10, c='r', marker="o", label='begnign')

plt.show()
