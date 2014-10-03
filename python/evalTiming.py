import numpy as np
import matplotlib.pyplot as plt


t = np.loadtxt('./timer.log')
x = np.loadtxt('./stats.log')
print t.shape

plt.figure()
plt.subplot(2,1,1)
plt.plot(np.arange(t.shape[0]),t[:,0],label='preparation')
plt.plot(np.arange(t.shape[0]),t[:,1],label='inference')
plt.plot(np.arange(t.shape[0]),t[:,0:2].sum(axis=1),label='inference+prep')
plt.plot(np.arange(t.shape[0]),t[:,2],label='total')
plt.plot(np.arange(t.shape[0]),np.ones(t.shape[0])*30,'b--',label='real-time')
plt.legend()
plt.subplot(2,1,2)
plt.plot(np.arange(x.shape[0]),x[:,0],label='N')
plt.plot(np.arange(x.shape[0]),x[:,1],label='residual')

plt.legend()
plt.show()
