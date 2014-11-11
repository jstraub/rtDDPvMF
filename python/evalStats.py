import numpy as np
import matplotlib.pyplot as plt

with open('./statsTest_ddp.log','r') as fin:
  log = fin.readlines()
Kmax = 40

import ipdb
Ns = np.zeros((len(log),Kmax))
Ks = np.zeros(len(log),dtype=np.int)
Costs = np.zeros(len(log))
for i,line in enumerate(log):
  n = np.fromstring(line,sep=' ')
  Ks[i] = int(n[0])
  Costs[i] = n[1]
  for k in range(Ks[i]):
    Ns[i,int(n[k+2+Ks[i]])] = n[k+2]
    
Ns = Ns[:,(Ns.sum(axis=0) > 0)]
csumNs = Ns.cumsum(axis=1)
csumNs = np.c_[np.zeros((Ns.shape[0],1)),csumNs]

#ipdb.set_trace()
from js.utils.plot.colors import colorScheme

fig = plt.figure()
for i in range(1,csumNs.shape[1]):
  plt.fill_between(np.arange(Ns.shape[0]),csumNs[:,i-1],csumNs[:,i], \
      alpha=0.7,color=colorScheme('label')[(i-1)%7],lw=0)
#  plt.plot(np.arange(Ns.shape[0]),csumNs[:,i],color=colorScheme('label')[i%7],lw=4.)
plt.plot(np.arange(Ns.shape[0]),csumNs[:,-1],color=(0,0,0),lw=2.)
plt.show()
