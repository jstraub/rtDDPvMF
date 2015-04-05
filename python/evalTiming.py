# Copyright (c) 2015, Julian Straub <jstraub@csail.mit.edu> Licensed
# under the MIT license. See the license file LICENSE.
import numpy as np
import matplotlib.pyplot as plt

def plotTiming(path,name):
  t = np.loadtxt(path)
  #x = np.loadtxt('./stats.log')
  print '--- {}; mean +- 1std'.format(name)
  print 'mean {} +- {}'.format(t[:,0].mean(axis=0),t[:,0].std(axis=0))

  fig = plt.figure()
  plt.subplot(1,1,1)
  plt.plot(np.arange(t.shape[0]),t[:,0],label='total')
  #plt.plot(np.arange(t.shape[0]),t[:,1],label='inference')
  #plt.plot(np.arange(t.shape[0]),t[:,0:2].sum(axis=1),label='inference+prep')
  plt.plot(np.arange(t.shape[0]),t[:,1],label='first iter')
  plt.plot(np.arange(t.shape[0]),np.ones(t.shape[0])*30,'b--',label='real-time')
  plt.legend()
  plt.title(name)
  #plt.subplot(2,1,2)
  #plt.plot(np.arange(x.shape[0]),x[:,0],label='N')
  #plt.plot(np.arange(x.shape[0]),x[:,1],label='residual')
  return fig

# /scratch/xtion/dpMMlowVar
# /home/jstraub/workspace/rtDDPvMF/results

path = './dp_fbf/timer.log'
plotTiming(path,'dp_fbf')
path = './dp/timer.log'
plotTiming(path,'dp')
path = './ddp/timer.log'
plotTiming(path,'ddp')
path = './spkm/timer.log'
plotTiming(path,'spkm')

#path = '/scratch/xtion/dpMMlowVar/timer.log'
#plotTiming(path)

plt.show()
