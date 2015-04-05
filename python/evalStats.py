# Copyright (c) 2015, Julian Straub <jstraub@csail.mit.edu> Licensed
# under the MIT license. See the license file LICENSE.
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import subprocess as subp
                                                                                
#paper 
mpl.rc('font',size=18)
mpl.rc('lines',linewidth=4.)                                                    
figSize = (20., 7.)                 

def plotNperCluster(path,labels=True,ticks=True,frameMax=None):
  with open(path,'r') as fin:
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
  nFrame = Ns.shape[0]


  csumNs = Ns.cumsum(axis=1)
  csumNs = np.c_[np.zeros((Ns.shape[0],1)),csumNs]

  csumNs /= 640*480. # csumNs.max()
  csumNs *= 100.

  yMax = csumNs.max()
  if not frameMax is None:
    Ns = Ns[0:frameMax, :]
    csumNs = csumNs[0:frameMax,:]

  #ipdb.set_trace()
  from js.utils.plot.colors import colorScheme

  Kmax = max(10,csumNs.shape[1]) #15
  print Kmax
  col = plt.get_cmap('hsv')(np.linspace(0.,1.,Kmax))

  fig = plt.figure(figsize=figSize, dpi=80, facecolor='w', edgecolor='k')
  ax = plt.subplot(111)
  for i in range(1,csumNs.shape[1]):
    plt.fill_between(np.arange(Ns.shape[0]),csumNs[:,i-1],csumNs[:,i], \
        alpha=0.4,color=col[i-1],lw=0)
    plt.plot(np.arange(Ns.shape[0]),csumNs[:,i],color=(0,0,0),lw=1.)
  plt.plot(np.arange(Ns.shape[0]),csumNs[:,-1],color=(0,0,0),lw=2.)
  plt.xlim((0.,nFrame))
  plt.ylim((0.,yMax))
  if labels:
    plt.ylabel('% of normals')
    plt.xlabel('frame number')
#  tiks = np.floor(np.linspace(0,csumNs.shape[0],13))[1:-1:2]
#  import ipdb
#  ipdb.set_trace()
  tiks = np.floor(np.linspace(0,csumNs.shape[0],15))[1:-1:2]
  ax.set_xticks(tiks)
  if not ticks:
    ax.set_xticks([])
#  tik = ax.get_yticklabels()
#  for i in range(0,len(tik),2):
#    tik[i] = ''
#  ax.set_yticklabels(tik)
  ax.set_yticks(np.arange(10,100,50))
  ax.spines['top'].set_visible(False)
#  ax.spines['right'].set_visible(False)
  ax.get_xaxis().tick_bottom()
  ax.get_yaxis().tick_right()
  plt.subplots_adjust(bottom=0.8);
  return fig, tiks, nFrame

outPath = '/scratch/xtion/dpMMlowVar/2014-11-11-22-18-13/'
outPath = '/home/jstraub/workspace/writing/paper/jstraub_2015_aistats_DPvMFMM/figuresDDP/'
outPath = '/scratch/xtion/dpMMlowVar/results/'
outPath = '/scratch/xtion/dpMMlowVar/resultsVideo/'

path = '/scratch/xtion/dpMMlowVar/dp//stats.log'
fig1, tiks, nFrame = plotNperCluster(path,False,False)
#plt.savefig(outPath+'statsDPplot.pdf',figure=fig1) 
if True:
  for frNumber in range(nFrame):
    print frNumber
    fig1, tiks, _ = plotNperCluster(path,False,False,frNumber)
    plt.savefig(outPath+'statsDPplot_{:05}.png'.format(frNumber),figure=fig1) 
    plt.close(fig1)
  #  fig1.show()
  #  raw_input()

path = '/scratch/xtion/dpMMlowVar/ddp/stats.log'
fig2, tiks,nFrame = plotNperCluster(path,False,False)
#plt.savefig(outPath+'statsDDPplot.pdf',figure=fig2) 
for frNumber in range(nFrame):
  print frNumber
  fig2, tiks, _ = plotNperCluster(path,False,False,frNumber)
  plt.savefig(outPath+'statsDDPplot_{:05}.png'.format(frNumber),figure=fig2) 
  plt.close(fig2)

#plt.show()
fig1.show()
fig2.show()

raw_input('save figs to {}?'.format(outPath))
#fig = plotNperCluster(path,False,True)
tiks -= tiks%30
for t in tiks:
  for tt in [int(t), int(t)+1]:
    subp.call('cp /scratch/xtion/dpMMlowVar/dp/frame{:05d}.png {}/dp/'.format(tt,outPath),shell=True)
    subp.call('cp /scratch/xtion/dpMMlowVar/ddp/frame{:05d}.png {}/ddp/'.format(tt,outPath),shell=True)
    subp.call('cp /scratch/xtion/dpMMlowVar/dp_fbf/frame{:05d}.png {}/dp_fbf/'.format(tt,outPath),shell=True)

    subp.call('cp /scratch/xtion/dpMMlowVar/dp/frame{:05d}_lbl.png {}/dp/'.format(tt,outPath),shell=True)
    subp.call('cp /scratch/xtion/dpMMlowVar/ddp/frame{:05d}_lbl.png {}/ddp/'.format(tt,outPath),shell=True)
    subp.call('cp /scratch/xtion/dpMMlowVar/dp_fbf/frame{:05d}_lbl.png {}/dp_fbf/'.format(tt,outPath),shell=True)

#path = '/scratch/xtion/dpMMlowVar/2014-11-11-22-18-13/statsDDP.log'
#fig = plotNperCluster(path,False,True)
#
#path = '/scratch/xtion/dpMMlowVar/2014-11-11-22-18-13/statsDP.log'
#fig = plotNperCluster(path,False,False)

#path = '/scratch/xtion/dpMMlowVar/stats.log'
#plotNperCluster(path)
#plt.show()

    

