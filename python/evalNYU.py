# Copyright (c) 2015, Julian Straub <jstraub@csail.mit.edu> Licensed
# under the MIT license. See the license file LICENSE.

#import matplotlib.pyplot as plt
#import matplotlib.cm as cm
import numpy as np
#import cv2
import scipy.io
import subprocess as subp

import os, re, time
import argparse

#from vpCluster.rgbd.rgbdframe import RgbdFrame
#from vpCluster.manifold.sphere import Sphere
#from js.utils.config import Config2String
#from js.utils.plot.pyplot import SaveFigureAsImage

def run(cfg,reRun):
  #args = ['../build/dpSubclusterSphereGMM',
#  args = ['../build/dpStickGMM',
  args = ['../build/bin/realtimeDDPvMF_file',
    '-i {}'.format(cfg['rootPath']+cfg['dataPath']+"_d.png"),
    '-o {}'.format(cfg['outName']),
    '-l {}'.format(100), # lambda
    '-s {}'.format(1),   # survival
    ]

  if reRun:
    print ' '.join(args)
    print ' --------------------- '
    time.sleep(1)
    err = subp.call(' '.join(args),shell=True)
    if err:
      print 'error when executing'
#      raw_input()
#  z = np.loadtxt(cfg['outName']+'.lbl',dtype=int,delimiter=' ')
#  sil = np.loadtxt(cfg['outName']+'.lbl_measures.csv',delimiter=" ")


def config2Str(cfg):
  use = ['mode','dt','tMax','nCGIter']
  st = use[0]+'_'+str(cfg[use[0]])
  for key in use[1::]:
    if key in cfg.keys():
      st += '-'+key+'_'+str(cfg[key])
  return st

parser = argparse.ArgumentParser(description = 'rtmf extraction for NYU')
parser.add_argument('-s','--start', type=int, default=0, 
    help='start image Nr')
parser.add_argument('-e','--end', type=int, default=1449, 
    help='end image Nr')
parser.add_argument('-m','--mode', default='DP-vMF-means', 
    help='spkm, DP-vMF-means')
parser.add_argument('-nyu', action='store_true', 
    help='switch to process the NYU dataset')
args = parser.parse_args()

cfg=dict()
cfg['rootPath'] = '/data/vision/fisher/data1/nyu_depth_v2/extracted/'
cfg['outputPath'] = '/data/vision/scratch/fisher/jstraub/dpMMlowVar/nyu2'

cfg['mode'] = args.mode;

mode = ['multiFromFile']

reRun = True
printCmd = True
onlyPaperEval = True

paperEval = [
'bathroom_0028_691',
'home_office_0012_395',
'kitchen_0037_831',
'bedroom_0085_1084',
'living_room_0064_1314',
'bathroom_0015_664',
'office_kitchen_0001_409',
'bedroom_0026_914',
'bedroom_0032_935',
'bedroom_0043_959',
'conference_room_0002_342',
'dining_room_0030_1422',
'kitchen_0004_1',
'kitchen_0004_2',
'kitchen_0007_131',
'kitchen_0011_143',
'kitchen_0024_774',
'kitchen_0024_776',
'kitchen_0045_865',
'kitchen_0046_870',
'kitchen_0057_567',
'office_0008_15',
'office_0008_17',
'office_0009_19',
'office_0022_618',
'office_0022_619',
'office_0027_635',
'office_0027_638',
'office_kitchen_0001_409']

if not args.nyu:
  print "only NYU supported now"
  exit(1)

cfg['evalStart'] = args.start
cfg['evalEnd'] = args.end
indexPath = '/data/vision/fisher/data1/nyu_depth_v2/index.txt'
cfg['rootPath'] = '/data/vision/fisher/data1/nyu_depth_v2/extracted/'
cfg['outputPath'] = '/data/vision/scratch/fisher/jstraub/dpMMlowVar/nyu2/'

names =[]
with open(indexPath) as f:
  allNames = f.read().splitlines() #readlines()
for i in range(len(allNames)):
  if cfg['evalStart'] <= i and i <cfg['evalEnd']:
    names.append(allNames[i])
    print '@{}: {}'.format(len(names)-1,names[-1])
print names

rndInds = range(len(names)) # np.random.permutation(len(names))
for ind in rndInds:
  if onlyPaperEval and names[ind] not in paperEval:
    continue
      
  cfg['dataPath'] = names[ind]
  cfg['outName'] = cfg['outputPath']+cfg['dataPath']+'_'+config2Str(cfg)

  print 'processing '+cfg['rootPath']+cfg['dataPath']
  run(cfg,reRun)
