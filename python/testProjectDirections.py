import numpy as np
import cv2

I = cv2.imread('/home/jstraub/workspace/research/vpCluster/data/table_0_rgb.png')
cv2.imshow("rgb",I)

Ip = I.copy()

scale = 0.1
f_d = 540
c = np.array([0.,0.,1.])
d = np.array([1.,0.,0.]) *scale

p1 = c
p2 = c+d

def project(p,f):
  u = p[0]/p[2]*f + 320.
  v = p[1]/p[2]*f + 240.
  return np.array([u,v])

u1 = (project(p1,f_d)).astype(int)
u2 = (project(p2,f_d)).astype(int)

cv2.line(Ip,(u1[0],u1[1]),(u2[0],u2[1]),(255,0,0),4)
cv2.imshow("rgb",Ip)

cv2.waitKey(0)
