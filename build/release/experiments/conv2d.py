#!/bin/python
#-----------------------------------------------------------------------------
# File Name : conv2d.py
# Author: Emre Neftci
#
# Creation Date : Mon 20 Feb 2017 12:27:52 PM PST
# Last Modified : 
#
# Copyright : (c) UC Regents, Emre Neftci
# Licence : GPLv2
#----------------------------------------------------------------------------- 
import numpy as np
from experimentLib import *
from pylab import *

def __stim_rotate(image, angle=45):
    from scipy import ndimage
    return ndimage.rotate(image, angle, reshape=False)

def stim_orientations(image, orientations = 4, flattened = True):
    if not hasattr(orientations, '__iter__'):
        orientations = np.linspace(0, 180, orientations)
    stimuli = [__stim_rotate(image, angle) for angle in orientations]
    if flattened:
        stimuli = [s.flatten() for s in stimuli]
    return np.array(stimuli)

def conv2d(imsize=28,ksize=5,stride=2):
    kx = ksize
    ky = ksize

    sx = imsize
    sy = imsize

    ox = imsize/stride
    oy = imsize/stride



    Kidx = np.arange(0, kx*ky, dtype='int').reshape(kx, ky)
    K = np.zeros([kx,ky])-10e-3
    K[[1,2],:]=10e-3


    W = np.zeros([sx,sy,sx,sy], dtype='int')-1

    for k in range(sx):
        for l in range(sx):
            for i in range(sx):
                for j in range(sy):
                    dx = k-i
                    dy = l-j
                    if dx in range(kx) and dy in range(ky):
                        W[k,l,i,j] = Kidx[dx,dy]
    Wd = W[:,:,::stride,::stride].reshape(sx*sy,ox*oy)
    return Wd, K

def save_auryn_conv2d(filename, Wd, K):
    save_auryn_wmat(filename+'cw',Wd, mask = Wd!=-1)
    save_auryn_wmat(filename+'w',K.reshape(-1,1))

def SLrates(filename,nx=28,ny=28):
    SL = monitor_to_spikelist(filename).id_slice(range(nx*ny))
    ssl = SL.time_slice(0,250)
    ssl.complete(range(nx*ny))
    return ssl.mean_rates().reshape(nx,ny)


if __name__=='__main__':
    Wd1,K1 = conv2d(28,5,2)
    save_auryn_conv2d('/tmp/layer1_', Wd1, K1)

    KK1 = np.zeros([4,5,5])
    KK1[0,:]=[[ 0, 0, 0, 0, 0 ],
              [ 1, 1, 1, 1, 1 ],
              [ 1, 1, 1, 1, 1 ],
              [ 0, 0, 0, 0, 0 ],
              [ 0, 0, 0, 0, 0 ]]
    KK1[1,:]=[[ 0, 0, 0, 1, 1 ],
              [ 0, 0, 1, 1, 0 ],
              [ 0, 1, 1, 0, 0 ],
              [ 1, 1, 0, 0, 0 ],
              [ 1, 0, 0, 0, 0 ]]
    KK1[2,:]=[[ 1, 0, 0, 0, 0 ],
              [ 1, 1, 0, 0, 0 ],
              [ 0, 1, 1, 0, 0 ],
              [ 0, 0, 1, 1, 0 ],
              [ 0, 0, 0, 1, 1 ]]
    KK1[3,:]=[[ 0, 1, 1, 0, 0 ],
              [ 0, 1, 1, 0, 0 ],
              [ 0, 1, 1, 0, 0 ],
              [ 0, 1, 1, 0, 0 ],
              [ 0, 1, 1, 0, 0 ]]

    KK2 = KK1.copy()
    KK2 = np.concatenate([KK1,KK1])

    for i,k in enumerate(KK1):
        save_auryn_wmat('/tmp/layer1_'+'w_{0}'.format(i), (k.reshape(-1,1)-.5)*100e-3)
    for i in range(4):
        for j,k in enumerate(KK2):
            save_auryn_wmat('/tmp/layer2_'+'w_{0}_{1}'.format(i,j), (k.reshape(-1,1)-.5)*100e-3)

    Wd2,K2 = conv2d(14,5,2)
    save_auryn_conv2d('/tmp/layer2_', Wd2, K2)

    os.system('mpirun -n 4 ./test_erbp_conv2d')
    figure();imshow(SLrates('outputs/test_srm/erbp_conv2d.*.vis.ras',28,28))
    figure();imshow(SLrates('outputs/test_srm/erbp_conv2d.*.c10.ras',14,14))
    figure();imshow(SLrates('outputs/test_srm/erbp_conv2d.*.c11.ras',14,14))
    figure();imshow(SLrates('outputs/test_srm/erbp_conv2d.*.c12.ras',14,14))
    figure();imshow(SLrates('outputs/test_srm/erbp_conv2d.*.c13.ras',14,14))
    figure();imshow(SLrates('outputs/test_srm/erbp_conv2d.*.c20.ras',7,7))
    figure();imshow(SLrates('outputs/test_srm/erbp_conv2d.*.c21.ras',7,7))
    figure();imshow(SLrates('outputs/test_srm/erbp_conv2d.*.c22.ras',7,7))
    figure();imshow(SLrates('outputs/test_srm/erbp_conv2d.*.c23.ras',7,7))

    show()

        
        


