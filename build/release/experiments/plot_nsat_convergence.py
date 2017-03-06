#!/bin/python
#-----------------------------------------------------------------------------
# File Name : plot_nsat_convergence.py
# Author: Emre Neftci
#
# Creation Date : Mon 16 Jan 2017 02:16:33 PM PST
# Last Modified : 
#
# Copyright : (c) UC Regents, Emre Neftci
# Licence : GPLv2
#----------------------------------------------------------------------------- 

from experimentLib import *
import experimentTools as et
from pylab import *
import matplotlib,pickle 

font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 14}

matplotlib.rc('font', **font)
matplotlib.rc('savefig',dpi=600)

dnsat = '/home/eneftci/Projects/code/python/HiAER-NSAT/examples/erbp/Results/021__15-01-2017/'
et.globaldata.directory = '/homes/eneftci/work/code/C/auryn_rbp/build/release/experiments/Results/127__13-01-2017/'
convergence = np.array(et.load('acc_hist.pkl'))
context = et.load('context.pkl')
figure(figsize=(6,4))
v = context['n_samples_train']/50000
semilogx(v+(convergence[:,0])*v,100-100*convergence[:,1], 'o-',linewidth=3, alpha=.6, label='peRBP (Spiking 64-bit)')
tt,nsat = np.array(pickle.load(file(dnsat+'pip.pkl','r'))).T
semilogx(1+tt,100-nsat, 'x-',linewidth=3, alpha=.6, label='peRBP (Spiking 8-bit)')
xlim([0,30])
xticks([1,10,20,30],[1,10,20,30],rotation=45)
ylim([0,15])
axhline(6, color='k', alpha=.6)
xlabel('Epochs')
ylabel('Error %')
legend()
title('784-100-10')
tight_layout()
savefig('Results/convergence_erbp_nsat.png', format='png', dpi=1200)

wvh = pickle.load(file(dnsat+'wvh.pkl','r'))
figure(figsize=(6,4))
hist(wvh.flatten(),bins = range(-128,128), edgecolor = "none")    
xlabel('Synaptic Weight')
xlim([-130,130])
ylabel('Count')
tight_layout()
savefig('Results/histogram_weights_nsat.png',  format='png', dpi=1200)
