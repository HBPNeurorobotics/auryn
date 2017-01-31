#!/bin/python
#-----------------------------------------------------------------------------
# File Name : report_classification_mnist.py
# Purpose:
#
# Author: Emre Neftci
#
# Creation Date : 14-11-2016
# Last Modified : Fri 25 Nov 2016 10:11:23 AM PST
#
# Copyright : (c) 
# Licence : GPLv2
#----------------------------------------------------------------------------- 
import matplotlib
from experimentLib import *
import os, sys
import getopt
import experimentTools as et
from pylab import *
from npamlib import *


def run_classify(context, labels_test):
    #Uses outputs as inputs for the matrix! This is because the weights are symmetrized and written in the output.
    os.system('rm -rf outputs/{directory}/test'.format(**context))
    os.system('mkdir -p outputs/{directory}/test/' .format(**context))
    ret = os.system('mpirun -n {ncores} ./exp_rbp \
        --learn false \
        --eta 0.\
        --simtime {simtime_test} \
        --stimtime {simtime_test} \
        --record_full false \
        --record_rasters false \
        --record_rates true \
        --dir  outputs/{directory}/test/ \
        --fvh  inputs/{directory}/train/{fvh} \
        --fho  inputs/{directory}/train/{fho} \
        --fhh  inputs/{directory}/train/{fhh} \
        --foe  inputs/{directory}/train/{foe} \
        --feo  inputs/{directory}/train/{feo} \
        --fve  inputs/{directory}/train/{fve} \
        --feh  inputs/{directory}/train/{feh} \
        --ip_v inputs/{directory}/test/{ip_v}\
        --prob_syn {prob_syn}\
        --sigma {sigma}\
        --nvis {nv} \
        --nhid {nh} \
        --nout {nc} \
        '.format(**context))

    if ret == 0:
        print 'ran'

    return float(sum(process_test_rbp(context)==labels_test))/len(labels_test)

directory = None

optlist,args=getopt.getopt(sys.argv[1:],'hn:t:d:sgc:')
for o,a in optlist:        
    if o == '-h': #hot init
        print 'Hot start'
        hot_init = True
    if o == '-n': #change number of epochs
        context['n_epochs'] = int(a)
    if o == '-t':
        context['test_every'] = int(a)
    if o == '-d':
        directory = a
    if o == '-s':
        save = True
    if o == '-g':
        generate = True
    if o == '-c': #custom parameters passed through input arguments
        context.update(eval(a))

et.globaldata.directory = directory
context = et.load('context.pkl')
context['n_samples_test'] = 10000
context['simtime_test'] = 3000.

n_samples_train = context['n_samples_train']
n_samples_test  = context['n_samples_test']
n_epochs        = context['n_epochs'] 
test_every      = context['test_every'] #never test
context['directory'] = 'mnist_testonly'
M = et.load('M.pkl')

if __name__ == '__main__':
    context['directory'] = 'mnist_testonly'
    M = et.load('M.pkl')
    write_parameters_rbp(M, context)

    labels_test, SL_test = create_data_rbp(n_samples = n_samples_test, 
                  output_directory = '{directory}/test'.format(**context),
                  data_url = context['test_data_url'],
                  labels_url = context['test_labels_url'],
                  randomize = False,
                  with_labels = False,
                  duration_data = context['sample_duration_test'],
                  duration_pause = context['sample_pause_test'],
                  **context) 

    acc_hist = []

    context['seed'] = 0
  
    res = run_classify(context, labels_test)
    print res

    stim_show(M['vh'].todense()[:,:100].T)
    et.savefig('features_100.png')

if __name__ == '__plot__':
    stim_show(M['vh'].todense()[:,:100].T)
    et.savefig('features_100.png')
    
    fh = file('/home/eneftci/Projects/code/python/dtp/monitor_gpu_rbp_500.pkl','r')
    import pickle
    cgpu = np.array(pickle.load(fh)['test'])

    convergence = np.array(et.load('acc_hist.pkl'))
    figure(figsize=(6,4))
    plot(20+convergence[:,0]*20,100-100*convergence[:,1], linewidth=3, alpha=.6, label='eRBP (Spiking)')
    plot(60*np.arange(len(cgpu)),cgpu[:], linewidth=3, alpha=.6, label='RBP (GPU)')
    xlabel('Samples presented [k]')
    ylabel('Error %')
    tight_layout()
    et.savefig('convergence_erbp_auryn.png')










