#!/bin/python
#-----------------------------------------------------------------------------
# File Name : report_classification_mnist.py
# Purpose:
#
# Author: Emre Neftci
#
# Creation Date : 14-11-2016
# Last Modified : Fri 06 Jan 2017 09:12:30 AM PST
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
import matplotlib 

font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 15}

matplotlib.rc('font', **font)
matplotlib.rc('savefig',dpi=600)
matplotlib.rcParams['figure.subplot.hspace']= 0.3
matplotlib.rcParams['figure.subplot.hspace']= 0.3
matplotlib.rcParams['figure.subplot.right']= 0.95
matplotlib.rcParams['figure.subplot.left']= 0.2
matplotlib.rcParams['figure.subplot.bottom']= 0.1
matplotlib.rcParams['figure.subplot.top']= 1.0


def run_classify(context, labels_test, monitor = False):
    #Uses outputs as inputs for the matrix! This is because the weights are symmetrized and written in the output.
    os.system('rm -rf outputs/{directory}/test'.format(**context))
    os.system('mkdir -p outputs/{directory}/test/' .format(**context))
    cmd = 'mpirun -n {ncores} ./exp_rbp_dual \
        --learn {0} \
        --eta {eta}\
        --simtime {simtime_test} \
        --stimtime {simtime_test} \
        --record_full {0} \
        --record_rasters {0} \
        --record_rates true \
        --dir  outputs/{directory}/test/ \
        --fvh  inputs/{directory}/train/{fvh} \
        --fho  inputs/{directory}/train/{fho} \
        --foe  inputs/{directory}/train/{foe} \
        --feo  inputs/{directory}/train/{feo} \
        --fve  inputs/{directory}/train/{fve} \
        --feh  inputs/{directory}/train/{feh} \
        --ip_v inputs/{directory}/test/{ip_v}\
        --nvis {nv} \
        --nhid {nh} \
        --nout {nc} \
        --prob_syn {prob_syn} \
        --sigma {sigma} \
        '.format(monitor,**context)
    if context.has_key('nh1'):
        cmd += " --fhh  inputs/{directory}/train/{fhh} ".format(**context)
    ret = os.system(cmd)

    if ret == 0:
        print 'ran'

    return float(sum(process_test_rbp(context)==labels_test))/len(labels_test)


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
context['sample_duration_test'] = .5
context['sample_pause_test'] = .0
context['simtime_test'] = context['n_samples_test']*(context['sample_duration_test']+context['sample_pause_test'])
if not context.has_key('prob_syn'):
    context['prob_syn']=1.0
if not context.has_key('sigma'):
    context['sigma']=50e-3

n_samples_train = context['n_samples_train']
n_samples_test  = context['n_samples_test']
n_epochs        = context['n_epochs'] 
test_every      = context['test_every'] #never test
M = et.load('M.pkl')
context['directory'] = 'mnist_testonly'

if __name__ == '__main__':
    context['ncores'] = 1
    context['eta']=context['eta_orig']
    context['nh']=int(context['nh'])
    context['seed'] = 300
    context['n_samples_test'] = 10
    context['sample_duration_test'] = .25
    context['sample_pause_test'] = .0
#    context['input_scale'] = .65
    n_samples_train = context['n_samples_train']
    n_samples_test  = context['n_samples_test']
    n_epochs        = context['n_epochs'] 
    test_every      = context['test_every'] #never test
    context['simtime_test'] = context['n_samples_test']*(context['sample_duration_test']+context['sample_pause_test'])

    M = et.load('M.pkl')
    write_allparameters_rbp(M, context)

    labels_test, SL_test = create_data_rbp(n_samples = n_samples_test, 
                  output_directory = '{directory}/test'.format(**context),
                  data_url = context['test_data_url'],
                  labels_url = context['test_labels_url'],
                  randomize = True,
                  with_labels = True,
                  duration_data = context['sample_duration_test'],
                  duration_pause = context['sample_pause_test'],
                  generate_sl = False,
                  **context) 



    acc_hist = []

  
    res = run_classify(context, labels_test, monitor = True)
    print res

    ion()

    os.system('../tools/aubs -i outputs/mnist_testonly/test/exp_rbp_online.0.bdendrite -o /tmp/h')
    bdend = np.loadtxt('/tmp/h')

    os.system('../tools/aubs -i outputs/mnist_testonly/test/exp_rbp_online.0.bmem -o /tmp/h')
    bmem = np.loadtxt('/tmp/h')

    os.system('../tools/aubs -i outputs/mnist_testonly/test/exp_rbp_online.0.bampa -o /tmp/h')
    bampa = np.loadtxt('/tmp/h')

    w = np.loadtxt('outputs/mnist_testonly/test/coba.0.v.w')

    figure(figsize=[6,6])

    sin = monitor_to_spikelist('outputs/mnist_testonly/test/coba.0.hid.ras')[32].spike_times
    sout = monitor_to_spikelist('outputs/mnist_testonly/test/coba.0.out.ras')[0].spike_times
    serrp = monitor_to_spikelist('outputs/mnist_testonly/test/coba.0.err1.ras')[0].spike_times
    serrn = monitor_to_spikelist('outputs/mnist_testonly/test/coba.0.err2.ras')[0].spike_times

    mi=000
    ma=1000

    from matplotlib.ticker import MaxNLocator
    subplot(4,1,2)
    plot([0,bdend[0,0]*1000],[0,0],'k')
    plot(bdend[:,0]*1000,bdend[:,1],'k')
    for i in serrn:
        axvline(i, color='b', alpha=.2)
    for i in serrp:
        axvline(i, color='r', alpha=.2)
    xlim([mi,ma])
    xticks([])
    ylabel('$U [V]$',fontsize='x-large')
    gca().yaxis.set_major_locator(MaxNLocator(3))
    gca().get_yaxis().set_label_coords(-0.18,0.5)

    subplot(4,1,1)
    for i in sout:
        axvline(i, color='k', alpha=.2)
    plot(bmem[:,0]*1000,bmem[:,1], 'k')
    ylabel('$V [V]$',fontsize='x-large')
    xlim([mi,ma])
    xticks([])
    gca().yaxis.set_major_locator(MaxNLocator(3))
    gca().get_yaxis().set_label_coords(-0.18,0.5)


    subplot(4,1,3)
    for i in sin:
        axvline(i, color='k', alpha=.2)
    ylabel('$I_{syn} [nA]$',fontsize='x-large')
    fill_between([0,context['simtime_test']*1000],-1.15,1.15,color='g',alpha=.2)
    plot(bampa[:,0]*1000,bampa[:,1], 'k')
    xlim([mi,ma])
    xticks([])
    gca().yaxis.set_major_locator(MaxNLocator(3))
    gca().get_yaxis().set_label_coords(-0.18,0.5)


    subplot(4,1,4)
    for i in sin:
        axvline(i, color='k', alpha=.2)
    ylabel('$w^p_{32,0}[nA]$',fontsize='x-large')
    plot(w[:,0]*1000,w[:,1], 'k')
    gca().yaxis.set_major_locator(MaxNLocator(3))
    gca().xaxis.set_major_locator(MaxNLocator(5))
    gca().get_yaxis().set_label_coords(-0.18,0.5)
    xlim([mi,ma])
    xlabel('Time[ms]')
    et.savefig('trace.png',format='png')




    

