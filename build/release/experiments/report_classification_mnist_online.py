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


def run_classify(context, labels_test, monitor_raster = False, monitor_full=False):
    #Uses outputs as inputs for the matrix! This is because the weights are symmetrized and written in the output.
    os.system('rm -rf outputs/{directory}/test'.format(**context))
    os.system('mkdir -p outputs/{directory}/test/' .format(**context))
    cmd = 'mpirun -n {ncores} ./exp_rbp_dual \
        --learn {0} \
        --eta {eta}\
        --simtime {simtime_test} \
        --stimtime {simtime_test} \
        --record_full {0} \
        --record_rasters {1} \
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
        '.format(monitor_full, monitor_raster,**context)
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
if __name__ == '__other__':
    context['eta']=0
    context['ncores'] = 4
    M = et.load('M.pkl')
    write_allparameters_rbp(M, context)
    if not context.has_key('sigma'):
        context['sigma'] = 50e-3
    if not context.has_key('nh1'):
        context['fhh']=''


    labels_test, SL_test = create_data_rbp(n_samples = n_samples_test, 
                  output_directory = '{directory}/test'.format(**context),
                  data_url = context['test_data_url'],
                  labels_url = context['test_labels_url'],
                  randomize = False,
                  with_labels = False,
                  duration_data = context['sample_duration_test'],
                  duration_pause = context['sample_pause_test'],
                  generate_sl = False,
                  **context) 

    acc_hist = []

    context['seed'] = 0
    res = run_classify(context, labels_test)
    print res




if __name__ == '__main__':

    et.globaldata.directory = directory
    context = et.load('context.pkl')
    context['n_samples_test'] = 10000
    context['sample_duration_test'] = .3
    context['sample_pause_test'] = .1
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
    context['eta']=0
    context['ncores'] = 4
    M = et.load('M.pkl')
    write_allparameters_rbp(M, context)
    if not context.has_key('sigma'):
        context['sigma'] = 50e-3
    if not context.has_key('nh1'):
        context['fhh']=''


    labels_test, SL_test = create_data_rbp(n_samples = n_samples_test, 
                  output_directory = '{directory}/test'.format(**context),
                  data_url = context['test_data_url'],
                  labels_url = context['test_labels_url'],
                  randomize = False,
                  with_labels = False,
                  duration_data = context['sample_duration_test'],
                  duration_pause = context['sample_pause_test'],
                  generate_sl = False,
                  **context) 

    acc_hist = []

    context['seed'] = 0
    res = run_classify(context, labels_test, monitor_raster = True)
    print res
    def count(s, n=2):
        S = np.zeros([10], 'int')
        for i in s[:n]:
            S[i]+=1
        return S

    def search_left(s,v):
        for i,ss in enumerate(s):
            if v<ss:
                return 0
            if v==ss:
                return i
        return -1


    SLv = monitor_to_spikelist('outputs/{directory}/test/coba.*.vis.ras'.format(**context))
    SLh = monitor_to_spikelist('outputs/{directory}/test/coba.*.hid.ras'.format(**context))
    SLo = monitor_to_spikelist('outputs/{directory}/test/coba.*.out.ras'.format(**context))
    por = [None for i in range(context['n_samples_test'])]
    cntv = [[0 for _ in range(50)] for i in range(context['n_samples_test'])]
    cnth = [[0 for _ in range(50)] for i in range(context['n_samples_test'])]
    T = (context['sample_duration_test']+context['sample_pause_test'])*1000
    sv = SLv.raw_data()
    sh = SLh.raw_data()
    idxh = np.argsort(sh[:,0])
    sh = sh[idxh,:].astype('int')
    idxv = np.argsort(sv[:,0])
    sv = sv[idxv,:].astype('int')
    sv[:,1] = 1
    sh[:,1] = 1
    cntha = np.row_stack([sh[:,0],np.cumsum(sh[:,1])]).T
    cntva = np.row_stack([sv[:,0],np.cumsum(sv[:,1])]).T
    chi = 0
    cvi = 0
    for i in range(context['n_samples_test']): #context['n_samples_test']):
        t0 = (i*T)+context['sample_pause_test']*1000
        t2 = t0+8
        print i,t2
        s = SLo.time_slice(t2, (i+1)*T).raw_data()
        idx = np.argsort(s[:,0])
        s = s[idx,:].astype('int')
        g = []
        ss = s.shape[0]
        chi = search_left(cntha[:,0],t0)
        cvi = search_left(cntva[:,0],t0)
        blvi = cntva[cvi,1]
        blhi = cntha[chi,1]
        cntha = cntha[chi:]
        cntva = cntva[cvi:]
        for j in range(1,50):
            g.append(np.argmax(count(s[:,1],j)) == labels_test[i])
            if ss>0:
                t3 = s[np.minimum(j,ss-1),0]            
                cntv[i][j]=cntva[search_left(cntva[:,0],t3), 1]-blvi
                cnth[i][j]=cntha[search_left(cntha[:,0],t3), 1]-blhi
        por[i] = g
    por = np.array(por)
    cnth = np.array(cnth)
    cntv = np.array(cntv)

    convergence = np.array(et.load('acc_hist.pkl'))
    figure(figsize=(6,4))
    ax1=axes()
    ax1.semilogx(range(1,50), 100*(1-np.mean(por, axis=0)), '.-', linewidth=3, alpha=.6, label = 'Error')
    ylim([2,5])
    xlim([0,20])
    legend(loc = 4)
    xlabel('Number of spikes in output layer')
    ylabel('Error %')
    ax2 = ax1.twinx()
    ax2 . semilogy(range(0,50), .200*np.mean(cntv,axis=0)+.10*np.mean(cnth,axis=0), '.-', linewidth=3, alpha=.6, label = 'SynOps', color = 'r')
    ax2.set_ylabel('kSynops')
    legend(loc = 9)
    tight_layout()
    et.savefig('convergence_spike.png', format='png', dpi=1200)


if __name__ == '__main0__':
#if __name__ == '__main__':
    context['ncores'] = 1
    context['eta']=context['eta_orig']*0
    context['n_samples_test'] = 100
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
                  randomize = False,
                  with_labels = True,
                  duration_data = context['sample_duration_test'],
                  duration_pause = context['sample_pause_test'],
                  generate_sl = False,
                  **context) 



    acc_hist = []

    context['seed'] = 0
  
    res = run_classify(context, labels_test, monitor = True)
    print res

    ion()

    f=figure(figsize=(6,4))
    a=axes()
    SLv = monitor_to_spikelist('outputs/{directory}/test/coba.*.vis.ras'.format(**context))
    SLv.id_slice(range(784)).raster_plot(kwargs={'marker':'.','markersize':3,'alpha':.2,'color':'k'}, display=a)
    tight_layout()
    a.set_ylabel('')
    a.set_xlim([0,1250])
    et.savefig('raster_vis.png')
#    for i in range( context['n_samples_test']):
#        sv = SLv.id_slice(range(784)).time_slice(i*context['sample_duration_test']*1000,(i+1)*context['sample_duration_test']*1000)
#        sv.complete(range(784))
#        f = figure(figsize=(4,4)) 
#        imshow(sv.mean_rates().reshape(28,28), interpolation='nearest')
#        xticks([])
#        yticks([])
#        xlabel('')
#        ylabel('')
#        bone()
#        tight_layout()
#        et.savefig('digit{0}.png'.format(i))

    figure(figsize=(6,4))
    a=axes()
    SLh = monitor_to_spikelist('outputs/{directory}/test/coba.*.hid.ras'.format(**context))
    SLh1 = SLh.id_slice(range(0,200)).copy()
    SLh1.raster_plot(kwargs={'marker':'.','markersize':3,'alpha':.2,'color':'b'}, display=a)
    SLh2 = SLh.id_slice(range(200,400)).copy()
    SLh2.raster_plot(kwargs={'marker':'.','markersize':3,'alpha':.2,'color':'g'}, display=a)
    a.set_ylim([0,400])
    a.set_xlim([0,1250])
    yticks([100,300], ['Layer 1','Layer 2'], rotation=90)
    a.set_ylabel('')
    tight_layout()
    et.savefig('raster_hid.png')

    figure(figsize=(6,2))
    a=axes()
    SLo = monitor_to_spikelist('outputs/{directory}/test/coba.*.out.ras'.format(**context))
    SLo.raster_plot(kwargs={'marker':'.','markersize':3,'alpha':.2,'color':'k'}, display=a)
    yticks([0,10])
    tight_layout()
    a.set_ylabel('')
    et.savefig('raster_out.png')
    
    figure(figsize=(6,2))
    a=axes()
    tbin = 10
    SLe1 = monitor_to_spikelist('outputs/{directory}/test/coba.*.err1.ras'.format(**context))
    T = SLe1.time_axis(tbin)[:-1]
    SLe2 = monitor_to_spikelist('outputs/{directory}/test/coba.*.err2.ras'.format(**context))
    T = SLe2.time_axis(tbin)[:-1]
    SLe1.raster_plot(display=a,kwargs={'marker':'.','markersize':3,'alpha':.5,'color':'b'})
    SLe2.raster_plot(display=a,kwargs={'marker':'.','markersize':3,'alpha':.5,'color':'r'})
    a.set_xlim([T.min(),T.max()])
    tight_layout()
    yticks([0,10])
    a.set_ylabel('')
    a.set_xlim([0,1250])
    et.savefig('rates_err.png')

    figure(figsize=(6,2))
    a=axes()
    tbin = 10
    SLl = SLv.id_slice(range(784,794)).raster_plot(kwargs={'marker':'.','markersize':3,'alpha':.2,'color':'k'}, display=a)
    tight_layout()
    yticks([784,794],[0,10])
    a.set_ylabel('')
    a.set_xlim([0,1250])
    et.savefig('rates_label.png')

    po = []
    ph = []
    pV = []
    T = (context['sample_duration_test']+context['sample_pause_test'])*1000
    for i in range(context['n_samples_test']):
        po.append(SLo.time_slice(i*T+context['sample_pause_test']*1000+4, (i+1)*T).first_spike_time()[1])
        ph.append(SLh.time_slice(i*T+context['sample_pause_test']*1000+.1, i*T+context['sample_pause_test']*1000+4).raw_data().__len__())
        pV.append(SLv.time_slice(i*T+context['sample_pause_test']*1000+.1, i*T+context['sample_pause_test']*1000+4).raw_data().__len__())
    print np.sum(np.array(po) == labels_test)

