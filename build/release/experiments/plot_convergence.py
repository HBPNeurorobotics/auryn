#!/bin/python
#-----------------------------------------------------------------------------
# File Name : plot_convergence.py
# Author: Emre Neftci
#
# Creation Date : Fri 06 Jan 2017 01:59:04 PM PST
# Last Modified : 
#
# Copyright : (c) UC Regents, Emre Neftci
# Licence : GPLv2
#----------------------------------------------------------------------------- 
from experimentLib import *
import experimentTools as et
from pylab import *
import matplotlib 

font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 14}

matplotlib.rc('font', **font)
matplotlib.rc('savefig',dpi=600)


def extract_spk_cnt(spk_cnt, pop_id):
    return np.array([sum(s[pop_id][:,1]) for s in spk_cnt])



def plot_convergence_200_200():
    #stim_show(M['vh'].todense()[:,:100].T)
    #et.savefig('features_100.png')
    pylab.ion()
    
    fh = file('/home/eneftci/Projects/code/python/dtp/Results/rbp_200_200.pkl','r')
    import pickle
    cgpu = np.array(pickle.load(fh)['test'])
    fh = file('/home/eneftci/Projects/code/python/dtp/Results/rbp_200_200.pkl','r')
    trbp = np.array(pickle.load(fh)['epoch'])

    fh = file('/home/eneftci/Projects/code/python/dtp/Results/bp_200_200.pkl','r')
    cgpubp = np.array(pickle.load(fh)['test'])
    fh = file('/home/eneftci/Projects/code/python/dtp/Results/bp_200_200.pkl','r')
    tbp = np.array(pickle.load(fh)['epoch'])

    et.globaldata.directory = '/homes/eneftci/work/code/C/auryn_rbp/build/release/experiments/Results/134__17-01-2017/'
    context = et.load('context.pkl')
    convergence_perbp = np.array(et.load('acc_hist.pkl'))

    et.globaldata.directory = '/homes/eneftci/work/code/C/auryn_rbp/build/release/experiments/Results/120__07-01-2017/'
    context = et.load('context.pkl')
    convergence_erbp = np.array(et.load('acc_hist.pkl'))

    figure(figsize=(6,4))
    title('784-200-200-10')
    v = context['n_samples_train']/50000
    semilogx(trbp,cgpu[:], '.-', linewidth=3, alpha=.6, label='RBP (GPU)')
    semilogx(tbp,cgpubp[:], '.-', linewidth=3, alpha=.6, label='BP (GPU)')
    semilogx(v+convergence_erbp[:,0]*v,100-100*convergence_erbp[:,1], '.-',linewidth=3, alpha=.6, label='eRBP (Spiking)')
    semilogx(v+convergence_perbp[:,0]*v,100-100*convergence_perbp[:,1], '.-',linewidth=3, alpha=.6, label='peRBP (Spiking)')

    xlim([0,2000])
    xlabel('Epochs')
    ylabel('Error %')
    xticks([1,10,100,1000,2000],[1,10,100,1000,2000],rotation=45)
    ylim([0,15])
    #legend(frameon=False)
    tight_layout()
    savefig('Results/convergence_erbp_auryn_200_200.png', format='png', dpi=1200)


def load_avg_convergence(d, prefix = ''):
    c = []
    for d_ in d:
        et.globaldata.directory = prefix + d_
        context = et.load('context.pkl')
        c.append(np.array(et.load('acc_hist.pkl')))
    return np.mean(c, axis = 0), context


def plot_convergence_200():
    #stim_show(M['vh'].todense()[:,:100].T)
    #et.savefig('features_100.png')
    pylab.ion()
    
    
    fh = file('/home/eneftci/Projects/code/python/dtp/Results/rbp_200.pkl','r')
    import pickle
    cgpu = np.array(pickle.load(fh)['test'])
    fh = file('/home/eneftci/Projects/code/python/dtp/Results/rbp_200.pkl','r')
    trbp = np.array(pickle.load(fh)['epoch'])

    fh = file('/home/eneftci/Projects/code/python/dtp/Results/bp_200.pkl','r')
    cgpubp = np.array(pickle.load(fh)['test'])
    fh = file('/home/eneftci/Projects/code/python/dtp/Results/bp_200.pkl','r')
    tbp = np.array(pickle.load(fh)['epoch'])

    d = ['005__23-01-2017/', '006__23-01-2017/', '007__25-01-2017/', '008__25-01-2017/']
    convergence_perbp, context =  load_avg_convergence(d, prefix = '/home/eneftci/Projects/code/C/auryn_rbp/build/release/experiments/Results_scripts/')
    assert context['sigma']==0
    assert context['nh']==200
    et.globaldata.directory = '/homes/eneftci/work/code/C/auryn_rbp/build/release/experiments/Results/132__16-01-2017/'
    context = et.load('context.pkl')
    assert context['sigma']>0
    assert context['nh']==200
    convergence_erbp = np.array(et.load('acc_hist.pkl'))

    figure(figsize=(6,4))
    title('784-200-10')
    v = context['n_samples_train']/50000
    semilogx(trbp,cgpu[:], '.-', linewidth=3, alpha=.6, label='RBP (GPU)')
    semilogx(tbp,cgpubp[:], '.-', linewidth=3, alpha=.6, label='BP (GPU)')
    semilogx(v+convergence_erbp[:,0]*v,100-100*convergence_erbp[:,1], '.-',linewidth=3, alpha=.6, label='eRBP (Spiking)')
    semilogx(v+convergence_perbp[:,0]*v,100-100*convergence_perbp[:,1], '.-',linewidth=3, alpha=.6, label='peRBP (Spiking)')

    xlim([0,2000])
    xticks([1,10,100,1000,2000],[1,10,100,1000,2000],rotation=45)
    xlabel('Epochs')
    ylabel('Error %')
    ylim([0,15])
    #legend(frameon=False)
    tight_layout()
    savefig('Results/convergence_erbp_auryn_200.png', format='png', dpi=1200)

def plot_convergence_100():
    #stim_show(M['vh'].todense()[:,:100].T)
    #et.savefig('features_100.png')
    pylab.ion()
    
    
    fh = file('/home/eneftci/Projects/code/python/dtp/Results/rbp_100.pkl','r')
    import pickle
    cgpu = np.array(pickle.load(fh)['test'])
    fh = file('/home/eneftci/Projects/code/python/dtp/Results/rbp_100.pkl','r')
    trbp = np.array(pickle.load(fh)['epoch'])

    fh = file('/home/eneftci/Projects/code/python/dtp/Results/bp_100.pkl','r')
    cgpubp = np.array(pickle.load(fh)['test'])
    fh = file('/home/eneftci/Projects/code/python/dtp/Results/bp_100.pkl','r')
    tbp = np.array(pickle.load(fh)['epoch'])

    et.globaldata.directory = '/homes/eneftci/work/code/C/auryn_rbp/build/release/experiments/Results/127__13-01-2017/'
    context = et.load('context.pkl')
    assert context['sigma'] > 0
    assert context['nh'] ==100
    convergence_erbp = np.array(et.load('acc_hist.pkl'))

    d = ['002__22-01-2017/', '003__22-01-2017/', '004__22-01-2017/', '001__22-01-2017/']
    convergence_perbp, context =  load_avg_convergence(d, prefix = '/home/eneftci/Projects/code/C/auryn_rbp/build/release/experiments/Results_scripts/')
    assert context['sigma'] == 0
    assert context['nh'] ==100

    figure(figsize=(6,4))
    title('784-100-10')
    v = context['n_samples_train']/50000
    semilogx(trbp,cgpu[:], '.-', linewidth=3, alpha=.6, label='RBP (GPU)')
    semilogx(tbp,cgpubp[:], '.-', linewidth=3, alpha=.6, label='BP (GPU)')
    semilogx(v+convergence_erbp[:,0]*v,100-100*convergence_erbp[:,1], '.-',linewidth=3, alpha=.6, label='eRBP (Spiking)')
    semilogx(v+convergence_perbp[:,0]*v,100-100*convergence_perbp[:,1], '.-',linewidth=3, alpha=.6, label='peRBP (Spiking)')

    xlim([0,2000])
    xlabel('Epochs')
    ylabel('Error %')
    ylim([0,15])
    xticks([1,10,100,1000,2000],[1,10,100,1000,2000],rotation=45)
    legend(frameon=False)
    tight_layout()
    savefig('Results/convergence_erbp_auryn_100.png', format='png', dpi=1200)

def plot_convergence_500():
    #stim_show(M['vh'].todense()[:,:100].T)
    #et.savefig('features_100.png')
    pylab.ion()
    
    
    fh = file('/home/eneftci/Projects/code/python/dtp/Results/rbp_500.pkl','r')
    import pickle
    cgpu = np.array(pickle.load(fh)['test'])
    fh = file('/home/eneftci/Projects/code/python/dtp/Results/rbp_500.pkl','r')
    trbp = np.array(pickle.load(fh)['epoch'])

    fh = file('/home/eneftci/Projects/code/python/dtp/Results/bp_500.pkl','r')
    cgpubp = np.array(pickle.load(fh)['test'])
    fh = file('/home/eneftci/Projects/code/python/dtp/Results/bp_500.pkl','r')
    tbp = np.array(pickle.load(fh)['epoch'])

    d = ['009__26-01-2017/', '010__26-01-2017/']
    convergence_perbp, context =  load_avg_convergence(d, prefix = '/home/eneftci/Projects/code/C/auryn_rbp/build/release/experiments/Results_scripts/')
    assert context['sigma'] == 0
    assert context['nh'] ==500

    et.globaldata.directory = '/homes/eneftci/work/code/C/auryn_rbp/build/release/experiments/Results/123__10-01-2017/'
    context = et.load('context.pkl')
    convergence_erbp = np.array(et.load('acc_hist.pkl'))

    figure(figsize=(6,4))
    title('784-500-10')
    v = context['n_samples_train']/50000
    semilogx(trbp,cgpu[:], '.-', linewidth=3, alpha=.6, label='RBP (GPU)')
    semilogx(tbp,cgpubp[:], '.-', linewidth=3, alpha=.6, label='BP (GPU)')
    semilogx(v+convergence_erbp[:,0]*v,100-100*convergence_erbp[:,1], '.-',linewidth=3, alpha=.6, label='eRBP (Spiking)')
    semilogx(v+convergence_perbp[:,0]*v,100-100*convergence_perbp[:,1], '.-',linewidth=3, alpha=.6, label='peRBP (Spiking)')

    xlim([0,2000])
    xlabel('Epochs')
    ylabel('Error %')
    ylim([0,15])
    xticks([1,10,100,1000,2000],[1,10,100,1000,2000],rotation=45)
    #legend(frameon=False)
    tight_layout()
    savefig('Results/convergence_erbp_auryn_500.png', format='png', dpi=1200)

def plot_synop_mac():
    et.globaldata.directory = '/homes/eneftci/work/code/C/auryn_rbp/build/release/experiments/Results/135__27-01-2017/'
    context = et.load('context.pkl')
    convergence_perbp = np.array(et.load('acc_hist.pkl'))

    nh = context['nh']
    nc = context['nc']
    nv = context['nv']
    spk_cnt = et.load('spkcnt.pkl')
    acc_hist = np.array(et.load('acc_hist.pkl'))
    nerr2  = extract_spk_cnt(spk_cnt, 0)
    nerr1  = extract_spk_cnt(spk_cnt, 1)
    nvis   = extract_spk_cnt(spk_cnt, 2)
    nout   = extract_spk_cnt(spk_cnt, 3)   
    nhid   = extract_spk_cnt(spk_cnt, 4)

    import pickle
    fh = file('/home/eneftci/Projects/code/python/dtp/Results/bp_200_200.pkl','r')
    cgpubp = np.array(pickle.load(fh)['test'])
    fh = file('/home/eneftci/Projects/code/python/dtp/Results/bp_200_200.pkl','r')
    tbp = np.array(pickle.load(fh)['epoch'])

    ss = np.cumsum((nv*nh + nh*nc + nv*nh + nh*nc)*np.ones(2000)*50000)


    plot(acc_hist[:,1],np.cumsum(nvis*nh + nhid*nc + 2*nout*nc + (nerr1+nerr2)*(nc+nh)).astype('float'),'r.-', linewidth=3, label='peRBP (Spiking)')
    plot(1-cgpubp/100,ss[tbp-1].astype('float'),'g.-', linewidth=3, label='BP (GPU)')
    ylabel('MACs / SynOps')
    xlabel('Accuracy')
    legend(loc=2)
    xticks([.95,.96,.97,.98])

def plot_mac():
    et.globaldata.directory = '/homes/eneftci/work/code/C/auryn_rbp/build/release/experiments/Results/133__17-01-2017/'
    context = et.load('context.pkl')
    convergence_perbp = np.array(et.load('acc_hist.pkl'))

    nh = context['nh']
    nc = context['nc']
    nv = context['nv']
    spk_cnt = et.load('spkcnt.pkl')
    acc_hist = np.array(et.load('acc_hist.pkl'))
    nerr2  = extract_spk_cnt(spk_cnt, 0)
    nerr1  = extract_spk_cnt(spk_cnt, 1)
    nvis   = extract_spk_cnt(spk_cnt, 2)
    nout   = extract_spk_cnt(spk_cnt, 3)   
    nhid   = extract_spk_cnt(spk_cnt, 4)

    import pickle
    fh = file('/home/eneftci/Projects/code/python/dtp/Results/bp_100.pkl','r')
    cgpubp = np.array(pickle.load(fh)['test'])
    fh = file('/home/eneftci/Projects/code/python/dtp/Results/bp_100.pkl','r')
    tbp = np.array(pickle.load(fh)['epoch'])

    ss = np.cumsum((nv*nh + nh*nc + nv*nh + nh*nc)*np.ones(2000)*50000)


    #plot(acc_hist[:,1],np.cumsum(nvis*nh + nhid*nc + 2*nout*nc + (nerr1+nerr2)*(nc+nh)).astype('float'),'r.-', linewidth=3)
    plot(1-cgpubp/100,ss[tbp-1].astype('float'),'g.-', linewidth=3)
    ylabel('MACs')
    xlabel('MNIST Accuracy')

if __name__ == '__main__':
    pass
#    plot_convergence_200_200()
#    plot_convergence_500()
#    plot_convergence_200()
#    plot_convergence_100()

