#!/bin/python
# -----------------------------------------------------------------------------
# File Name : run_classification_mnist_online_deep.py
# Purpose:
#
# Author: Emre Neftci
#
# Creation Date : 01-04-2015
# Last Modified : Wed 12 Apr 2017 08:22:26 PM PDT
# Copyright : (c)
# Licence : GPLv2
# -----------------------------------------------------------------------------
import matplotlib
from experimentLib import *
import os, sys
import getopt
import experimentTools as et
from pylab import *
import utils.erbp_plotter as plotter
import utils.file_io as fio


def run_classify(context, labels_test):
    # Uses outputs as inputs for the matrix! This is because the weights are symmetrized and written in the output.
    os.system('rm -rf outputs/{directory}/test'.format(**context))
    os.system('mkdir -p outputs/{directory}/test/'.format(**context))
    ret = os.system('mpirun -n {ncores} ./exp_rbp_dual \
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
        --nvis {nv} \
        --nhid {nh} \
        --nout {nc} \
        --sigma {sigma}\
        '.format(**context))

    if ret == 0:
        print 'ran'

    plotter.plot_ras_spikes('outputs/{}/test/coba.*.{}.ras'.format(context['directory'], '{}'), start=0, end=2,
                            layers=['out'], res=28 * 28, number_of_classes=context['nc'], save=True)
    return float(sum(process_test_rbp(context) == labels_test)) / len(labels_test)


def run_learn(context):
    print "eta: " + str(context['eta'])
    os.system('rm -rf outputs/{directory}/train'.format(**context))
    os.system('mkdir -p outputs/{directory}/train/'.format(**context))
    run_cmd = 'mpirun -n {ncores} ./exp_rbp_dual \
        --learn true \
        --simtime {tsimtime_train} \
        --stimtime {simtime_train} \
        --record_full true \
        --record_rasters true \
        --record_rates true \
        --dir outputs/{directory}/train/ \
        --eta  {eta}\
        --prob_syn {prob_syn}\
        --fvh  inputs/{directory}/train/{fvh} \
        --fho  inputs/{directory}/train/{fho} \
        --fhh  inputs/{directory}/train/{fhh} \
        --foe  inputs/{directory}/train/{foe} \
        --feo  inputs/{directory}/train/{feo} \
        --fve  inputs/{directory}/train/{fve} \
        --feh  inputs/{directory}/train/{feh} \
        --ip_v inputs/{directory}/train/{ip_v}\
        --gate_low  {gate_low}\
        --gate_high  {gate_high}\
        --sigma {sigma}\
        --nvis {nv} \
        --nhid {nh} \
        --nout {nc} \
        '.format(**context)
    ret = os.system(run_cmd)
    return ret, run_cmd


# Parameters (move to json or similar at some point)


context = {'ncores': 4,
           'directory': 'mnist_online_deep_dual_regate',
           'nv': 784 + 10,  # Include nc
           'nh': 400,
           'nh2': 200,
           'nh1': 200,
           'nc': 10,
           'eta': 6.0e-4,
           'ncpl': 1,
           'gate_low': -.6,
           'gate_high': .6,
           'fvh': 'fwmat_vh.mtx',
           'fho': 'fwmat_ho.mtx',
           'fhh': 'fwmat_hh.mtx',
           'foe': 'fwmat_oe.mtx',
           'feo': 'fwmat_eo.mtx',
           'fve': 'fwmat_ve.mtx',
           'feh': 'fwmat_eh.mtx',
           'ip_v': 'input_current_file',  # Hard-coded
           'beta_prm': 1.0,
           'tau_rec': 4e-3,
           'tau_ref': 4e-3,
           'seed': 32412,
           'min_p': 1e-5,
           'max_p': .98,
           'binary': False,
           'sample_duration_train': .25,  # Includes pause,
           'sample_pause_train': 0.00,
           'sample_duration_test': .4,  # Includes pause,
           'sample_pause_test': 0.,
           'sigma': 0e-3,
           'n_samples_train': 500,
           'n_samples_test': 10000,
           'n_epochs': 10,  # 60
           'n_loop': 1,
           'prob_syn': .65,
           'init_mean_bias_v': -.1,
           'init_mean_bias_h': -.1,
           'init_std_bias_v': 1e-32,
           'init_std_bias_h': 1e-32,
           'input_thr': .43,
           'input_scale': .5,
           'mean_weight': .0,
           'std_weight': 7.0,
           'test_data_url': 'data/t10k-images-idx3-ubyte',
           'test_labels_url': 'data/t10k-labels-idx1-ubyte',
           'train_data_url': 'data/train-images-idx3-ubyte',
           'train_labels_url': 'data/train-labels-idx1-ubyte',
           'test_every': 2,
           'recurrent': False}  # never test

context['eta_orig'] = context['eta']


def read_file(filename, context):
    os.system('../tools/aubs -i outputs/{directory}/train/{0} -o /tmp/h'.format(filename, **context))
    return np.loadtxt('/tmp/h')


def update_weight_stats():
    global weight_stats
    stat_dict = (
        fio.get_weight_stats('inputs/{}/train/fwmat_{}.mtx'.format(context['directory'], '{}'), context))
    if len(weight_stats) > 0:
        for key in stat_dict.keys():
            weight_stats[key] += stat_dict[key]
    else:
        weight_stats = stat_dict


if __name__ == '__main__':
    hot_init = False  # True: Start from last iteration
    save = False
    last_perf = 0.1
    directory = None
    bestM = None
    generate = False

    optlist, args = getopt.getopt(sys.argv[1:], 'hn:t:d:sgc:')
    for o, a in optlist:
        if o == '-h':  # hot init
            print 'Hot start'
            hot_init = True
        if o == '-n':  # change number of epochs
            context['n_epochs'] = int(a)
        if o == '-t':
            context['test_every'] = int(a)
        if o == '-d':
            directory = a
        if o == '-s':
            save = True
        if o == '-g':
            generate = True
        if o == '-c':  # custom parameters passed through input arguments
            context.update(eval(a))

    context['nc_perlabel'] = context['nc'] / 10
    context['simtime_train'] = context['n_samples_train'] * (
            context['sample_duration_train'] + context['sample_pause_train'])
    context['tsimtime_train'] = context['n_samples_train'] * (
            context['sample_duration_train'] + context['sample_pause_train'])
    context['simtime_test'] = context['n_samples_test'] * (
            context['sample_duration_test'] + context['sample_pause_test'])

    n_samples_train = context['n_samples_train']
    n_samples_test = context['n_samples_test']
    n_epochs = context['n_epochs']
    test_every = context['test_every']  # never test

    # For convenience
    nv = context['nv']
    nh = context['nh']
    nc = context['nc']

    if not hot_init and directory is None:
        print 'Cold initialization..'
        os.system('rm -rf inputs/{directory}/train/'.format(**context))
        os.system('rm -rf inputs/{directory}/test/'.format(**context))
        os.system('mkdir -p inputs/{directory}/train/'.format(**context))
        os.system('mkdir -p inputs/{directory}/test/'.format(**context))
        create_rbp_init(base_filename='inputs/{directory}/train/fwmat'.format(**context), **context)

    elif directory is not None:
        print 'Loading previous run...'
        et.globaldata.directory = directory
        M = et.load('M.pkl')
        M = process_allparameters_rbp(context)
        # save_parameters(M, context)

    if test_every > 0:
        labels_test, SL_test = create_data_rbp(n_samples=n_samples_test,
                                               output_directory='{directory}/test'.format(**context),
                                               data_url=context['test_data_url'],
                                               labels_url=context['test_labels_url'],
                                               randomize=False,
                                               with_labels=False,
                                               duration_data=context['sample_duration_test'],
                                               duration_pause=context['sample_pause_test'],
                                               generate_sl=False,
                                               **context)

    if n_epochs > 0:
        eta_decay = 0 * context['eta'] / n_epochs
    else:
        eta_decay = 0

    acc_hist = []
    weight_stats = {}

    context['seed'] = 0
    spkcnt = [None for i in range(n_epochs)]
    for i in range(n_epochs):
        labels_train, SL = create_data_rbp(n_samples=n_samples_train,
                                           output_directory='{directory}/train'.format(**context),
                                           data_url=context['train_data_url'],
                                           labels_url=context['train_labels_url'],
                                           randomize=True,
                                           with_labels=True,
                                           duration_data=context['sample_duration_train'],
                                           duration_pause=context['sample_pause_train'],
                                           generate_sl=False,
                                           **context)

        ret, run_cmd = run_learn(context)

        plotter.plot_ras_spikes('outputs/{}/train/coba.*.{}.ras'.format(context['directory'], '{}'), start=0,
                                end=2,
                                layers=['vis', 'hid', 'out'], res=28 * 28, number_of_classes=context['nc'],
                                save=True)

        context['eta'] = context['eta'] - eta_decay
        spkcnt[i] = get_spike_count('outputs/{directory}/train'.format(**context))

        M = process_parameters_rbp_dual(context)

        plotter.plot_weight_matrix('inputs/{}/train/fwmat_{}.mtx'.format(context['directory'], '{}'), save=True)
        plotter.plot_weight_histogram('inputs/{}/train/fwmat_{}.mtx'.format(context['directory'], '{}'),
                                      nh1=context['nh1'], save=True)
        update_weight_stats()
        # Monitor SBM progress
        if test_every > 0:
            if i % test_every == test_every - 1:
                res = run_classify(context, labels_test)
                acc_hist.append([i, res])
                print res
                if res > last_perf:
                    last_perf = res
                    bestM = read_allparamters_dual(context)
        #       # Necessary otherwise will only train on 1000 images total.
        context['seed'] = None

    #############
    if n_epochs == 0 and test_every > 0:  # Test only
        print 'Test only'
        res = run_classify(context, labels_test)
        acc_hist.append([0, res])
        print res

    plotter.plot_weight_stats(weight_stats, save=True)
    #
    M = read_allparamters_dual(context)
    d = et.mksavedir()
    et.globaldata.context = context
    et.save()
    et.save(context, 'context.pkl')
    et.save(sys.argv, 'sysargv.pkl')
    et.save(M, 'M.pkl')
    et.save(spkcnt, 'spkcnt.pkl')
    et.save(bestM, 'bestM.pkl')
    et.save(acc_hist, 'acc_hist.pkl')
    et.annotate('res', text=str(acc_hist))

    textannotate('last_res', text=str(acc_hist))
    textannotate('last_dir', text=d)
#
#
#
#
#
#
#
