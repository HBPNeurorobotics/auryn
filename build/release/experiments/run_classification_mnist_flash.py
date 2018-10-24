import utils.generate_ras as gras
import os, sys
import experimentLib as elib
import experimentTools as et
import traceback
import pdb
import numpy as np
import utils.erbp_plotter as plotter
import utils.file_io as fio
import json


# "-m yappi" to profile


def run_classify(context, labels_test, sample_duration_test):
    # Uses outputs as inputs for the matrix! This is because the weights are symmetrized and written in the output.
    os.system('rm -rf outputs/{directory}/test'.format(**context))
    os.system('mkdir -p outputs/{directory}/test/'.format(**context))
    ret = os.system('mpirun -n {ncores} ./exp_rbp_flash \
        --learn false \
        --eta 0.\
        --simtime {simtime_test} \
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
        print('ran')

    plotter.plot_ras_spikes('outputs/{}/test/coba.*.{}.ras'.format(context['directory'], '{}'), start=0, end=9,
                            layers=['out'], res=context['nv'] - context['nc'], save=True)
    # first 5 labels: 7,2,1,0,4
    rate_class, first_class, rate_confusion_data_frame, first_confusion_data_frame, output_spikes_per_label, output_spikes_per_label_norm, snr, snr_per_label = elib.process_test_classification(
        context, sample_duration_test, labels_test)
    plotter.plot_output_spike_count(output_spikes_per_label, 'Output spike count per label', 0, save=True,
                                    image_title='out_spk')
    plotter.plot_output_spike_count(output_spikes_per_label_norm, 'Normalized ouput spike count per label', 0, save=True,
                                    image_title='out_spk_norm')
    plotter.plot_confusion_matrix(rate_confusion_data_frame, save=True)
    # plotter.plot_confusion_matrix(first_confusion_data_frame, save=True)
    print('snr per label: {}'.format(snr_per_label))
    print('total snr: {}'.format(snr))
    return (rate_class, first_class), (snr_per_label, snr)


def run_learn(context):
    print("eta: " + str(context['eta']))
    os.system('rm -rf outputs/{directory}/train'.format(**context))
    os.system('mkdir -p outputs/{directory}/train/'.format(**context))
    run_cmd = 'mpirun -n {ncores} ./exp_rbp_flash \
        --learn true \
        --simtime {simtime_train} \
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


context = {'ncores': 4,
           'directory': 'dvs_mnist_flash',
           'nv': (32 * 32) + 10,  # Include nc
           'nh': 400,
           'nh2': 200,
           'nh1': 200,
           'nc': 10,
           'eta': 6e-4,
           'eta_decay': 0.9,
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
           'ip_v': 'input.ras',  # Hard-coded
           'beta_prm': 1.0,
           'tau_rec': 4e-3,
           'tau_ref': 4e-3,
           'min_p': 1e-5,
           'max_p': .98,
           'binary': False,
           'sample_pause_train': 0.4,
           'sample_pause_test': 0.4,
           'sigma': 0e-3,
           'max_samples_train': 60000,
           'max_samples_test': 10000,
           'n_samples_train': 5000,
           'n_samples_test': 10000,
           'n_epochs': 30,  # 60
           'n_loop': 1,
           'prob_syn': 0.65,
           'init_mean_bias_v': -.1,
           'init_mean_bias_h': -.1,
           'init_std_bias_v': 1e-32,
           'init_std_bias_h': 1e-32,
           'test_labels_name': 't10k-labels-idx1-ubyte',
           'train_labels_name': 'train-labels-idx1-ubyte',
           'input_thr': .43,
           'input_scale': .5,
           'mean_weight': .0,
           'std_weight': .01,
           'test_every': 1,
           'recurrent': False,
           'polarity': 'on',
           'label_frequency': 200,
           'no_noise': False}

context['eta_orig'] = context['eta']


def update_weight_stats(weight_stats):
    stat_dict = (
        fio.get_weight_stats('inputs/{}/train/fwmat_{}.mtx'.format(context['directory'], '{}'), context))
    if len(weight_stats) > 0:
        for key in stat_dict.keys():
            weight_stats[key] += stat_dict[key]
    else:
        weight_stats = stat_dict
    return weight_stats


def update_output_weights(output_weights):
    output_mtx = fio.mtx_file_to_matrix('inputs/{}/train/fwmat_ho.mtx'.format(context['directory']))
    output_mtx = output_mtx[200:, :].flatten()
    output_weights.append(output_mtx)
    return output_weights


if __name__ == '__main__':
    try:
        last_perf = (0.0, 0.0)
        init = False
        new_test_data = True
        test = True
        save = False

        folder = '064__19-07-2018'
        directory = 'Results/{}/'.format(folder)
        #directory = None
        if directory is not None:
            print 'Loading previous run...'
            et.globaldata.directory = directory
            M = et.load('M.pkl')
            old_context = et.load('context.pkl')
            context.update(old_context)
            elib.write_allparameters_rbp(M, context)
            context['test_every'] = 20
            context['n_epochs'] = 0
            context['n_samples_train'] = 1

        test_labels_name = context['test_labels_name']
        train_labels_name = context['train_labels_name']
        max_samples_train = context['max_samples_train']
        max_samples_test = context['max_samples_test']
        n_samples_train = context['n_samples_train']
        n_samples_test = context['n_samples_test']
        n_epochs = context['n_epochs']
        test_every = context['test_every']
        nv = context['nv']
        nh = context['nh']
        nc = context['nc']
        max_neuron_id = nv - nc

        if init:
            os.system('rm -rf inputs/{directory}/train/'.format(**context))
            os.system('mkdir -p inputs/{directory}/train/'.format(**context))
            elib.create_rbp_init(base_filename='inputs/{directory}/train/fwmat'.format(**context), **context)

        if new_test_data:
            os.system('rm -rf inputs/{directory}/test/'.format(**context))
            os.system('mkdir -p inputs/{directory}/test/'.format(**context))
            sample_duration_test, labels_test = gras.create_ras_from_aedat(n_samples_test, context['directory'], "test",
                                                                           test_labels_name, randomize=False,
                                                                           pause_duration=context['sample_pause_test'],
                                                                           cache=True,
                                                                           event_polarity=context['polarity'],
                                                                           max_neuron_id=max_neuron_id,
                                                                           label_frequency=context['label_frequency'],
                                                                           no_noise=context['no_noise'])
            context['simtime_test'] = sample_duration_test[-1]
            print(context['simtime_test'])
            with open('inputs/{directory}/test/simtime.json'.format(**context), 'w+') as simtime_file:
                json.dump(
                    {'context': context, 'labels_test': labels_test, 'sample_duration_test': sample_duration_test},
                    simtime_file)
                print('New test data : {}\n{}\n{}'.format(n_samples_test, labels_test, sample_duration_test))
        else:
            with open('inputs/{directory}/test/simtime.json'.format(**context), 'r') as simtime_file:
                old_test_sim = json.load(simtime_file)
                read_n_samples = old_test_sim['context']['n_samples_test']
                read_polarity = old_test_sim['context']['polarity']
                if int(read_n_samples) != n_samples_test or read_polarity != context['polarity']:
                    print("Current test ras file does not fit current context.")
                    sys.exit()
                labels_test = old_test_sim['labels_test']
                sample_duration_test = old_test_sim['sample_duration_test']
                context['simtime_test'] = sample_duration_test[-1]
                print(context['simtime_test'])
                print('Old test data : {}\n{}\n{}'.format(n_samples_test, labels_test, sample_duration_test))

        acc_hist = []
        snr_hist = []
        weight_stats = {}
        output_weights = []
        # context['seed'] = 0
        spkcnt = [None for i in range(n_epochs)]

        if test:
            res, snr = run_classify(context, labels_test, sample_duration_test)
            acc_hist.append([0, res])
            snr_hist.append([0, snr])

        plotter.plot_weight_matrix('inputs/{}/train/fwmat_{}.mtx'.format(context['directory'], '{}'), save=True)
        plotter.plot_weight_histogram('inputs/{}/train/fwmat_{}.mtx'.format(context['directory'], '{}'),
                                      nh1=context['nh1'], save=True)

        weight_stats = update_weight_stats(weight_stats)
        output_weights = update_output_weights(output_weights)
        for i in xrange(n_epochs):
            sample_duration_train, labels_train = gras.create_ras_from_aedat(n_samples_train, context['directory'],
                                                                             "train", train_labels_name, randomize=True,
                                                                             pause_duration=context[
                                                                                 'sample_pause_train'], cache=True,
                                                                             event_polarity=context['polarity'],
                                                                             max_neuron_id=max_neuron_id,
                                                                             label_frequency=context['label_frequency'],
                                                                             no_noise=context['no_noise'])
            context['simtime_train'] = sample_duration_train[-1]
            print('New train data : {}\n{}\n{}'.format(n_samples_train, labels_train, sample_duration_train))
            ret, run_cmd = run_learn(context)

            plotter.plot_ras_spikes('outputs/{}/train/coba.*.{}.ras'.format(context['directory'], '{}'), start=sample_duration_train[-3],
                                    end=sample_duration_train[-1] - context['sample_pause_train'],
                                    layers=['vis', 'hid', 'out'],
                                    res=context['nv'] - context['nc'], save=True)

            context['eta'] = context['eta'] * context['eta_decay']
            spkcnt[i] = elib.get_spike_count('outputs/{directory}/train/'.format(**context))
            M = elib.process_parameters_rbp_dual(context)

            plotter.plot_weight_matrix('inputs/{}/train/fwmat_{}.mtx'.format(context['directory'], '{}'), save=True)
            plotter.plot_weight_histogram('inputs/{}/train/fwmat_{}.mtx'.format(context['directory'], '{}'),
                                          nh1=context['nh1'], save=True)
            output_weights = update_output_weights(output_weights)
            weight_stats = update_weight_stats(weight_stats)
            if test_every > 0:
                if i % test_every == test_every - 1:
                    res, snr = run_classify(context, labels_test, sample_duration_test)
                    snr_hist.append([i + 1, snr])
                    acc_hist.append([i + 1, res])
                    if res > last_perf:
                        last_perf = res
                        bestM = elib.read_allparamters_dual(context)
            # context['seed'] = None

        plotter.plot_weight_stats(weight_stats, save=True)
        plotter.plot_output_weights_over_time(output_weights, save=True)
        if len(acc_hist) > 0:
            plotter.plot_accuracy_rate_first(acc_hist, save=True)
        if len(snr_hist) > 0:
            print(snr_hist)
        if save:
            M = elib.read_allparamters_dual(context)
            d = et.mksavedir()
            et.globaldata.context = context
            et.save()
            et.save(context, 'context.pkl')
            et.save(sys.argv, 'sysargv.pkl')
            et.save(M, 'M.pkl')
            et.save(spkcnt, 'spkcnt.pkl')
            et.save(bestM, 'bestM.pkl')
            et.save(acc_hist, 'acc_hist.pkl')
            et.save(snr_hist, 'snr_hist.pkl')
            et.annotate('res', text=str(acc_hist))

            elib.textannotate('last_res', text=str(acc_hist))
            elib.textannotate('last_dir', text=d)
    except:
        type, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)
