import utils.generate_ras as gras
import os, sys
import experimentLib as elib
import experimentTools as et
import traceback
import pdb
import numpy as np
from utils.erbp_plotter import Plotter
import utils.file_io as fio
import json
import argparse
import datetime
import time



def parse_args():
    parser = argparse.ArgumentParser(description='eRBP DvsGesture')
    parser.add_argument('--n_epochs', type=int, default=100, help='number of epochs to train')
    parser.add_argument('--n_hidden', type=int, default=400, help='number of hidden units')
    parser.add_argument('--n_cores', type=int, default=4, help='number of cores')
    parser.add_argument('--testinterval', type=int, default=5, help='how epochs to run before testing')
    parser.add_argument('--input_size', type=int, default=64, help='size of the input (attention window or resize)')
    parser.add_argument('--attention_event_amount', type=int, default=1000, help='number of past events to compute attention window (set to 0 to disable attention window)')
    parser.add_argument('--no_save', type=bool, default=False, help='disables saving into Results directory')
    parser.add_argument('--eta', type=float, default=6e-4, help='learning rate')
    parser.add_argument('--prob_syn', type=float, default=0.65, help='probability passing a spike')
    parser.add_argument('--output', type=str, default='dvs_gesture_split', help='folder name for the results')
    parser.add_argument('--plot_as_training', action='store_true', default=False, help='plot spiketrains and weights while learning')
    parser.add_argument('--gen_data', action='store_true', default=False, help='generate train and test data')
    parser.add_argument('--test_first', action='store_true', default=False, help='run one test before starting to learn')


    parser.add_argument('--resume', type=str, default='', help='Resume training from directory')
    return parser.parse_args()

args = parse_args()
plotter = Plotter(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               '{date:%Y-%m-%d_%H:%M}_dvs_gesture'.format(date=datetime.datetime.now())))

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

    # first 5 labels: 7,2,1,0,4
    rate_class, first_class, rate_confusion_data_frame, first_confusion_data_frame, output_spikes_per_label, ouput_spikes_per_label_norm, snr, snr_per_label = elib.process_test_classification(
        context, sample_duration_test, labels_test)
    plotter.plot_output_spike_count(output_spikes_per_label, 'Output spike count per label', 1, save=True,
                                    image_title='out_spk')
    plotter.plot_output_spike_count(ouput_spikes_per_label_norm, 'Normalized ouput spike count per label', 1, save=True,
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

context = {'ncores': args.n_cores,
           'directory': 'dvs_gesture_split',
           'nv': (args.input_size * args.input_size) * 2 + 12,  # Include nc
           'nh': args.n_hidden,
           'nh2': args.n_hidden // 2,
           'nh1': args.n_hidden // 2,
           'nc': 12,
           'eta': args.eta,
           'eta_linear_decay': args.eta / float(args.n_epochs),
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
           'sigma': 0.0,
           'max_samples_train': 1176,  # useless
           'max_samples_test': 288,  # useless
           'n_samples_train': 1176,  # 1176
           'n_samples_test': 288,  # 288
           'n_epochs': args.n_epochs,  # 10
           'n_loop': 1,
           'prob_syn': args.prob_syn,
           'init_mean_bias_v': -.1,
           'init_mean_bias_h': -.1,
           'init_std_bias_v': 1e-32,
           'init_std_bias_h': 1e-32,
           'input_thr': .43,
           'input_scale': .5,
           'mean_weight': 0.0,  # useless
           'std_weight': 7.,
           'test_every': args.testinterval,
           'recurrent': False,
           'polarity': 'dual',
           'delay': 0.0,
           'attention_event_amount': args.attention_event_amount,
           'attention_window_size': args.input_size,
           'input_window_position': False,
           'only_input_position': False,
           'new_pos_weight': .1,
           'label_frequency': 500}
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
        last_perf = (-1.0, -1.0)
        new_test_data = args.gen_data
        save = True
        et.mksavedir()
        et.globaldata.context = context

        start_epoch = 0
        acc_hist = []
        snr_hist = []
        weight_stats = {}
        output_weights = []
        spkcnt = [None for i in range(context['n_epochs'])]

        if args.resume is not '':
            if args.resume[-1] is not '/':
                args.resume += '/'
            print 'Loading previous run from {}'.format(args.resume)
            et.globaldata.directory = args.resume
            M = et.load('M.pkl')
            old_context = et.load('context.pkl')
            acc_hist = et.load('acc_hist.pkl')
            spkcnt = et.load('spkcnt.pkl')
            snr_hist = et.load('snr_hist.pkl')
            bestM = et.load('bestM.pkl')
            start_epoch = acc_hist[-1][0] + 1
            context.update(old_context)

            # take n_cores and test interval from current args
            context['ncores'] = args.n_cores
            context['test_every'] = args.testinterval
            print('Restored context: {}'.format(context))
            elib.write_allparameters_rbp(M, context)


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
        if context['input_window_position']:
            max_neuron_id -= 128 * 2

        if args.resume is '':
            print('Cleaning previous weights')
            os.system('rm -rf inputs/{directory}/train/'.format(**context))
            os.system('mkdir -p inputs/{directory}/train/'.format(**context))
            elib.create_rbp_init(base_filename='inputs/{directory}/train/fwmat'.format(**context), **context)

        if new_test_data:
            os.system('rm -rf inputs/{directory}/test/'.format(**context))
            os.system('mkdir -p inputs/{directory}/test/'.format(**context))

            sample_duration_test, labels_test = gras.create_ras_from_aedat(n_samples_test,
                                                                           context['directory'], "test",
                                                                           randomize=False,
                                                                           pause_duration=context['sample_pause_test'],
                                                                           event_polarity=context['polarity'],
                                                                           cache=True,
                                                                           max_neuron_id=max_neuron_id,
                                                                           delay=context['delay'],
                                                                           attention_event_amount=context[
                                                                               'attention_event_amount'],
                                                                           attention_window_size=context[
                                                                               'attention_window_size'],
                                                                           input_window_position=context[
                                                                               'input_window_position'],
                                                                           only_input_position=context[
                                                                               'only_input_position'],
                                                                           new_pos_weight=context['new_pos_weight'],
                                                                           recurrent=context['recurrent'],
                                                                           label_frequency=context['label_frequency'])
            context['simtime_test'] = sample_duration_test[-1]
            print(context['simtime_test'])
            with open('inputs/{directory}/test/simtime.json'.format(**context), 'w+') as simtime_file:
                json.dump(
                    {'context': context, 'labels_test': labels_test, 'sample_duration_test': sample_duration_test},
                    simtime_file)
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
        print('Number of test samples: {}'.format(n_samples_test))

        # plotter.plot_2d_input_ras('{}/{}'.format(context['directory'], 'test'), 32, 0, 3)

        weight_stats = update_weight_stats(weight_stats)
        output_weights = update_output_weights(output_weights)

        if args.test_first:
            print('Early testing for epoch {}'.format(start_epoch))
            res, snr = run_classify(context, labels_test, sample_duration_test)

        for i in xrange(start_epoch, n_epochs):
            print('Epoch {} / {}'.format(i, n_epochs))

            start_execution_create_ras = time.time()
            sample_duration_train, labels_train = gras.create_ras_from_aedat(n_samples_train,
                                                                             context['directory'], "train",
                                                                             randomize=True,
                                                                             pause_duration=context[
                                                                                 'sample_pause_train'],
                                                                             event_polarity=context['polarity'],
                                                                             cache=True,
                                                                             max_neuron_id=max_neuron_id,
                                                                             delay=context['delay'],
                                                                             attention_event_amount=context[
                                                                                 'attention_event_amount'],
                                                                             attention_window_size=context[
                                                                                 'attention_window_size'],
                                                                             input_window_position=context[
                                                                                 'input_window_position'],
                                                                             only_input_position=context[
                                                                                 'only_input_position'],
                                                                             new_pos_weight=context['new_pos_weight'],
                                                                             recurrent=context['recurrent'],
                                                                             label_frequency=context['label_frequency']
                                                                             )
            print("---- create_ras_from_aedat: execution took {} minutes ----".format(int((time.time() - start_execution_create_ras)//60)))

            context['simtime_train'] = sample_duration_train[-1]
            print(context['simtime_train'])
            print('New train data: {} samples'.format(n_samples_train))
            start_execution_learn = time.time()
            ret, run_cmd = run_learn(context)
            print("---- run_learn: execution took {} minutes ----".format(int((time.time() - start_execution_learn)//60)))

            context['eta'] = context['eta'] - context['eta_linear_decay']
            spkcnt[i] = elib.get_spike_count('outputs/{directory}/train/'.format(**context))
            M = elib.process_parameters_rbp_dual(context)

            output_weights = update_output_weights(output_weights)
            weight_stats = update_weight_stats(weight_stats)

            if args.plot_as_training:
                plotter.plot_weight_matrix('inputs/{}/train/fwmat_{}.mtx'.format(context['directory'], '{}'), save=True)
                plotter.plot_weight_histogram('inputs/{}/train/fwmat_{}.mtx'.format(context['directory'], '{}'),
                                              nh1=context['nh1'], save=True)
                plotter.plot_ras_spikes('outputs/{}/train/coba.*.{}.ras'.format(context['directory'], '{}'),
                                        start=sample_duration_train[-3],
                                        end=sample_duration_train[-1] - context['sample_pause_train'],
                                        layers=['vis', 'hid', 'out'],
                                        res=context['nv'] - context['nc'],
                                        number_of_classes=context['nc'],
                                        save=True,
                                        input_att_window=context['input_window_position'])


            print("---- learning epoch iteration took {} minutes ----".format(int((time.time() - start_execution_create_ras)//60)))
            if test_every > 0:
                if i % test_every == 0:
                    res, snr = run_classify(context, labels_test, sample_duration_test)
                    acc_hist.append([i, res])
                    snr_hist.append([i, snr])
                    if res > last_perf:
                        last_perf = res
                        bestM = elib.read_allparamters_dual(context)
                    if save:
                        et.globaldata.context = context
                        et.save()
                        et.save(sys.argv, 'sysargv.pkl')
                        et.save(M, 'M.pkl')
                        et.save(bestM, 'bestM.pkl')
                        et.save(args, 'args.pkl')
                        et.save(spkcnt, 'spkcnt.pkl')
                        et.save(acc_hist, 'acc_hist.pkl')
                        et.save(snr_hist, 'snr_hist.pkl')
                        et.annotate('res', text=str(acc_hist))
                        et.save(context, 'context.pkl')
                        elib.textannotate('last_res', text=str(acc_hist))
                        elib.textannotate('last_dir', text=et.globaldata.directory)



        plotter.plot_weight_stats(weight_stats, save=True)
        plotter.plot_output_weights_over_time(output_weights, save=True)
        if len(acc_hist) > 0:
            plotter.plot_accuracy_rate_first(acc_hist, save=True)
        if len(snr_hist) > 0:
            print(snr_hist)

        if save:
            et.globaldata.context = context
            et.save()
            et.save(context, 'context.pkl')
            et.save(sys.argv, 'sysargv.pkl')
            et.save(M, 'M.pkl')
            et.save(spkcnt, 'spkcnt.pkl')
            et.save(acc_hist, 'acc_hist.pkl')
            et.save(snr_hist, 'snr_hist.pkl')
            et.annotate('res', text=str(acc_hist))
            et.save(bestM, 'bestM.pkl')

            elib.textannotate('last_res', text=str(acc_hist))
            elib.textannotate('last_dir', text=et.globaldata.directory)
    except:
        type, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)
