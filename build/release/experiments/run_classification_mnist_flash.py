import utils.generate_ras as gras
import os, sys
import experimentLib as elib
import experimentTools as et
import traceback
import pdb
import numpy as np
import utils.erbp_plotter as plotter


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

    return elib.process_test_rate_classification(context, sample_duration_test, labels_test)


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


context = {'ncores': 8,
           'directory': 'dvs_mnist_flash',
           'nv': 32 * 32 + 10,  # Include nc
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
           'ip_v': 'input.ras',  # Hard-coded
           'beta_prm': 1.0,
           'tau_rec': 4e-3,
           'tau_ref': 4e-3,
           'min_p': 1e-5,
           'max_p': .98,
           'binary': False,
           'sample_pause_train': 2.,
           'sample_pause_test': 2.,
           'sigma': 0e-3,
           'max_samples_train': 60000,
           'max_samples_test': 10000,
           'n_samples_train': 10,
           'n_samples_test': 1000,
           'n_epochs': 0,  # 60
           'n_loop': 1,
           'prob_syn': 1.,
           'init_mean_bias_v': -.1,
           'init_mean_bias_h': -.1,
           'init_std_bias_v': 1e-32,
           'init_std_bias_h': 1e-32,
           'test_labels_name': 't10k-labels-idx1-ubyte',
           'train_labels_name': 'train-labels-idx1-ubyte',
           'input_thr': .43,
           'input_scale': .5,
           'mean_weight': .0,
           'std_weight': 7.0,
           'test_every': 4}

context['eta_orig'] = context['eta']

if __name__ == '__main__':
    try:
        last_perf = 0.1
        init = False
        new_test_data = False
        test = False
        save = False

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

        if init:
            os.system('rm -rf inputs/{directory}/train/'.format(**context))
            os.system('mkdir -p inputs/{directory}/train/'.format(**context))
            elib.create_rbp_init(base_filename='inputs/{directory}/train/fwmat'.format(**context), **context)

        if new_test_data:
            os.system('rm -rf inputs/{directory}/test/'.format(**context))
            os.system('mkdir -p inputs/{directory}/test/'.format(**context))
            sample_duration_test, labels_test = gras.create_ras_from_aedat(n_samples_test, max_samples_test,
                                                                           context['directory'], "test",
                                                                           test_labels_name, randomize=False,
                                                                           pause_duration=context['sample_pause_test'],
                                                                           cache=True)
            context['simtime_test'] = sample_duration_test[-1]
            print(context['simtime_test'])
            with open('inputs/{directory}/test/simtime.txt'.format(**context), 'w+') as simtime_file:
                simtime_file.write('{}\n{}\n{}'.format(n_samples_test, labels_test, sample_duration_test))
                # print('New test data : {}\n{}\n{}'.format(n_samples_test, labels_test, sample_duration_test))
        else:
            with open('inputs/{directory}/test/simtime.txt'.format(**context), 'r') as simtime_file:
                file_string_list = simtime_file.read().splitlines()
                read_n_samples = file_string_list[0]
                if int(read_n_samples) != n_samples_test:
                    print("Current test ras file does not fit the number of test samples.")
                    sys.exit()
                labels_test = np.array(file_string_list[1][1:-1].split(','), dtype=int)
                sample_duration_test = np.array(file_string_list[2][1:-1].split(','), dtype=float)
                context['simtime_test'] = sample_duration_test[-1]
                print(context['simtime_test'])
                #  print('Old test data : {}\n{}\n{}'.format(n_samples_test, labels_test, sample_duration_test))

        # eta_decay = context['eta'] / n_epochs
        acc_hist = []
        spkcnt = [None for i in range(n_epochs)]

        plotter.plot_ras_spikes_whole('outputs/dvs_mnist_flash/train/coba.*.{}.ras', start=0, end=20,
                                      layers=['vis', 'hid', 'out'], res=32 * 32, save=False)

        for i in xrange(n_epochs):
            sample_duration_train, labels_train = gras.create_ras_from_aedat(n_samples_train, max_samples_train,
                                                                             context['directory'], "train",
                                                                             train_labels_name, randomize=True,
                                                                             pause_duration=context[
                                                                                 'sample_pause_train'],
                                                                             cache=True)
            context['simtime_train'] = sample_duration_train[-1]
            print(context['simtime_train'])
            # print('New train data : {}\n{}\n{}'.format(n_samples_train, labels_train, sample_duration_train))
            ret, run_cmd = run_learn(context)
            # context['eta'] = context['eta'] - eta_decay
            spkcnt[i] = elib.get_spike_count('outputs/{directory}/train/'.format(**context))
            M = elib.process_parameters_rbp_dual(context)
            plotter.plot_ras_spikes('outputs/{directory}/train/coba.*.c.ras', 32, 0, 10, i, True)

            if test_every > 0:
                if i % test_every == test_every - 1:
                    res = run_classify(context, labels_test, sample_duration_test)
                    acc_hist.append([i, res])
                    print res
                    if res > last_perf:
                        last_perf = res
                        bestM = elib.read_allparamters_dual(context)

        if test:
            res = run_classify(context, labels_test, sample_duration_test)
            acc_hist.append([0, res])
            print res

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
            et.annotate('res', text=str(acc_hist))

            elib.textannotate('last_res', text=str(acc_hist))
            elib.textannotate('last_dir', text=d)
    except:
        type, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)
