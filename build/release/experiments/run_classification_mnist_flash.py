import utils.generate_ras as gras
import os, sys
import experimentLib as elib
import experimentTools as et
import traceback
import pdb


# "-m yappi" to profile

def run_learn(context):
    print "eta: " + str(context['eta'])
    os.system('rm -rf outputs/{directory}/train'.format(**context))
    os.system('mkdir -p outputs/{directory}/train/'.format(**context))
    run_cmd = 'mpirun -n {ncores} ./exp_rbp_flash \
        --learn true \
        --simtime {tsimtime_train} \
        --stimtime {simtime_train} \
        --record_full false \
        --record_rasters false \
        --record_rates false \
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
           'nv': 128 * 128 + 10,  # Include nc
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
           'ip_v': 'input_ras_file',  # Hard-coded
           'beta_prm': 1.0,
           'tau_rec': 4e-3,
           'tau_ref': 4e-3,
           'seed': 32412,
           'min_p': 1e-5,
           'max_p': .98,
           'binary': False,
           'sample_pause_train': 2.,
           'sample_pause_test': 2.,
           'sigma': 0e-3,
           'max_samples_train': 60000,
           'max_samples_test': 10000,
           'n_samples_train': 10,
           'n_samples_test': 100,
           'n_epochs': 2,  # 60
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
           'test_every': 5}

context['eta_orig'] = context['eta']

if __name__ == '__main__':
    try:
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

        os.system('rm -rf inputs/{directory}/train/'.format(**context))
        os.system('mkdir -p inputs/{directory}/train/'.format(**context))

        #elib.create_rbp_init(base_filename='inputs/{directory}/train/fwmat'.format(**context), **context)

        new_test_data = True
        if new_test_data:
            os.system('rm -rf inputs/{directory}/test/'.format(**context))
            os.system('mkdir -p inputs/{directory}/test/'.format(**context))
            simtime = gras.create_ras_from_aedat(n_samples_test, max_samples_test, context['directory'], "test",
                                                 test_labels_name,
                                                 randomize=False, pause_duration=context['sample_pause_test'], cache=True)
            context['simtime_train'] = simtime
        else:
            with open('inputs/{directory}/test/input.ras'.format(**context)) as test_input:
                print('-1:')
                print(list(test_input)[-1])
                print('-2:')
                print(list(test_input)[-2])

        eta_decay = context['eta'] / n_epochs
        acc_hist = []

        context['seed'] = 0
        spkcnt = [None for i in range(n_epochs)]

        for i in range(n_epochs):
            tsimtime = gras.create_ras_from_aedat(n_samples_train, max_samples_train, context['directory'], "train",
                                                  train_labels_name, randomize=True,
                                                  pause_duration=context['sample_pause_train'],
                                                  cache=True)
            context['tsimtime_train'] = tsimtime
            # ret, run_cmd = run_learn(context)
            # context['eta'] = context['eta'] - eta_decay
            spkcnt[i] = elib.get_spike_count('outputs/{directory}/train'.format(**context))

            M = elib.process_parameters_rbp_dual(context)

        save = False
        if save:
            M = elib.read_allparamters_dual(context)
            d = et.mksavedir()
            et.globaldata.context = context
            et.save()
            et.save(context, 'context.pkl')
            et.save(sys.argv, 'sysargv.pkl')
            et.save(M, 'M.pkl')
            et.save(spkcnt, 'spkcnt.pkl')
            # et.save(bestM, 'bestM.pkl')
            et.save(acc_hist, 'acc_hist.pkl')
            et.annotate('res', text=str(acc_hist))

            elib.textannotate('last_res', text=str(acc_hist))
            elib.textannotate('last_dir', text=d)
    except:
        type, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)
