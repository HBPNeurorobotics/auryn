import utils.generate_ras as gras
import os, sys
import experimentLib as elib
import traceback, code
import pdb

context={'ncores':8,
         'directory' : 'dvs_mnist_flash',
         'nv' : 128*128+10, #Include nc
         'nh' : 400,
         'nh2' : 200,
         'nh1' : 200,
         'nc' : 10,
         'eta': 6.0e-4,
         'ncpl' : 1,
         'gate_low' : -.6,
         'gate_high' : .6,
         'fvh': 'fwmat_vh.mtx',
         'fho': 'fwmat_ho.mtx',
         'fhh': 'fwmat_hh.mtx',
         'foe': 'fwmat_oe.mtx',
         'feo': 'fwmat_eo.mtx',
         'fve': 'fwmat_ve.mtx',
         'feh': 'fwmat_eh.mtx',
         'ip_v': 'input_ras_file', #Hard-coded
         'beta_prm' : 1.0,
         'tau_rec' : 4e-3,
         'tau_ref' : 4e-3,
         'seed' : 32412,
         'min_p' : 1e-5,
         'max_p' : .98,
         'binary' : False,
         'sample_pause_train' : 0.,
         'sample_pause_test' : 0.,
         'sigma' : 0e-3,
         'max_samples_train' : 60000,
         'max_samples_test' : 10000,
         'n_samples_train' : 10,
         'n_samples_test' : 10000,
         'n_epochs' : 1, #60
         'n_loop' : 1,
         'prob_syn' : .65,
         'init_mean_bias_v' : -.1,
         'init_mean_bias_h' : -.1,
         'init_std_bias_v' : 1e-32,
         'init_std_bias_h' : 1e-32,
         'test_labels_name' :  't10k-labels-idx1-ubyte',
         'train_labels_name' : 'train-labels-idx1-ubyte',
         'input_thr' : .43,
         'input_scale' : .5,
         'mean_weight' : .0,
         'std_weight' : 7.0,
         'test_every' : 5} #never test

context['eta_orig'] = context['eta']

if __name__ == '__main__':
    try:
        test_labels_name = context['test_labels_name']
        train_labels_name = context['train_labels_name']
        max_samples_train = context['max_samples_train']
        max_samples_test = context['max_samples_test']
        n_samples_train = context['n_samples_train']
        n_samples_test  = context['n_samples_test']
        n_epochs        = context['n_epochs']
        test_every      = context['test_every']
        nv = context['nv']
        nh = context['nh']
        nc = context['nc']

        os.system('rm -rf inputs/{directory}/train/'.format(**context))
        os.system('rm -rf inputs/{directory}/test/'.format(**context))
        os.system('mkdir -p inputs/{directory}/train/' .format(**context))
        os.system('mkdir -p inputs/{directory}/test/' .format(**context))

        elib.create_rbp_init(base_filename = 'inputs/{directory}/train/fwmat'.format(**context), **context)

        gras.create_event_data_rbp(n_samples_test, max_samples_test, '{directory}'.format(**context), "test", test_labels_name, randomize = False)
    except:
        type, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)
