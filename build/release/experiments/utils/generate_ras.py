import jaer_data_handler as jloader
import numpy as np
import random
import struct
import pandas as pd
import os
import glob

from tqdm import tqdm


def create_ras_from_aedat(n_samples, exp_directory, test_or_train, labels_name='', randomize=False, pause_duration=0,
                          event_polarity='on', cache=False, max_neuron_id=32 * 32, delay=0.0):
    filename = "input"
    os.system('rm inputs/{}/{}/{}.ras'.format(exp_directory, test_or_train, filename))

    sample_names = get_file_names(exp_directory, test_or_train)
    if exp_directory == 'dvs_mnist_flash':
        labels, sample_ids = get_exp_data_flash(labels_name, n_samples, randomize, test_or_train)
        version = 'aedat'
    elif exp_directory == 'dvs_mnist_saccade':
        labels, sample_ids = get_exp_data_saccade(n_samples, randomize, test_or_train)
        version = 'aedat'
    elif exp_directory == 'dvs_gesture_split':
        labels, sample_ids = get_exp_data_gesture(n_samples, randomize, sample_names)
        version = 'aedat3'

    if n_samples > len(sample_names):
        print('Number of total files has to be bigger than number of epoch samples.')
        return
    else:
        sample_names = [sample_names[i] for i in sample_ids]

    current_timestamp = 0.
    label_list = []
    sample_duration_list = []

    print('\nloading {} data:'.format(test_or_train))
    with pd.HDFStore(
            'data/{exp_dir}/{test_or_train}_{event_pol}_{delay}.h5'.format(exp_dir=exp_directory, test_or_train=test_or_train,
                                                                   event_pol=event_polarity, delay=delay)) as store:
        for i, sample_id in enumerate(tqdm(sample_ids)):
            key = 'm{mod}/s{sample_id}'.format(sample_id=sample_id,
                                               mod=sample_id % 10)
            if cache and key in store:
                df_concat = store[key]
            else:
                timestamps, neuron_id, pol, min_ts = load_events_from_aedat(
                    sample_names[i], version)
                df = pd.DataFrame({'ts': timestamps, 'n_id': neuron_id, 'pol': pol})

                if event_polarity == 'on':
                    df = df[df.pol == 1]
                elif event_polarity == 'off':
                    df = df[df.pol == 0]
                elif event_polarity == 'dual':
                    if delay != 0.0:
                        frac = 4
                    else:
                        frac = 2
                    df.loc[df.pol == 1, 'n_id'] += int(max_neuron_id / frac)
                df = df.drop('pol', axis=1)
                if delay != 0.0:
                    df_copy = df.copy(deep=True)
                    df_copy.loc[:, 'n_id'] += int(max_neuron_id / 2)
                    df_copy.loc[:, 'ts'] += delay
                    df = df.append(df_copy)
                df2 = get_label_spikes_df(labels[i], max_neuron_id, timestamps[-1], 2500)
                df_concat = df.append(df2)
                df_concat.sort_values(by=['ts'], inplace=True)
                if cache:
                    store[key] = df_concat
            label_list.append(labels[i])
            df_concat.ts = df_concat.ts.add(current_timestamp)
            write_on_ras(df_concat, exp_directory, test_or_train, filename)
            current_timestamp = df_concat.ts.values[-1] + pause_duration
            sample_duration_list.append(current_timestamp)

    return sample_duration_list, label_list


def get_label_spikes_df(label, max_neuron_id, end_ts, frequency):
    label_spikes = get_label_spikes(end_ts, frequency)
    df2 = pd.DataFrame({'ts': label_spikes, 'n_id': label + max_neuron_id})
    return df2


def get_exp_data_saccade(n_samples, randomize, test_or_train):
    ids = []
    for i in range(0, 10000, 1000):
        if test_or_train == 'train':
            ids += range(i, i + 900)
        elif test_or_train == 'test':
            ids += range(i + 900, i + 1000)
    if randomize:
        sample_ids = random.sample(ids, n_samples)
    else:
        sample_ids = ids[:n_samples]
    labels = np.zeros(10000, dtype=int)
    for i in range(10):
        labels[i * 1000:(i + 1) * 1000] = i
    labels = [labels[i] for i in sample_ids]
    return labels, sample_ids


def get_exp_data_flash(labels_name, n_samples, randomize, test_or_train):
    if test_or_train == 'train':
        max_samples = 60000
    else:
        max_samples = 10000
    if randomize:
        sample_ids = random.sample(range(max_samples), n_samples)
    else:
        sample_ids = range(n_samples)
    labels = read_labels(labels_name)
    labels = [labels[i] for i in sample_ids]
    return labels, sample_ids


def get_exp_data_gesture(n_samples, randomize, sample_names):
    ids = range(len(sample_names))
    if randomize:
        sample_ids = random.sample(ids, n_samples)
    else:
        sample_ids = ids[:n_samples]
    labels = get_gesture_labels(sample_ids, sample_names)
    return labels, sample_ids


def get_gesture_labels(sample_ids, sample_names):
    labels = []
    for id in sample_ids:
        label = int(sample_names[id].split('__')[1].split('.')[0])  # labels start from 1, without noise
        labels.append(label)
    return labels


def get_file_names(exp_directory, test_or_train=''):
    if exp_directory == 'dvs_mnist_flash':
        path = glob.glob('data/{exp_dir}/{test_or_train}/*.aedat'.format(test_or_train=test_or_train,
                                                                         exp_dir=exp_directory))

    elif exp_directory == 'dvs_mnist_saccade':
        path = glob.glob('data/{exp_dir}/grabbed_data*/scale16/*.aedat'.format(exp_dir=exp_directory))

    elif exp_directory == 'dvs_gesture_split':
        path = glob.glob(
            'data/{exp_dir}/{test_or_train}/*.aedat'.format(exp_dir=exp_directory, test_or_train=test_or_train))
    return sorted(path)


def get_label_spikes(last_timestamp, frequency):
    return np.sort(np.random.sample(np.random.poisson(last_timestamp * frequency))) * last_timestamp


def read_labels(labels_name):
    with open('data/{name}'.format(name=labels_name), 'rb') as flbl:
        magic, num = struct.unpack('>II', flbl.read(8))
        labels = np.fromfile(flbl, dtype=np.int8)
    return labels


def load_events_from_aedat(file_path, version):
    if version == 'aedat3':
        timestamps, xaddr, yaddr, pol = jloader.load_aedat31(file_path, debug=0)
    else:
        timestamps, xaddr, yaddr, pol = jloader.load_jaer(file_path, version=version, debug=0)
    timestamps = np.array(timestamps).astype(float)
    timestamps *= 1e-6
    if timestamps[0] > timestamps[-1]:
        print('HAD TO RESTORE TS ORDER')
        timestamps = restore_ts_order(timestamps)
    min_ts = min(timestamps)
    timestamps -= min_ts
    xaddr = np.array(xaddr) // 4  # group to 32
    yaddr = np.array(yaddr) // 4  # group to 32
    neuron_id = (yaddr * 32) + xaddr  # group neuron_id to 32x32

    return timestamps, neuron_id, pol, min_ts


def restore_ts_order(timestamps):
    for i in range(len(timestamps) - 1):
        if timestamps[i] > timestamps[i + 1]:
            timestamps[:i + 1] -= (2 ** 32 * 1e-6)
            return timestamps


def write_new_ras(df_list, exp_directory, test_or_train, ras_file_name):
    # TODO make parallel
    with open(
            "inputs/{exp_dir}/{test_or_train}/{file_name}.ras".format(exp_dir=exp_directory,
                                                                      test_or_train=test_or_train,
                                                                      file_name=ras_file_name), "w+") as f:
        print('\nwriting {} ras:'.format(test_or_train))
        for df in tqdm(df_list):
            f.write(gen_ras_string(df))


def write_on_ras(df, exp_directory, test_or_train, ras_file_name):
    with open(
            "inputs/{exp_dir}/{test_or_train}/{file_name}.ras".format(exp_dir=exp_directory,
                                                                      test_or_train=test_or_train,
                                                                      file_name=ras_file_name), "a+") as f:
        f.write(gen_ras_string(df))


def gen_ras_string(df):
    return ''.join(["%f %d\n" % (ts, idx) for ts, idx in zip(df['ts'].values, df['n_id'].values)])
