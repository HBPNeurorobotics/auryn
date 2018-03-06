import jaer_data_loader as jloader
import numpy as np
import random
import struct
import pandas as pd
import os

from tqdm import tqdm


def create_ras_from_aedat(n_samples, max_samples, exp_directory, test_or_train, labels_name, randomize=False,
                          pause_duration=0, cache=False):
    if n_samples > max_samples:
        print("{n} has to be smaller than {max}".format(n=n_samples, max=max_samples))
        return

    filename = "input"
    os.system('rm inputs/{}/{}/{}.ras'.format(exp_directory, test_or_train, filename))

    if randomize:
        sample_ids = random.sample(range(max_samples), n_samples)
    else:
        sample_ids = range(n_samples)

    tot = test_or_train.capitalize()
    labels = read_labels(labels_name)
    max_neuron_id = 32 * 32
    current_timestamp = 0.
    df_list = []
    label_list = []
    sample_duration_list = []

    print('\nloading {} data:'.format(test_or_train))
    with pd.HDFStore(
            'data/{exp_dir}_{test_or_train}.h5'.format(exp_dir=exp_directory, test_or_train=test_or_train)) as store:
        for sample_id in tqdm(sample_ids):
            key = 'm{mod}/s{sample_id}'.format(sample_id=sample_id,
                                               mod=sample_id % 10)
            if cache and key in store:
                df_concat = store[key]
            else:
                timestamps, neuron_id, pol = load_events_from_aedat(
                    "data/{test_or_train}_{exp_dir}/{tot}{s_id}.aedat".format(test_or_train=test_or_train,
                                                                              exp_dir=exp_directory, s_id=sample_id + 1,
                                                                              tot=tot))
                df = pd.DataFrame({'ts': timestamps, 'n_id': neuron_id, 'pol': pol})

                df = df[df.pol == 1]
                df = df.drop('pol', axis=1)

                label_spikes = get_label_spikes(timestamps)
                df2 = pd.DataFrame({'ts': label_spikes, 'n_id': labels[sample_id] + max_neuron_id})

                df_concat = df.append(df2)
                df_concat.sort_values(by=['ts'], inplace=True)
                if cache:
                    store[key] = df_concat
            label_list.append(labels[sample_id])
            df_concat.ts = df_concat.ts.add(current_timestamp)
            #  df_list.append(df_concat)
            write_on_ras(df_concat, exp_directory, test_or_train, filename)
            current_timestamp = df_concat.ts.values[-1] + pause_duration
            sample_duration_list.append(current_timestamp)

    #  write_new_ras(df_list, exp_directory, test_or_train, "input")
    return sample_duration_list, label_list


def get_label_spikes(timestamps):
    first_timestamp = timestamps[0]
    current_timestamp = timestamps[-1]
    sample_length = current_timestamp - first_timestamp
    assert first_timestamp < current_timestamp, "{} vs {}".format(first_timestamp, current_timestamp)
    return np.sort(np.random.sample(np.random.poisson(sample_length * 2500))) * sample_length + first_timestamp


def read_labels(labels_name):
    with open("data/{name}".format(name=labels_name), 'rb') as flbl:
        magic, num = struct.unpack(">II", flbl.read(8))
        labels = np.fromfile(flbl, dtype=np.int8)
    return labels


def load_events_from_aedat(file_path):
    timestamps, xaddr, yaddr, pol = jloader.load_jaer(file_path, debug=0)
    timestamps = np.array(timestamps).astype(float)
    timestamps *= 1e-6
    if timestamps[0] > timestamps[-1]:
        timestamps = restore_ts_order(timestamps)
    timestamps -= min(timestamps)

    xaddr = np.array(xaddr) // 4  # group to 32
    yaddr = np.array(yaddr) // 4  # group to 32
    neuron_id = (yaddr * 32) + xaddr  # group neuron_id to 32x32

    return timestamps, neuron_id, pol


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
