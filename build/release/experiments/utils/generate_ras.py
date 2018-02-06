import jaer_data_loader as jloader
import numpy as np
import random
import struct
import pandas as pd

from tqdm import tqdm


def create_ras_from_aedat(n_samples, max_samples, exp_directory, test_or_train, labels_name, randomize=False,
                          pause_duration=0, cache=False):
    if n_samples > max_samples:
        print("{n} has to be smaller than {max}".format(n=n_samples, max=max_samples))
        return

    if randomize:
        sample_ids = random.sample(range(max_samples), n_samples)
    else:
        sample_ids = range(n_samples)

    tot = test_or_train.capitalize()
    labels = read_labels(labels_name)
    max_neuron_id = 128 * 128
    current_timestamp = 0.
    df_list = []

    print('\nloading {} data:'.format(test_or_train))
    with pd.HDFStore('data/{exp_dir}.h5'.format(exp_dir=exp_directory)) as store:
        for sample_id in tqdm(sample_ids):
            key = '{test_or_train}/m{mod}/s{sample_id}'.format(test_or_train=test_or_train, sample_id=sample_id,
                                                               mod=sample_id % 10)
            if cache and key in store:
                df_concat = store[key]
            else:
                timestamps, neuron_id = load_events_from_aedat(
                    "data/{test_or_train}_{exp_dir}/{tot}{s_id}.aedat".format(test_or_train=test_or_train,
                                                                              exp_dir=exp_directory, s_id=sample_id + 1,
                                                                              tot=tot))
                df = pd.DataFrame({'ts': timestamps, 'n_id': neuron_id})

                label_spikes = get_label_spikes(timestamps)
                df2 = pd.DataFrame({'ts': label_spikes, 'n_id': labels[sample_id] + max_neuron_id})

                df_concat = df.append(df2)
                df_concat.sort_values(by=['ts'], inplace=True)
                if cache:
                    store[key] = df_concat

            df_concat.ts.add(current_timestamp)
            df_list.append(df_concat)
            current_timestamp = df_concat.ts.values[-1] + pause_duration

    write_ras(df_list, exp_directory, test_or_train, "input")
    return current_timestamp - pause_duration


def get_label_spikes(timestamps):
    first_timestamp = timestamps[0]
    current_timestamp = timestamps[-1]
    sample_length = current_timestamp - first_timestamp
    assert first_timestamp < current_timestamp, "{} vs {}".format(first_timestamp, current_timestamp)
    return np.sort(np.random.sample(np.random.poisson(sample_length * 250))) * sample_length + first_timestamp


def read_labels(labels_name):
    with open("data/{name}".format(name=labels_name), 'rb') as flbl:
        magic, num = struct.unpack(">II", flbl.read(8))
        labels = np.fromfile(flbl, dtype=np.int8)
    return labels


def load_events_from_aedat(file_path):
    timestamps, xaddr, yaddr, pol = jloader.load_jaer(file_path, debug=0)
    timestamps = np.array(timestamps).astype(float)
    if timestamps[0] > timestamps[-1]:
        timestamps = restore_ts_order(timestamps)
    timestamps -= min(timestamps)
    timestamps *= 1e-6

    xaddr = np.array(xaddr)
    yaddr = np.array(yaddr)
    neuron_id = (yaddr * 128) + xaddr

    return timestamps, neuron_id


def restore_ts_order(timestamps):
    for i in range(len(timestamps) - 1):
        if timestamps[i] > timestamps[i + 1]:
            timestamps[:i] -= 2 ** 32
            return timestamps


def write_ras(df_list, exp_directory, test_or_train, ras_file_name):
    with open(
        "inputs/{exp_dir}/{test_or_train}/{file_name}.ras".format(exp_dir=exp_directory, test_or_train=test_or_train,
                                                                  file_name=ras_file_name), "w+") as f:
        print('\nwriting {} ras:'.format(test_or_train))
        for df in tqdm(df_list):
            f.write(gen_ras_string(df))


def gen_ras_string(df):
    return ''.join(["{ts:f} {idx}\n".format(ts=ts, idx=idx) for ts, idx in zip(df['ts'].values, df['n_id'].values)])
