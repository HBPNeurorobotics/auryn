import jaer_data_loader as jloader
import numpy as np
import random
import struct
import pandas as pd
import time

from tqdm import tqdm


def create_event_data_rbp(n_samples, max_samples, exp_directory, test_or_train, labels_name, randomize=False,
                          pause_duration=0):
    if n_samples > max_samples:
        print("{n} has to be smaller than {max}".format(n=n_samples, max=max_samples))
        return

    labels = read_labels(labels_name)
    max_neuron_id = 128 * 128
    current_timestamp = 0.

    if randomize:
        sample_ids = random.sample(range(max_samples), n_samples)
    else:
        sample_ids = range(n_samples)

    tot = test_or_train.capitalize()

    t_before = time.clock()
    df_list = []
    for i, sample_id in tqdm(enumerate(sample_ids)):
        timestamps, neuron_id = load_events_from_aedat(
            "data/{test_or_train}_{exp_dir}/{tot}{s_id}.aedat".format(test_or_train=test_or_train,
                                                                      exp_dir=exp_directory, s_id=sample_id + 1,
                                                                      tot=tot))
        timestamps += current_timestamp
        df = pd.DataFrame({'ts': timestamps, 'n_id': neuron_id})

        label_spikes = get_label_spikes(timestamps)
        df2 = pd.DataFrame({'ts': label_spikes, 'n_id': labels[sample_id] + max_neuron_id})

        df_concat = df.append(df2)
        df_concat.sort_values(by=['ts'], inplace=True)
        df_list.append(df_concat)

        current_timestamp = df_concat.ts.values[-1] + pause_duration

    # dfs = pd.concat(df_list)
    print('time: {t:.3f}s'.format(t=(time.clock() - t_before)))
    write_ras(df_list, exp_directory, test_or_train, "input")


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
    f = open(
        "inputs/{exp_dir}/{test_or_train}/{file_name}.ras".format(exp_dir=exp_directory, test_or_train=test_or_train,
                                                                  file_name=ras_file_name), "w+")
    for df in df_list:
        for ts, idx in zip(df['ts'].values, df['n_id'].values):
            f.write("{ts:f} {idx}\n".format(ts=ts, idx=idx))

    f.close()
