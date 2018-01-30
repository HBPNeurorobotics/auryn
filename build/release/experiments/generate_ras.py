import utils.jaer_data_loader as jloader
import numpy as np
import random
import struct

def create_event_data_rbp(n_samples, max_samples, exp_directory, test_or_train, labels_name, randomize=False,
                          pause_duration=0):

    if n_samples > max_samples:
        print("{n} has to be smaller than {max}".format(n=n_samples, max=max_samples))
        return

    labels = read_labels(labels_name)
    current_timestamp = 0

    used_labels = []
    next_sample_ts = []
    time = np.array([])
    n_id = np.array([])

    if randomize:
        sample_ids = random.sample(range(1, max_samples + 1), n_samples)
    else:
        sample_ids = range(1, n_samples + 1)

    tot = test_or_train.capitalize()

    for sample_id in sample_ids:
        timestamps, neuron_id = load_events_from_aedat("data/{test_or_train}_{exp_dir}/{tot}{i}.aedat".format(test_or_train=test_or_train, exp_dir=exp_directory, i=sample_id, tot=tot))
        timestamps += current_timestamp

        current_timestamp = max(timestamps)
        time = np.append(time, timestamps)
        n_id = np.append(n_id, neuron_id)
        used_labels.append(labels[sample_id])
        next_sample_ts.append(current_timestamp)

    write_ras(time, n_id, exp_directory, test_or_train, "input")
    write_ras(np.array(next_sample_ts), np.array(used_labels), exp_directory, test_or_train, "labels")

def read_labels(labels_name):
    with open("data/{name}".format(name=labels_name), 'rb') as flbl:
        magic, num = struct.unpack(">II", flbl.read(8))
        labels = np.fromfile(flbl, dtype=np.int8)
    return labels

def load_events_from_aedat(file_path):
    timestamps, xaddr, yaddr, pol = jloader.load_jaer(file_path, debug=0)
    timestamps = np.array(timestamps)
    timestamps -= min(timestamps)

    xaddr = np.array(xaddr)
    yaddr = np.array(yaddr)
    neuron_id = (yaddr * 128) + xaddr

    return timestamps, neuron_id

def write_ras(time, id, exp_directory, test_or_train, ras_file_name):
    file = open("inputs/{exp_dir}/{test_or_train}/{file_name}.ras".format(exp_dir=exp_directory, test_or_train=test_or_train, file_name=ras_file_name),"w+")

    time = time * 1e-6
    id = id.astype(int)

    for ts, idx in zip(np.nditer(time), np.nditer(id)):
        file.write("{ts:f} {idx}\n".format(ts=ts, idx=idx))

    file.close()
