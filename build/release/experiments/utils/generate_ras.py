import jaer_data_handler as jloader
import numpy as np
import random
import struct
import pandas as pd
import os
import glob
import erbp_plotter as plotter
from attention_mechanism import attention
import scipy.stats as stats

from tqdm import tqdm


def create_ras_from_aedat(n_samples, exp_directory, test_or_train, labels_name='', randomize=False, pause_duration=0,
                          event_polarity='on', cache=False, max_neuron_id=32 * 32, delay=0.0, attention_event_amount=1000,
                          attention_window_size=32, input_window_position=False,
                          only_input_position=False, new_pos_weight=0.1,
                          recurrent=False, no_noise=False, label_frequency=2500, clean_h5file=False):
    filename = "input"
    os.system('rm inputs/{}/{}/{}.ras'.format(exp_directory, test_or_train, filename))

    sample_names = get_file_names(exp_directory, test_or_train)
    if exp_directory == 'dvs_mnist_flash':
        if no_noise:
            sample_names, labels, sample_ids = get_no_noise_samples(sample_names, labels_name, exp_directory, n_samples,
                                                                    randomize, test_or_train)
        else:
            labels, sample_ids = get_exp_data_flash(labels_name, n_samples, randomize, test_or_train)
        version = 'aedat'

    elif exp_directory == 'dvs_mnist_saccade':
        labels, sample_ids = get_exp_data_saccade(n_samples, randomize, test_or_train)
        version = 'aedat'
    elif exp_directory == 'dvs_gesture_split':
        labels, sample_ids = get_exp_data_gesture(n_samples, randomize, sample_names)
        version = 'aedat3'

    if n_samples > len(sample_names):
        raise ValueError(
            'Desired number of test samples ({}) is smaller than total amount of samples ({})'.
            format(n_samples, len(sample_names)))
    else:
        sample_names = [sample_names[i] for i in sample_ids]

    current_timestamp = 0.
    sample_duration_list = []

    h5_filepath = 'data/{exp_dir}/{test_or_train}_{max_neuron_id}_{event_pol}_{delay}delay_{attention_event_amount}attention{attention_window_size}{input_window_position}_posonly{only_input_position}_{new_pos_weight}new_rec{recurrent}.h5'.format(
        exp_dir=exp_directory,
        test_or_train=test_or_train,
        event_pol=event_polarity,
        delay=delay,
        attention_event_amount=attention_event_amount,
        attention_window_size=attention_window_size,
        input_window_position=input_window_position,
        only_input_position=only_input_position,
        new_pos_weight=new_pos_weight,
        recurrent=recurrent,
        max_neuron_id=max_neuron_id)

    print('\nfilepath for h5 cache: {}'.format(h5_filepath))
    if clean_h5file:
        try:
            print('Removing previous h5 cache')
            os.remove(h5_filepath)
        except FileNotFoundError:
            print('No previous h5 cache found')
            pass

    print('loading {} data:'.format(test_or_train))
    with pd.HDFStore(h5_filepath) as store:
        for i, sample_id in enumerate(tqdm(sample_ids)):
            key = 'm{mod}/s{sample_id}'.format(sample_id=sample_id,
                                               mod=sample_id % 10)
            if cache and key in store:
                df = store[key]
                timestamps = np.array(df.ts)
            else:
                timestamps, xaddr, yaddr, pol, min_ts = load_events_from_aedat(
                    sample_names[i], version)

                if attention_event_amount == 0:
                    # we resize the event stream to the same size the attention window would be
                    group_by = 128 // attention_window_size
                    neuron_id = get_grouped_n_id(xaddr, yaddr, group_by)
                    df = pd.DataFrame({'ts': timestamps, 'n_id': neuron_id, 'pol': pol})
                else:
                    df = pd.DataFrame({'ts': timestamps, 'x': xaddr, 'y': yaddr, 'pol': pol})
                    df = attention.get_attention_df_rolling(df, attention_event_amount, attention_window_size)

                if not only_input_position:
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
                if cache:
                    store[key] = df
            if attention_event_amount != 0 and input_window_position:
                att_win_position_neurons = 2 * 128
            else:
                att_win_position_neurons = 0
            df2 = get_label_spikes_df(labels[i], max_neuron_id + att_win_position_neurons, 0, timestamps[-1],
                                      label_frequency)
            df = df.append(df2)
            if recurrent:
                df3 = pd.DataFrame()
                for c in range(1, 12):
                    df_tmp = get_label_spikes_df(c, max_neuron_id + att_win_position_neurons, 0, 1, label_frequency)
                    df3 = df3.append(df_tmp)
                df3.loc[:, 'ts'] -= 1
                df = df.append(df3)
            df.sort_values(by=['ts'], inplace=True)
            df.ts = df.ts.add(current_timestamp)
            write_on_ras(df, exp_directory, test_or_train, filename)
            current_timestamp = df.ts.values[-1] + pause_duration
            sample_duration_list.append(current_timestamp)

    return sample_duration_list, labels


def get_no_noise_samples(sample_names, labels_name, exp_directory, n_samples, randomize, test_or_train):
    labels = read_labels(labels_name)
    noise_sample_names = get_file_names('{}/noise'.format(exp_directory), test_or_train)
    noise_sample_names = [i.replace('/noise', '') for i in noise_sample_names]
    if test_or_train == 'train':
        max_samples = 48000
    else:
        max_samples = 8000
    if randomize:
        sample_ids = random.sample(range(max_samples), n_samples)
    else:
        sample_ids = range(n_samples)
    sample_names, labels = zip(*filter(lambda x: x[0] not in noise_sample_names, zip(sample_names, labels)))
    labels = [int(labels[i]) for i in sample_ids]

    return sample_names, labels, sample_ids


def get_attention_df(timestamps, xaddr, yaddr, pol, attention_window_time, attention_window_size, input_window_position,
                     max_neuron_id, frequency, attention_mechanism, attention_window_position_std, only_input_position,
                     new_pos_weight, label=0):
    dfs = []
    df = pd.DataFrame({'ts': timestamps, 'x': xaddr, 'y': yaddr, 'pol': pol})
    dvs_res = 128
    x_dist = np.zeros(dvs_res)
    y_dist = np.zeros(dvs_res)
    half_att_win = int(attention_window_size / 2)

    for i, time_window in enumerate(np.arange(0.0, max(timestamps), attention_window_time)):
        event_slice = df.loc[(df.ts >= time_window) & (df.ts < time_window + attention_window_time)]
        centroid_x, centroid_y = get_centroid(event_slice, half_att_win, False, attention_mechanism)
        # print_extreme_att_pos(dvs_res, median_x, median_y)
        # if label in [10]:
        # plotter.plot_attention_window_on_hist(event_slice, centroid_x, centroid_y, attention_window_size, i, save=True)
        event_slice = get_events_in_window(centroid_x, centroid_y, attention_window_size, event_slice)
        event_slice = shift_for_attention(event_slice, centroid_x, centroid_y)
        event_slice.loc[:, 'n_id'] = (event_slice.y * attention_window_size) + event_slice.x
        if event_slice.size > 0:
            min_n_id = event_slice['n_id'].min()
            assert min_n_id >= 0, 'n_id smaller than zero'
            max_n_id = event_slice['n_id'].max()
            assert max_n_id < attention_window_size * attention_window_size, 'n_id greater than input size'
        if input_window_position:
            x_dist, y_dist, input_pos_df = get_window_position_df(time_window, attention_window_time,
                                                                  centroid_x + half_att_win,
                                                                  centroid_y + half_att_win, max_neuron_id,
                                                                  frequency, attention_window_position_std,
                                                                  x_dist, y_dist, new_pos_weight)
            dfs.append(input_pos_df)
        if not only_input_position:
            dfs.append(event_slice)
    return pd.concat(dfs)


def get_centroid(event_slice, half_att_win, clip, attention_mechanism):
    if attention_mechanism == 'median':
        att_fun = np.median
    elif attention_mechanism == 'mean':
        att_fun = np.mean
    if clip:
        centroid_x = int(np.clip(att_fun(event_slice.x), half_att_win, 127 - half_att_win) - half_att_win)
        centroid_y = int(np.clip(att_fun(event_slice.y), half_att_win, 127 - half_att_win) - half_att_win)
    else:  # no clipping starting with and including Result 49
        centroid_x = int(att_fun(event_slice.x) - half_att_win)
        centroid_y = int(att_fun(event_slice.y) - half_att_win)
    return centroid_x, centroid_y


def print_extreme_att_pos(dvs_res, median_x, median_y):
    if median_x < 0 or median_x > dvs_res:
        print('median_x: {}'.format(median_x))
    if median_y < 0 or median_y > dvs_res:
        print('median_y: {}'.format(median_y))


def get_window_position_df(time_window, attention_window_time, x, y, max_n_id, frequency,
                           attention_window_position_std, old_x_norm, old_y_norm, new_pos_weight):
    std = attention_window_position_std
    curr_x_norm = stats.norm(x, std)
    curr_y_norm = stats.norm(y, std)
    ts = []
    n_id = []
    dvs_res = 128
    for i in xrange(dvs_res):
        x_norm_i = curr_x_norm.pdf(i) * new_pos_weight + old_x_norm[i] * (1 - new_pos_weight)
        y_norm_i = curr_y_norm.pdf(i) * new_pos_weight + old_y_norm[i] * (1 - new_pos_weight)
        assert (x_norm_i < 1 and y_norm_i < 1)
        x_ts = (np.random.sample(np.random.poisson(
            attention_window_time * x_norm_i * std * frequency)) * attention_window_time + time_window).tolist()
        y_ts = (np.random.sample(np.random.poisson(
            attention_window_time * y_norm_i * std * frequency)) * attention_window_time + time_window).tolist()
        ts += x_ts + y_ts
        n_id += [i + max_n_id] * len(x_ts)
        n_id += [i + max_n_id + dvs_res] * len(y_ts)
        old_x_norm[i] = x_norm_i
        old_y_norm[i] = y_norm_i
    df = pd.DataFrame({'ts': ts, 'n_id': n_id})
    return old_x_norm, old_y_norm, df


def find_perfect_window_coords(event_slice, attention_window_size):
    for x in xrange(128 - attention_window_size + 1):
        for y in xrange(128 - attention_window_size + 1):
            event_count = get_events_in_window(x, y, attention_window_size, event_slice).size
            if event_count > max_event_count:
                max_event_count = event_count
                max_x = x
                max_y = y
    return max_x, max_y


def get_events_in_window(x, y, attention_window_size, event_slice):
    return event_slice.loc[
        (event_slice.x >= x) & (event_slice.x < x + attention_window_size) & (event_slice.y >= y) & (
                event_slice.y < y + attention_window_size)]


def shift_for_attention(event_slice, median_x, median_y):
    event_slice.loc[:, 'x'] -= median_x
    event_slice.loc[:, 'y'] -= median_y
    return event_slice


def get_grouped_n_id(xaddr, yaddr, group_by):
    xaddr = np.array(xaddr) // group_by
    yaddr = np.array(yaddr) // group_by
    return (yaddr * (128 / group_by)) + xaddr


def get_label_spikes_df(label, max_neuron_id, first_ts, end_ts, frequency):
    label_spikes = get_label_spikes(first_ts, end_ts, frequency)
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
    labels = [int(labels[i]) for i in sample_ids]
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
    labels = [int(labels[i]) for i in sample_ids]
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
    if exp_directory == 'dvs_mnist_flash' or exp_directory == 'dvs_mnist_flash/noise' or exp_directory == 'dvs_gesture_split':
        path = glob.glob('data/{exp_dir}/{test_or_train}/*.aedat'.format(test_or_train=test_or_train,
                                                                         exp_dir=exp_directory))
    elif exp_directory == 'dvs_mnist_saccade':
        path = glob.glob('data/{exp_dir}/grabbed_data*/scale16/*.aedat'.format(exp_dir=exp_directory))
    return sorted(path)


def get_label_spikes(first_timestamp, last_timestamp, frequency):
    return np.sort(np.random.sample(np.random.poisson(last_timestamp * frequency))) * last_timestamp + first_timestamp


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

    return timestamps, xaddr, yaddr, pol, min_ts


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
