import rosbag
import pandas as pd
import numpy as np
import os
import errno
import yaml

import attention_mechanism.attention as attention

import random
import glob
from tqdm import tqdm
import rospy
import time
import pdb

import subprocess

def event_to_dict(ev):
    # rospy.logerr(ev)
    ret = {
        'x': ev.x,
        'y': ev.y,
        'p': ev.polarity,
        'ts': ev.ts.to_sec()
    }
    return ret


def state_to_dict(state):
    ret = {
        'ts': state.header.stamp.to_sec(),
        'jointA': state.position[0],
        'jointB': state.position[1],
        'jointC': state.position[2]
    }
    return ret


def rosbag_to_df(filename, topics, joint_state_topic="/head/joint_states"):
    all_events = []
    joint_states = []
    try:
        opened_bag = rosbag.Bag(filename, 'r')
    except rosbag.ROSBagUnindexedException as e:
        print('Bag {} is unindexed. Reindexing'.format(filename))
        reindex_cmd = "rosbag reindex {} -f".format(filename)
        process = subprocess.Popen(reindex_cmd.split(), stdout=subprocess.PIPE)
        output, error = process.communicate()
        print(output, error)
        opened_bag = rosbag.Bag(filename, 'r')

    with opened_bag as bag:
        bag_topics = bag.get_type_and_topic_info()[1].keys()
        if bag.get_message_count(topics) <= 0:
            raise ValueError('Rosbag {} topic {} is empty.\nPossible topics: {}'
                             .format(filename, topics, bag_topics))
        for topics, msg, t in bag.read_messages(topics=topics):
            all_events += map(event_to_dict, msg.events)
        if joint_state_topic in bag_topics:
            for topics, msg, t in bag.read_messages(topics=[joint_state_topic]):
                joint_states.append(state_to_dict(msg))

    df = pd.DataFrame(all_events)
    if joint_states:
        joint_state_df = pd.DataFrame(joint_states)
        microsaccade_ts = 0
        for i in range(len(joint_state_df.index)):
            if different_joint_state(joint_state_df.loc[i], joint_state_df.loc[i + 1]):
                microsaccade_ts = joint_state_df.loc[i].ts + 0.05
                break
        df = df.loc[df.ts > microsaccade_ts, :]
    df.ts -= df.ts.min()
    df = df.loc[df.ts < 0.7, :]

    return df


def different_joint_state(js_1, js_2):
    return js_1.jointA != js_2.jointA or js_1.jointB != js_2.jointB or js_1.jointC != js_2.jointC


def get_grouped_n_id(xaddr, yaddr, orig_res, new_res):
    assert (orig_res % new_res == 0), 'Original resolution has to be a multiple of the new resolution.'
    group_by = orig_res // new_res
    xaddr = np.array(xaddr) // group_by
    yaddr = np.array(yaddr) // group_by
    return (yaddr * new_res) + xaddr  # + 1


def spike_times_from_ras(ras_path, nvis, nc, offset = 0):
    df = read_ras_to_df(ras_path)

    # the first neurons are input layer (visible), then comes the class neurons
    input_spike_times = [ (df['ts'][ df['n_id'] == i ] * 1000. + offset).astype(int).tolist() for i in range(nvis)]
    class_spike_times = [ (df['ts'][ df['n_id'] == i + nvis ] * 1000. + offset).astype(int).tolist() for i in range(nc)]

    return input_spike_times, class_spike_times


def read_ras_to_df(ras_path):
    with open(ras_path, "r") as f:
        lines = f.readlines()
        ts_id_list = [ (float(line.split(' ')[0]), int(line.split(' ')[1])) for line in lines]
        df = pd.DataFrame(ts_id_list, columns = ['ts', 'n_id'])
    return df


def read_ras_to_df(ras_path):
    with open(ras_path, "r") as f:
        lines = f.readlines()
        ts_id_list = [ (float(line.split(' ')[0]), int(line.split(' ')[1])) for line in lines]
        df = pd.DataFrame(ts_id_list, columns = ['ts', 'n_id'])
    return df


def write_df_to_ras(df, ras_path):
    with open(ras_path, "a+") as f:
        f.write(gen_ras_string(df))


def gen_ras_string(df):
    s = ''.join(["%f %d\n" % (ts, idx) for ts, idx in zip(df['ts'].values, df['n_id'].values)])
    return s


def silentremove(filename):
    try:
        os.remove(filename)
    except OSError as e:
        if e.errno != errno.ENOENT:
            raise


def get_rosbag_names(path_to_pkg, data_dir):
    return sorted(glob.glob('{}/scripts/data/{}/*/*.bag'.format(path_to_pkg, data_dir)))


def get_exp_data(config, test_or_train, number_of_classes=3):
    ids = []
    total_n_samples = config['max_samples_train'] + config['max_samples_test']
    samples_per_class = total_n_samples / number_of_classes
    train_samples_per_class = config['max_samples_train'] / number_of_classes
    if test_or_train == 'train':
        for i in range(0, total_n_samples, samples_per_class):
            ids += range(i, i + train_samples_per_class)
        sample_ids = random.sample(ids, config['n_samples_train'])
    else:
        for i in range(0, total_n_samples, samples_per_class):
            ids += range(i + train_samples_per_class, i + samples_per_class)
        sample_ids = ids[:config['n_samples_test']]
    labels = np.zeros(total_n_samples, dtype=int)
    for i in range(number_of_classes):
        labels[i * samples_per_class:(i + 1) * samples_per_class] = i
    labels = [int(labels[i]) for i in sample_ids]
    return labels, sample_ids


def get_label_spikes_df(label, max_neuron_id, first_ts, end_ts, frequency):
    label_spikes = get_label_spikes(first_ts, end_ts, frequency)
    df2 = pd.DataFrame({'ts': label_spikes, 'n_id': label + max_neuron_id})
    return df2


def get_label_spikes(first_timestamp, last_timestamp, frequency):
    return np.sort(np.random.sample(np.random.poisson(last_timestamp * frequency))) * last_timestamp + first_timestamp


def calc_max_neuron_id(config):
    if config['use_attention_window']:
        max_n_id = config['att_win_size'] * config['att_win_size']
    else:
        max_n_id = config['down_sample_res'] * config['down_sample_res']
    if config['event_polarity']:
        max_n_id *= 2
    return max_n_id * len(config['topics'])


def create_single_rosbag_ras(exp_dir, path_to_pkg, rosbag_path):
    config = get_config(path_to_pkg)
    att_time_frame, att_win_size, down_sample_res, dvs_res, event_polarity, label_freq, online_median_event_amount, pause_duration_test, pause_duration_train, topics, use_attention_window, use_online_median = read_config(
        config)
    ras_path = '{path_to_pkg}/scripts/inputs/{exp_dir}/predict/input.ras'.format(path_to_pkg=path_to_pkg,
                                                                                 exp_dir=exp_dir)
    silentremove(ras_path)

    input_df, max_neuron_id = get_input_df(att_time_frame, att_win_size, down_sample_res, dvs_res,
                                           event_polarity, rosbag_path,
                                           online_median_event_amount, topics,
                                           use_attention_window, use_online_median)
    assert max(list(input_df.n_id)) < max_neuron_id + 1, '{} > {}'.format(max(list(input_df.n_id)), max_neuron_id)
    write_df_to_ras(input_df, ras_path)
    return input_df.ts.values[-1]


def create_batch_ras(path_to_pkg, exp_dir, data_dir, test_or_train, nc, cache=True):
    config = get_config(path_to_pkg)
    att_time_frame, att_win_size, down_sample_res, dvs_res, event_polarity, label_freq, online_median_event_amount, pause_duration_test, pause_duration_train, topics, use_attention_window, use_online_median = read_config(
        config)
    if test_or_train == 'train':
        pause_duration = pause_duration_train
    else:
        pause_duration = pause_duration_test
    max_neuron_id = 0

    sample_names = get_rosbag_names(path_to_pkg, data_dir)

    labels, sample_ids = get_exp_data(config, test_or_train, nc)
    sample_names = [sample_names[i] for i in sample_ids]
    ras_path = '{path}/scripts/inputs/{exp_dir}/{test_or_train}/input.ras'.format(path=path_to_pkg, exp_dir=exp_dir,
                                                                                  test_or_train=test_or_train)
    silentremove(ras_path)

    current_timestamp = 0.
    sample_duration_list = []
    with pd.HDFStore(
            '{path}/scripts/data/{data_dir}/{test_or_train}_{event_pol}_{down_sample_res}_{attention}attention{attention_window_size}_online{online_att}.h5'.format(
                path=path_to_pkg,
                data_dir=data_dir,
                exp_dir=exp_dir,
                test_or_train=test_or_train,
                event_pol=event_polarity,
                down_sample_res=down_sample_res,
                attention=use_attention_window,
                attention_window_size=att_win_size,
                online_att=use_online_median
            )) as store:
        for i, sample_id in enumerate(tqdm(sample_ids)):
            key = 'm{mod}/s{sample_id}'.format(sample_id=sample_id,
                                               mod=sample_id % 10)
            if cache and key in store:
                input_df = store[key]
            else:
                input_df, max_neuron_id = get_input_df(att_time_frame, att_win_size, down_sample_res, dvs_res,
                                                       event_polarity, sample_names[i],
                                                       online_median_event_amount, topics,
                                                       use_attention_window, use_online_median)
                if cache:
                    store[key] = input_df
            if not max_neuron_id:
                max_neuron_id = calc_max_neuron_id(config)
            assert max(list(input_df.n_id)) < max_neuron_id + 1, '{} > {}'.format(max(list(input_df.n_id)),
                                                                                  max_neuron_id)
            label_df = get_label_spikes_df(labels[i], max_neuron_id, 0, list(input_df.ts)[-1], label_freq)
            input_df = input_df.append(label_df, sort=False)
            input_df = input_df.sort_values('ts')
            input_df.ts = input_df.ts.add(current_timestamp)
            write_df_to_ras(input_df, ras_path)
            current_timestamp = input_df.ts.values[-1] + pause_duration
            sample_duration_list.append(current_timestamp)
    return sample_duration_list, labels


def read_config(config):
    topics = config['topics']
    dvs_res = config['dvs_res']
    att_win_size = config['att_win_size']
    use_attention_window = config['use_attention_window']
    use_online_median = config['use_online_median']
    online_median_event_amount = config['online_median_event_amount']
    att_time_frame = config['att_time_frame']
    down_sample_res = config['down_sample_res']
    event_polarity = config['event_polarity']
    label_freq = config['label_freq']
    pause_duration_train = config['sample_pause_train']
    pause_duration_test = config['sample_pause_test']
    return att_time_frame, att_win_size, down_sample_res, dvs_res, event_polarity, label_freq, online_median_event_amount, pause_duration_test, pause_duration_train, topics, use_attention_window, use_online_median


def get_config(path_to_pkg):
    with open('{path_to_pkg}/config/preprocessing_config.yaml'.format(path_to_pkg=path_to_pkg)) as config_file:
        config = yaml.safe_load(config_file)
    return config


def get_input_df(att_time_frame, att_win_size, down_sample_res, dvs_res, event_polarity, path_to_rosbag,
                 online_median_event_amount, topics, use_attention_window,
                 use_online_median, crop=True):
    dfs = []
    max_id_per_cam = 0
    for topic in topics:
        bag_events_df = rosbag_to_df(path_to_rosbag, [topic])
        assertEqualCoordRange(bag_events_df)
        if use_attention_window:
            if use_online_median:
                bag_events_df = attention.get_attention_df_rolling(bag_events_df,
                                                                   online_median_event_amount, att_win_size)
            else:
                bag_events_df = attention.get_attention_df_offline(bag_events_df, att_time_frame,
                                                                   att_win_size, np.median)
            max_id_per_cam = att_win_size * att_win_size
        elif crop:
            # translate all events by lower corner address of the attention window
            bag_events_df = attention.take_window_events(down_sample_res,
                                                         pd.DataFrame({'x': [dvs_res / 2] *
                                                                            bag_events_df.shape[0],
                                                                       'y': [dvs_res / 2] *
                                                                            bag_events_df.shape[0]}),
                                                         bag_events_df)
            max_id_per_cam = down_sample_res * down_sample_res
        else:
            bag_events_df.loc[:, 'n_id'] = get_grouped_n_id(bag_events_df.x, bag_events_df.y, dvs_res,
                                                            down_sample_res)
            max_id_per_cam = down_sample_res * down_sample_res
        dfs.append(bag_events_df)
    max_neuron_id = max_id_per_cam
    if len(dfs) == 2:
        dfs[1].loc[:, 'n_id'] += max_id_per_cam
        max_neuron_id *= 2
        input_df = pd.concat(dfs)
        input_df.sort_values(by=['ts'], inplace=True)
    else:
        input_df = dfs[0]
    if event_polarity == 'on':
        input_df = input_df[input_df.p == 1]
    elif event_polarity == 'off':
        input_df = input_df[input_df.p == 0]
    elif event_polarity == 'dual':
        input_df.loc[input_df.p == 1, 'n_id'] += max_neuron_id
        max_neuron_id *= 2
    return input_df, max_neuron_id


def assertEqualCoordRange(bag_events_df):
    max_x = max(list(bag_events_df.x))
    min_x = min(list(bag_events_df.x))
    max_y = max(list(bag_events_df.y))
    min_y = min(list(bag_events_df.y))
    assert min_x >= 0 and max_x < 128 and min_y >= 0 and max_y < 128, 'Events out of range. min_x: {min_x} max_x: {max_x} min_y: {min_y} max_y {max_y}'.format(
        min_x=min_x, max_x=max_x, min_y=min_y, max_y=max_y)
