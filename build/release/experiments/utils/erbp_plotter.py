import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
import time
from memory_profiler import profile

from tqdm import tqdm
import pdb


def ras_to_df(filepath, start=0., end=sys.maxint, startseek=0):
    if isinstance(filepath, str):
        if filepath.find('*') > 0:
            import glob
            filenames = glob.glob(filepath)
        else:
            filenames = [filepath]
    else:
        assert hasattr(filepath, '__len__')
        filenames = filepath
    df = pd.DataFrame({'ts': [], 'n_id': []})
    seek = 0
    for filename in filenames:
        with open(filename, 'r') as f:
            f.seek(startseek)
            ret = []
            read = True
            while read:
                line = f.readline()
                sol = np.array(line.split(), dtype=float)
                if sol.size > 0 and sol[0] < end:
                    if start <= sol[0]:
                        ret.append(sol)
                else:
                    read = False
            reta = np.array(ret)
            seek = f.tell()
        if reta.size != 0:
            df = df.append(pd.DataFrame({'ts': reta[:, 0], 'n_id': reta[:, 1]}))
    return df, seek


def mtx_file_to_matrix(filepath):
    with open(filepath, 'r') as f:
        filelines = f.read().splitlines()
        dims = map(int, filelines[2].split())
        weight_matrix = np.zeros((dims[0], dims[1]), dtype=float)
        for line in filelines[3:]:
            words = line.split()
            weight_matrix[int(words[0]) - 1][int(words[1]) - 1] = float(words[2])
    return weight_matrix


def plot_2d_input_ras(path, dimension, start=0, end=sys.maxint):
    if start > end:
        print('start time has to be smaller than end time.')
        return
    res = dimension * dimension
    stepsize = 3000
    bucket = np.zeros((32, 32), dtype=int)
    labels = np.array([])
    seek = 0
    for i in xrange(end // stepsize + 1):
        local_start = start + i * stepsize
        local_end = (i + 1) * stepsize
        if end < local_end:
            local_end = end
        df, seek = ras_to_df("inputs/{}/input.ras".format(path), local_start, local_end, seek)
        if df.ts.values.size == 0:
            break
        label_df = df.loc[df.n_id >= res]
        labels = np.unique(np.append(labels, label_df.n_id.unique() - res)).astype(int)
        df2 = df.loc[df.n_id < res]
        for _, n_id in df2.n_id.iteritems():
            bucket[int(n_id // dimension)][int(n_id % dimension)] += 1
        del df
        del df2
        del label_df

    # np.savetxt('plots/{path}_ras_input_{start}_to_{end}.txt'.format(path=path, start=start, end=end), bucket)
    plot_heat_map(bucket, 'Spike count from {}s to {}s (labels: {})'.format(start, end, labels))


def plot_2d_from_file(path, title, start=0, end=9223372036854775807):
    bucket = np.loadtxt('plots/{path}_ras_input_{start}_to_{end}.txt'.format(path=path, start=start, end=end))
    plot_heat_map(bucket, title)


def plot_heat_map(bucket, title):
    plt.clf()
    fig, ax = plt.subplots()
    cax = ax.imshow(bucket, cmap='viridis', interpolation='nearest')
    ax.set_title(title)
    cbar = fig.colorbar(cax)
    plt.show()


# TODO Weightplot
def plot_weight_matrix(path, connections=['vh', 'hh', 'ho'], save=False):
    num_subplots = len(connections)
    for i, connection in enumerate(connections):
        weight_matrix = mtx_file_to_matrix(path.format(connection))
        plt.subplot(num_subplots, 1, i+1)
        plt.gca().set_title(connection)
        cax = plt.imshow(weight_matrix, cmap='viridis', interpolation='nearest', aspect='auto')
        cbar = plt.colorbar(cax)
    plt.tight_layout()
    if save:
        plt.savefig('plots/weight_matrix_{}_{}.png'.format(connections, time.time()), dpi=700)
    else:
        plt.show()
    plt.close('all')


def plot_ras_spikes(pathinput, start, end, layers=['vis', 'hid', 'out'], res=sys.maxint, save=False):
    title = 'Spike times'
    plt.title(title)
    counter = 1
    num_plots = len(layers)
    if 'vis' in layers:
        num_plots += 1

    for layer in layers:
        path = pathinput.format(layer)
        data_df, seek = ras_to_df(path, start, end)
        if counter == 1:
            ax1 = plt.subplot(num_plots, 1, counter)
        else:
            ax1 = plt.subplot(num_plots, 1, counter, sharex=ax1)
        if layer == 'vis':
            label_df = data_df.loc[data_df.n_id >= res]
            data_df = data_df.loc[data_df.n_id < res]
            for i in xrange(10):
                plt.plot((start, end), (i, i), 'r--', linewidth=0.5)
            plt.plot(label_df.ts.values, label_df.n_id.values - res, linestyle='None', marker=u'|', color=[0, 0, 1, 1],
                     markersize=1)
            plt.ylabel("Label")
            counter += 1
            plt.subplot(num_plots, 1, counter, sharex=ax1)
        elif layer == 'hid':
            pass
        elif layer == 'out':
            for i in xrange(10):
                plt.plot((start, end), (i, i), 'r--', linewidth=0.5)

        print(data_df.ts.values.size)
        plt.plot(data_df.ts.values, data_df.n_id.values, linestyle='None', marker=u'|', color=[0, 0, 1, 1],
                 markersize=1)
        plt.ylabel("Neuron id [{}]".format(layer))
        counter += 1

        x1, x2, y1, y2 = plt.axis()
        plt.axis((start, end, y1, y2))

    plt.xlabel("Time in seconds")
    plt.tight_layout()

    if save:
        plt.savefig('plots/{}_{}_{}_{}.png'.format(start, end, layers, time.time()), dpi=700)
    else:
        plt.show()
    plt.close('all')


# def plot_weight_convolution():
#    # vh = load vis to hidden weight matrix
#    # hh = load hid to hid weight matrix
#    # ho = load hid to out weight matrix
#    bucket = np.zeros((32, 32), dtype=float)
#    for output_neuron in xrange(10):
#        for id in vh.input_ids:
#            pixel_weight_sum = pixel_weight_sum(id, output_neuron, vh, hh, ho)
#            bucket[int(n_id // dimension)][int(n_id % dimension)] = pixel_weight_sum
