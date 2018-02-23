import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
from memory_profiler import profile

from tqdm import tqdm
import pdb


def ras_to_df(filepath, start=0., end=sys.maxint, seek=0):
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
    for filename in filenames:
        with open(filename, 'r') as f:
            f.seek(seek)
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


def plot_2d_input_ras(path, dimension, start=0, end=sys.maxint):
    if start > end:
        print('start time has to be smaller than end time.')
        return
    res = dimension*dimension
    stepsize = 3000
    bucket = np.zeros((32, 32), dtype=int)
    labels = np.array([])
    seek = 0
    for i in xrange(end // stepsize + 1):
        local_start = start + i*stepsize
        local_end = (i+1)*stepsize
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
    fig, ax = plt.subplots()
    cax = ax.imshow(bucket, cmap='viridis', interpolation='nearest')
    ax.set_title(title)
    cbar = fig.colorbar(cax)
    plt.show()
    plt.clf()


def plot_ras_spikes(path, start, end, layername, res=sys.maxint, save=False):
    df, seek = ras_to_df(path, start, end)
    label_df = df.loc[df.n_id >= res]
    data_df = df.loc[df.n_id < res]
    title = '{} layer spike times'.format(layername)

    if label_df.ts.size > 0:
        ax1 = plt.subplot(2, 1, 1)
    plt.plot(data_df.ts.values, data_df.n_id.values, linestyle='None', marker=u'|', color=[0, 0, 1, 1], markersize=1)
    plt.ylabel("Neuron id")
    plt.title(title)

    if label_df.ts.size > 0:
        plt.subplot(2, 1, 2, sharex=ax1)
        plt.plot(label_df.ts.values, label_df.n_id.values - res, linestyle='None', marker=u'|', color=[0, 0, 1, 1],
                 markersize=1)
        for i in xrange(10):
            plt.plot((start, end), (i, i), 'r--', linewidth=0.5)
        plt.ylabel("Label")
    plt.xlabel("Time in seconds")

    x1, x2, y1, y2 = plt.axis()
    plt.axis((start, end, y1, y2))

    if save:
        plt.savefig('{}_{}.png'.format(path, layername))
    else:
        plt.show()
        plt.clf()

# TODO Weightplot


def plot_ras_spikes_whole(pathinput, start, end, layers=['vis', 'hid', 'out'], res=sys.maxint, save=False):
    title = 'Spike times'
    plt.title(title)
    counter = 1
    numPlots = len(layers)
    if 'vis' in layers:
        numPlots += 1

    for layer in layers:
        path = pathinput.format(layer)
        print(path)
        df, seek = ras_to_df(path, start, end)
        label_df = df.loc[df.n_id >= res]
        data_df = df.loc[df.n_id < res]

        if label_df.ts.size > 0 or len(layers) > 1:
            if counter == 1:
                ax1 = plt.subplot(numPlots, 1, counter)
            else:
                ax1 = plt.subplot(numPlots, 1, counter, sharex=ax1)
        plt.plot(data_df.ts.values, data_df.n_id.values, linestyle='None', marker=u'|', color=[0, 0, 1, 1],
                 markersize=1)
        plt.ylabel("Neuron id [{}]".format(layer))

        counter += 1

        if label_df.ts.size > 0:
            plt.subplot(numPlots, 1, counter, sharex=ax1)  #
            plt.plot(label_df.ts.values, label_df.n_id.values - res, linestyle='None', marker=u'|', color=[0, 0, 1, 1],
                     markersize=1)
            for i in xrange(10):
                plt.plot((start, end), (i, i), 'r--', linewidth=0.5)
            plt.ylabel("Label")
            counter += 1

        x1, x2, y1, y2 = plt.axis()
        plt.axis((start, end, y1, y2))

    plt.xlabel("Time in seconds")

    if save:
        plt.savefig('{}_{}.png'.format(path, layers))
    else:
        plt.show()
        plt.clf()
