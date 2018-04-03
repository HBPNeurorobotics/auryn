import matplotlib.pyplot as plt
import numpy as np
import sys
import time
import file_io as fio
import jaer_data_handler as jhandler
import pandas as pd


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
        df, seek = fio.ras_to_df("inputs/{}/input.ras".format(path), local_start, local_end, seek)
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


def plot_2d_from_txt(path, title, start=0, end=9223372036854775807):
    bucket = np.loadtxt('plots/{path}_ras_input_{start}_to_{end}.txt'.format(path=path, start=start, end=end))
    plot_heat_map(bucket, title)


def plot_2d_from_aedat31(path, start=0, end=sys.maxint, image_title=''):
    timestamps, xaddr, yaddr, pol = jhandler.load_aedat31(path, debug=0)
    df = pd.DataFrame({'ts': timestamps, 'x': xaddr, 'y': yaddr, 'p': pol})
    df.ts = df.ts * 1e-6
    if end > max(df.ts):
        end = max(df.ts)
    df = df[(df.ts >= start) & (df.ts <= end)]
    bucket = np.zeros((128, 128), dtype=int)
    for event in df.itertuples():
        bucket[event.y][event.x] += 1
    plot_heat_map(bucket, 'Event count from {:0.2f}s to {:.2f}s'.format(start, end), save=True, image_title=image_title)


def plot_heat_map(bucket, plot_title, save=False, image_title=''):
    plt.clf()
    fig, ax = plt.subplots()
    cax = ax.imshow(bucket, cmap='viridis', interpolation='nearest', vmin=0, vmax=10)
    ax.set_title(plot_title)
    cbar = fig.colorbar(cax)
    if save:
        plt.savefig('{}.png'.format(image_title), dpi=700)
    else:
        plt.show()
    plt.close('all')


def plot_weight_matrix(path, connections=['vh', 'hh', 'ho'], save=False):
    num_subplots = len(connections)
    for i, connection in enumerate(connections):
        weight_matrix = fio.mtx_file_to_matrix(path.format(connection))
        plt.subplot(num_subplots, 1, i + 1)
        plt.gca().set_title(connection)
        cax = plt.imshow(weight_matrix, cmap='viridis', interpolation='nearest', aspect='auto')
        cbar = plt.colorbar(cax)
    plt.tight_layout()
    if save:
        plt.savefig('plots/weight_matrix_{}_{}.png'.format(connections, time.time()), dpi=700)
    else:
        plt.show()
    plt.close('all')


def plot_ras_spikes(pathinput, start, end, layers=['vis', 'hid', 'out'], res=sys.maxint, number_of_classes=10,
                    save=False):
    title = 'Spike times'
    plt.title(title)
    counter = 1
    num_plots = len(layers)
    if 'vis' in layers:
        num_plots += 1

    for layer in layers:
        path = pathinput.format(layer)
        data_df, seek = fio.ras_to_df(path, start, end)
        if counter == 1:
            ax1 = plt.subplot(num_plots, 1, counter)
        else:
            ax1 = plt.subplot(num_plots, 1, counter, sharex=ax1)
        if layer == 'vis':
            label_df = data_df.loc[data_df.n_id >= res]
            data_df = data_df.loc[data_df.n_id < res]
            for i in xrange(number_of_classes):
                plt.plot((start, end), (i, i), 'r--', linewidth=0.5)
            plt.plot(label_df.ts.values, label_df.n_id.values - res, linestyle='None', marker=u'|', color=[0, 0, 1, 1],
                     markersize=1)
            plt.ylabel("Label")
            counter += 1
            plt.subplot(num_plots, 1, counter, sharex=ax1)
        elif layer == 'hid':
            pass
        elif layer == 'out':
            for i in xrange(number_of_classes):
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


def plot_weight_stats(stats, save=False):
    num_subplots = len(stats.keys())
    if num_subplots == 3:
        keys = ['vh', 'hh', 'ho']
    elif num_subplots == 4:
        keys = ['vh', 'h1h2', 'h2h1', 'ho']
    for i, key in enumerate(keys):
        stat = stats[key]
        plt.subplot(num_subplots, 1, i + 1)
        plt.title(key)
        length = len(stat)
        plt.plot((0, length), (0, 0), 'r--', linewidth=0.5)
        plt.errorbar(range(length), [i[0] for i in stat], [i[1] for i in stat], linestyle='None', marker='o')
        plt.ylabel('weight')
    plt.xlabel('epoch')
    plt.tight_layout()
    if save:
        plt.savefig('plots/weight_stats_{}.png'.format(time.time()), dpi=700)
    else:
        plt.show()
    plt.close('all')


def plot_accuracy(acc_hist, save=False):
    x = [i[0] for i in acc_hist]
    y_rate = [i[1][0] for i in acc_hist]
    y_first = [i[1][1] for i in acc_hist]
    plt.plot(x, y_rate, marker='o')
    plt.plot(x, y_first, marker='o', color='r')
    plt.title('classification accuracy')
    plt.xlabel('epoch')
    plt.ylabel('accuracy in percent')
    plt.legend(['rate', 'first'])
    if save:
        plt.savefig('plots/acc_hist_{}.png'.format(time.time()), dpi=700)
    else:
        plt.show()
    plt.close('all')


def plot_weight_histogram(path, nh1, connections=['vh', 'hh', 'ho'], save=False):
    num_subplots = len(connections)
    for i, connection in enumerate(connections):
        weight_matrix = fio.mtx_file_to_matrix(path.format(connection))
        plt.subplot(num_subplots, 1, i + 1)
        plt.gca().set_title(connection)
        if connection == 'hh':
            weight_matrix = weight_matrix[:, nh1:]
        elif connection == 'ho':
            weight_matrix = weight_matrix[nh1:]
        weight_matrix = weight_matrix.flatten()
        cax = plt.hist(weight_matrix, 300)
    plt.tight_layout()
    if save:
        plt.savefig('plots/weight_histogram_{}_{}.png'.format(connections, time.time()), dpi=700)
    else:
        plt.show()
    plt.close('all')
