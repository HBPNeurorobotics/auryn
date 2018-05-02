import matplotlib.pyplot as plt
import numpy as np
import sys
import time
import file_io as fio
import jaer_data_handler as jhandler
import pandas as pd
from matplotlib.lines import Line2D
import glob


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


def plot_2d_hist_from_aedat31(pathname, start=0, end=sys.maxint, image_title=''):
    paths = glob.glob(pathname)
    bucket = np.zeros((128, 128), dtype=int)
    for path in paths:
        timestamps, xaddr, yaddr, pol = jhandler.load_aedat31(path, debug=1)
        df = pd.DataFrame({'ts': timestamps, 'x': xaddr, 'y': yaddr, 'p': pol})
        df.ts = df.ts * 1e-6
        if end > max(df.ts):
            end = max(df.ts)
        df = df[(df.ts >= start) & (df.ts <= end)]
        for event in df.itertuples():
            bucket[event.y][event.x] += 1
    plot_heat_map(bucket, 'Spatial event distribution - label 1'.format(start, end), save=True, image_title=image_title,
                  dynamic_v=True)


def plot_2d_events_from_aedat31(path, start=0, end=sys.maxint, image_title=''):
    timestamps, xaddr, yaddr, pol = jhandler.load_aedat31(path, debug=0)
    df = pd.DataFrame({'ts': timestamps, 'x': xaddr, 'y': yaddr, 'p': pol})
    df.ts = df.ts * 1e-6
    if end > max(df.ts):
        end = max(df.ts)
    df = df[(df.ts >= start) & (df.ts <= end)]
    bucket = np.zeros((128, 128), dtype=int)
    for event in df.itertuples():
        if event.p == 1:
            bucket[event.y][event.x] += 1
        elif event.p == 0:
            bucket[event.y][event.x] -= 1
    bucket[bucket > 0] = 1
    bucket[bucket < 0] = -1
    plot_heat_map(bucket, 'Events from {:0.2f}s to {:.2f}s'.format(start, end), save=True, image_title=image_title,
                  show_cbar=False, vmin=-1, vmax=1)


def plot_heat_map(bucket, plot_title, save=False, image_title='', show_cbar=True, vmin=0, vmax=10, dynamic_v=False):
    plt.clf()
    fig, ax = plt.subplots()
    if dynamic_v:
        cax = ax.imshow(bucket, cmap='viridis', interpolation='nearest')
    else:
        cax = ax.imshow(bucket, cmap='viridis', interpolation='nearest', vmin=vmin, vmax=vmax)
    ax.set_title(plot_title)
    if show_cbar:
        cbar = fig.colorbar(cax)
    else:
        custom_lines = [Line2D([0], [0], color=plt.cm.viridis(1.), lw=0, marker='s'),
                        Line2D([0], [0], color=plt.cm.viridis(-1.), lw=0, marker='s')]
        plt.legend(custom_lines, ['ON', 'OFF'], handlelength=0.5, borderpad=0.5, framealpha=0.5)
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
                plt.plot((start, end), (i, i), 'r--', linewidth=0.1, alpha=1)
            plt.plot(label_df.ts.values, label_df.n_id.values - res, linestyle='None', marker=u'|', color=[0, 0, 1, 1],
                     markersize=1, alpha=0.1)
            plt.ylabel("Label")
            counter += 1
            plt.subplot(num_plots, 1, counter, sharex=ax1)
            plt.plot(data_df.ts.values, data_df.n_id.values, linestyle='None', marker=u',', color=[0, 0, 1, 1])
            plt.ylabel("Input")
        elif layer == 'hid':
            plt.plot(data_df.ts.values, data_df.n_id.values, linestyle='None', marker=u',', color=[0, 0, 1, 1])
            plt.ylabel("Hidden")
        elif layer == 'out':
            for i in xrange(number_of_classes):
                plt.plot((start, end), (i, i), 'r--', linewidth=0.1, alpha=1)
            plt.plot(data_df.ts.values, data_df.n_id.values, linestyle='None', marker=u'|', color=[0, 0, 1, 1],
                     markersize=1, alpha=0.1)
            plt.ylabel("Output")

        print(data_df.ts.values.size)
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


def plot_accuracy_rate_first(acc_hist, save=False):
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


def plot_confusion_matrix(df_confusion, save=False):
    plt.imshow(df_confusion, cmap=plt.cm.viridis)
    plt.colorbar()
    tick_marks = np.arange(len(df_confusion.columns))

    plt.xticks(tick_marks, df_confusion.columns)
    plt.yticks(tick_marks, df_confusion.index)
    plt.tight_layout()
    plt.ylabel(df_confusion.index.name)
    plt.xlabel(df_confusion.columns.name)
    plt.gca().xaxis.set_label_position('top')
    plt.gca().xaxis.set_ticks_position('top')
    if save:
        plt.savefig('plots/confusion_matrix_{}.png'.format(time.time()), dpi=700, bbox_inches='tight')
    else:
        plt.show()
    plt.close('all')


def plot_weight_convolution(path, nh1, nc, connections=['vh', 'hh', 'ho'], save=False):
    weight_matrices = {}
    for connection in connections:
        weight_matrix = mtx_file_to_matrix("{path}/fwmat_{connection}.mtx".format(path=path, connection=connection))
        if connection == 'vh':
            weight_matrix = weight_matrix[:32 * 32]
        elif connection == 'hh':
            weight_matrix = weight_matrix[:, nh1:]
        elif connection == 'ho':
            weight_matrix = weight_matrix[nh1:]
        weight_matrices[connection] = weight_matrix
    conv_matrix = calc_conv(connections, weight_matrices)  # .reshape(32,32,12)
    # conv_matrix = np.array(map(lambda x: np.argmax(x), conv_matrix)).reshape(32,32)
    for i in range(nc):
        plt.clf()
        conv_label = np.array([item[i] for item in conv_matrix]).reshape(32, 32)

        fig, ax = plt.subplots()
        ax.set_title('Weight convolution for label {}'.format(i))
        cbar_tick_size = 5000
        conv_plot = ax.imshow(conv_label, cmap='PiYG', interpolation='nearest', vmin=-cbar_tick_size,
                              vmax=cbar_tick_size)
        cbar = fig.colorbar(conv_plot, ticks=[-cbar_tick_size, 0, cbar_tick_size])
        cbar.ax.set_yticklabels(['< -{}'.format(cbar_tick_size), '0', '> {}'.format(cbar_tick_size)])
        if save:
            plt.savefig('plots/convolution/weight_conv_{}.png'.format(time.time()), dpi=700)
        else:
            plt.show()
        plt.close('all')


def calc_conv(connections, weight_matrices):
    weight_matrix = weight_matrices[connections[0]]
    if len(connections) == 1:
        return weight_matrix
    else:
        connections = connections[1:]
        conv_vec = []
        for n_id in range(weight_matrix.shape[0]):
            conv_vec.append(sum(map(lambda x: x[0]*x[1], zip(weight_matrix[n_id], calc_conv(connections, weight_matrices)))))
        return conv_vec
